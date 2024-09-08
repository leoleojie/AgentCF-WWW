from logging import getLogger
import os
from typing import Dict, List, Optional, Union
import asyncio
from pydantic import BaseModel, Field

from agentverse.llms.base import LLMResult

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs
import time
import openai
from openai.error import OpenAIError
logger = getLogger()

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.proxy = os.environ.get("http_proxy")
openai.proxy = os.environ.get("http_proxy")
openai.api_base = os.environ.get("api_base")
if openai.proxy is None:
    openai.proxy = os.environ.get("HTTP_PROXY")
if openai.api_key is None:
    logger.info(
        "OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY"
    )
    is_openai_available = False
else:
    is_openai_available = True


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


class OpenAICompletionArgs(OpenAIChatArgs):
    model: str = Field(default="text-davinci-003")
    best_of: int = Field(default=1)


@llm_registry.register("text-davinci-003")
class OpenAICompletion(BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)
    api_key_list: list = []
    current_key_idx = 0
    def __init__(self, max_retry: int = 15, **kwargs):
        args = OpenAICompletionArgs()
        args = args.dict()
        
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        # if len(kwargs) > 0:
        #     logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)
        self.api_key_list = kwargs.pop('api_key_list')
        # self.current_api_idx = 0

    def generate_response(self, prompt: str) -> LLMResult:
        response = openai.Completion.create(prompt=prompt, **self.args.dict())
        return LLMResult(
            content=response["choices"][0]["text"],
            send_tokens=response["usage"]["prompt_tokens"],
            recv_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

    async def agenerate_response(self, prompt: str) -> LLMResult:
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                openai.api_key = self.api_key_list[self.current_key_idx]
                response = await openai.Completion.acreate(prompt=prompt, **self.args.dict())
        # print(response)
                return [i['text'] for i in response['choices']]
            # except Exception as e:
            #     print(e)
            #     logger.info("rate limit error \n Retrying...")
            #     self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
            #     time.sleep(30)

            except Exception as e:
                if 'You exceeded your current quota, please check your plan and billing details.' in str(e) or 'The OpenAI account associated with this API key has been deactivated.' in str(e):
                    print("bill error")
                    print(openai.api_key)
                    if openai.api_key in self.api_key_list:
                        self.api_key_list.remove(openai.api_key)
                    continue
                if "This model's maximum context length" in str(e):
                    print(prompt)
                    print("length error")
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(20)
                continue

        # except Exception as e:
        #     print(e)
        #     print(openai.api_key)
        #     if openai.api_key in self.api_key_list:
        #         self.api_key_list.remove(openai.api_key)
        
    
@llm_registry.register("embedding")
class OpenAIEmbedding(BaseCompletionModel):
    args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)
    api_key_list: list = []
    current_key_idx = 0
    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAICompletionArgs()
        args = args.dict()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logger.info(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)
        self.api_key_list = kwargs.pop('api_key_list')

    def generate_response(self, prompt: str) -> LLMResult:
        response = openai.Embedding.create(input=prompt, **self.args.dict())
        return LLMResult(
            content=response["data"][0]["embedding"],
        )

    async def agenerate_response(self, sentences: str) -> LLMResult:

        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                openai.api_key = self.api_key_list[self.current_key_idx]
                response = [openai.Embedding.acreate(input=sentence, model="text-embedding-ada-002") for sentence in sentences]
                return await asyncio.gather(*response)
            except Exception as e:
                if 'You exceeded your current quota, please check your plan and billing details.' in str(e) or 'The OpenAI account associated with this API key has been deactivated.' in str(e) :
                    print("bill error")
                    print(openai.api_key)
                    if openai.api_key in self.api_key_list:
                        self.api_key_list.remove(openai.api_key)
                    continue
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(20)
                continue


@llm_registry.register("gpt-3.5-turbo-16k-0613")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)
    api_key_list: list = []
    current_key_idx = 0
    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()

        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        # if len(kwargs) > 0:
        #     logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)
        self.api_key_list = kwargs.pop('api_key_list')

    def _construct_messages(self, prompts: list):
        messages = []
        for prompt in prompts:
            messages.append([
                # {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent date: [2023-07]"},
                            {"role": "user", "content": prompt}])
        return messages


    def generate_response(self, prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt)
        try:
            response = openai.ChatCompletion.create(
                messages=messages, **self.args.dict()
            )
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        return LLMResult(
            content=response["choices"][0]["message"]["content"],
            send_tokens=response["usage"]["prompt_tokens"],
            recv_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
        )

    async def agenerate_response(self, prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt)
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                openai.api_key = self.api_key_list[self.current_key_idx]
                response = [openai.ChatCompletion.acreate(
                    messages = x, **self.args.dict()
                ) for x in messages]
                return await asyncio.gather(*response)
            except Exception as e:
                if 'You exceeded your current quota, please check your plan and billing details.' in str(e) or 'The OpenAI account associated with this API key has been deactivated.' in str(e):
                    print("bill error")
                    print(str(e))
                    print(openai.api_key)
                    if openai.api_key in self.api_key_list:
                        self.api_key_list.remove(openai.api_key)
                    continue

                logger.info(e)
                logger.info("Retrying...")
                time.sleep(20)
                continue


    async def agenerate_response_without_construction(self, messages: str) -> LLMResult:
        while True:
            try:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_key_list)
                openai.api_key = self.api_key_list[self.current_key_idx]
                response = [openai.ChatCompletion.acreate(
                    messages = x, **self.args.dict()
                ) for x in messages]
                return await asyncio.gather(*response)
            except Exception as e:
                if 'You exceeded your current quota, please check your plan and billing details.' in str(e) or 'The OpenAI account associated with this API key has been deactivated.' in str(e) :
                    print("bill error")
                    print(e)
                    print(openai.api_key)
                    if openai.api_key in self.api_key_list:
                        self.api_key_list.remove(openai.api_key)
                    continue

                logger.info(e)
                logger.info("Retrying...")
                time.sleep(20)
                continue
            
