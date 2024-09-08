from typing import List

from pydantic import Field

from agentverse.message import Message

from . import memory_registry
from .base import BaseMemory


@memory_registry.register("chat_history")
class ChatHistoryMemory(BaseMemory):
    # 每一条messages就是一条string
    messages: List[str] = Field(default=[])

    def add_message(self, messages: List[str]) -> None:
        for message in messages:
            self.messages.append(message)

    def to_string(self, add_sender_prefix: bool = False) -> str:
        
        return "\n".join(self.messages)

    def reset(self) -> None:
        self.messages = []


