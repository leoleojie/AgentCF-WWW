

import asyncio
import torch
import torch.nn as nn
from logging import getLogger
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import os.path as osp
import os
from agentverse.initialization import load_agent,  prepare_task_config
from fuzzywuzzy import process
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import random
from itertools import chain
import numpy as np
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

class AgentCF(SequentialRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(AgentCF, self).__init__(config, dataset)
        self.n_users = dataset.num(self.USER_ID)
        self.config = config
        self.sample_num = config['sample_num']
        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.logger = getLogger()
        self.loss = BPRLoss()
        self.item_token_id = dataset.field2token_id['item_id']
        self.item_id_token = dataset.field2id_token['item_id']
        self.user_id_token = dataset.field2id_token['user_id']
        self.user_token_id = dataset.field2token_id['user_id']
        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.api_batch = config['api_batch']
        self.chat_api_batch = config['chat_api_batch']
        embedding_context = \
            {'agent_type': 'embeddingagent',
             #  'role_description': dict(),
             'role_task': '',
             'memory': [],
             'prompt_template': '',
             'llm': {'model': self.config['embedding_model'], 'temperature': self.config['llm_temperature'],
                     'max_tokens': self.config['max_tokens'], 'llm_type': 'embedding',
                     'api_key_list': self.config['api_key_list'], 'current_key_idx': self.config['current_key_idx']},
             'llm_chat': {'model': 'gpt-3.5-turbo-16k-0613', 'llm_type': 'gpt-3.5-turbo-16k-0613',
                          'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens_chat'],
                          'api_key_list': self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
             'agent_mode': 'embedding', 'output_parser_type': 'recommender',

             }
        self.embedding_agent = load_agent(embedding_context)
        self.item_text = self.load_text()
        self.user_context = self.load_user_context()
        self.item_context = self.load_item_context()
        self.max_his_len = config['max_his_len']
        self.record_idx = 0



        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            if os.path.exists(path):
                self.record_idx += 1
                continue
            else: break

        print(f"In this interaction, the updation process is recorded in {str(self.record_idx)}")
        self.user_agents = {}
        for user_id, user_context in self.user_context.items():
            agent = load_agent(user_context)
            self.user_agents[user_id] = agent
            user_id = str(user_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}',)
            user_description = user_context['memory_1'][-1]
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path,f'user.{user_id}'),'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The user wrote the following self-description as follows: {user_description}\n')


        self.item_agents = {}
        item_descriptions = []

        for item_id, item_context in self.item_context.items():
            agent = load_agent(item_context)
            self.item_agents[item_id] = agent
            item_id = str(item_id)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}',)
            item_description = item_context['role_description']
            item_descriptions.append(item_description)
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path,f'item.{item_id}'),'w') as f:
                f.write('~'*20 + 'Meta information' + '~'*20 + '\n')
                f.write(f'The item has the following characteristics: {item_description} \n')


        rec_context = \
            {'agent_type':'recagent',
             'memory':[],
            'prompt_template': self.config['system_prompt_template'],
            'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'], 'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx'], },
            'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature_test'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx'],},
            'agent_mode':'system','output_parser_type':'recommender',
            'system_prompt_template_backward': self.config['system_prompt_template_backward'],
            'system_prompt_template_evaluation_basic': self.config['system_prompt_template_evaluation_basic'],
             'system_prompt_template_evaluation_sequential': self.config['system_prompt_template_evaluation_sequential'],
             'system_prompt_template_evaluation_retrieval': self.config['system_prompt_template_evaluation_retrieval'],
            'n_users': self.n_users
            }
        self.rec_agent = load_agent(rec_context)





    def load_user_context(self):
        user_context = {}
        user_context[0] = {'agent_type':'useragent', 'role_description':{'age': '[PAD]', 'user_gender': '[PAD]','user_occupation':'[PAD]'},'memory_1':['[PAD]'],'update_memory':['[PAD]'],
                     'role_description_string_1':'[PAD]','role_description_string_3':'[PAD]', 'role_task':'[PAD]','prompt_template': self.config['user_prompt_template'], 'user_prompt_system_role': self.config['user_prompt_system_role'],
                     'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                     'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                     'agent_mode':'user','output_parser_type':'useragent','historical_interactions':[], 'user_prompt_template_true': self.config['user_prompt_template_true']}
        feat_path = None
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.user')
        if feat_path != None :
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    user_id, user_age, user_gender, user_occupation,_ = line.strip().split('\t')
                    if user_id not in self.user_token_id:
                        continue
                    if user_occupation == 'other':
                        user_occupation_des = ' movie enthusiast'
                    else:
                        user_occupation_des = user_occupation
                    if user_gender == 'M':
                        user_gender_des = 'man'
                    else:
                        user_gender_des = 'woman'

                    user_context[self.user_token_id[user_id]] = \
                        {'agent_type':'useragent',
                        'role_description': {'age': user_age, 'user_gender': user_gender,'user_occupation':user_occupation},
                        'role_description_string_3': f'The user is a {user_gender_des}. The user is a {user_occupation_des}. ',
                        'role_description_string_1': f'I am a {user_gender_des}. I am a {user_occupation_des}.',
                         'user_prompt_system_role': self.config['user_prompt_system_role'],
                        'memory_1': [f' I am a {user_gender_des}. I am a {user_occupation_des}.',],
                        'update_memory': [f' I am a {user_gender_des}. I am a {user_occupation_des}.',],
                        'prompt_template': self.config['user_prompt_template'],

                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'user','output_parser_type':'useragent','historical_interactions':[], 'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context
        else:
            for user_id in range(self.n_users):
                user_context[user_id] = \
                        {'agent_type':'useragent',
                        'role_description': {},
                        'role_description_string_3': f'This user enjoys listening CDs very much.',
                        'role_description_string_1': f'I enjoy listening to CDs very much.',
                         'user_prompt_system_role': self.config['user_prompt_system_role'],
                        'memory_1': [f' I enjoy listening to CDs very much.',],
                        'update_memory': [f' I enjoy listening to CDs very much.', ],
                        'prompt_template': self.config['user_prompt_template'],

                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'user','output_parser_type':'useragent','historical_interactions':[], 'user_prompt_template_true': self.config['user_prompt_template_true']}
            return user_context



    def load_item_context(self):
        item_context = {}
        item_context[0] = {'agent_type':'itemagent', 'role_description':{'item_title': '[PAD]', 'item_release_year': '[PAD]','item_class':'[PAD]'},'memory':['[PAD]'],'memory_embedding':{},'update_memory':['[PAD]'], 'item_prompt_template_true': self.config['item_prompt_template_true'],
                     'role_description_string':'[PAD]', 'role_task':'[PAD]', 'prompt_template': self.config['user_prompt_template'],
                     'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                     'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                     'agent_mode':'user','output_parser_type':'itemagent'}
        feat_path = None
        init_item_descriptions = []
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, item_title, item_release_year, item_class = line.strip().split('\t')
                    if item_id not in self.item_token_id:
                        continue
                        role_description_string = f'The movie is called {self.item_text[self.item_token_id[item_id]]}. The theme of this movie is about {item_class}.'
                    item_context[self.item_token_id[item_id]] = \
                        {'agent_type':'itemagent',
                         'update_memory':[role_description_string],
                        'role_description':{'item_title': self.item_text[self.item_token_id[item_id]], 'item_class':item_class},
                        'role_description_string': role_description_string,
                        'prompt_template': self.config['item_prompt_template'],
                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'item',
                         'item_prompt_template_true': self.config['item_prompt_template_true'],
                        'output_parser_type':'itemagent'}
                    init_item_descriptions.append(role_description_string)
            # return item_context
        else:
            feat_path = osp.join(self.data_path, f'CDs.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    try:
                        item_id, item_title, item_class = line.strip().split('\t')
                    except ValueError:
                        item_id, item_title = line.strip().split('\t')
                        item_class = 'CDs'
                    if item_id not in self.item_token_id:
                        continue
                    role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'. The category of this CD is: '{item_class}'."
                    # role_description_string = f"The CD is called '{self.item_text[self.item_token_id[item_id]]}'."
                    item_context[self.item_token_id[item_id]] = \
                        {'agent_type':'itemagent',
                         'update_memory': [role_description_string],
                        'role_description':{'item_title': self.item_text[self.item_token_id[item_id]], 'item_class':item_class},
                        'role_description_string': role_description_string,
                        'prompt_template': self.config['item_prompt_template'],
                         'item_prompt_template_true': self.config['item_prompt_template_true'],
                        'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                        'agent_mode':'item',
                        'output_parser_type':'itemagent'}
                    init_item_descriptions.append(role_description_string)

            # return item_context
        if self.config['evaluation'] == 'rag':
            init_item_description_embeddings = self.generate_embedding(init_item_descriptions)
            for i, item in enumerate(item_context.keys()):
                if item == 0: continue
                item_context[item]['memory_embedding'] = {init_item_descriptions[i-1]: init_item_description_embeddings[i-1]}
        else:
            for i, item in enumerate(item_context.keys()):
                if item == 0: continue
                item_context[item]['memory_embedding'] = {init_item_descriptions[i - 1]: None}

        return item_context





    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        if 'ml-' in self.dataset_name:
            feat_path = osp.join(self.data_path, f'ml-100k.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.item_id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        else:
            feat_path = osp.join(self.data_path, f'CDs.item')
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    try:
                        item_id, movie_title, genre = line.strip().split('\t')
                    except ValueError:
                        print(line)
                        item_id, movie_title = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.item_id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text




    def generate_embedding(self, embedding_contents):
        batch_size = len(embedding_contents)
        embeddings = []

        for i in range(0, batch_size, self.api_batch):
            embeddings += asyncio.run(self.embedding_agent.llm.agenerate_response(embedding_contents[i:i+self.api_batch]))

        embeddings = [_["data"][0]["embedding"] for _ in embeddings]
        embeddings = torch.Tensor(embeddings).to(self.device) # batch_size, embedding_size
        embeddings = embeddings / embeddings.norm(p=2,dim=-1, keepdim=True)

        return embeddings

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        batch_size = batch_user.size(0)
        user_descriptions, pos_item_descriptions, neg_item_descriptions = [], [], []
        for i, user in enumerate(batch_user):
            user_agent = self.user_agents[int(user)]
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            user_descriptions.append(user_agent.update_memory[-1])
            pos_item_descriptions.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions.append(neg_item_agent.update_memory[-1])

        system_forward_prompts = [self.rec_agent.astep_forward(int(batch_user[i]), user_descriptions[i], pos_item_descriptions[i], neg_item_descriptions[i]) for i in range(batch_size)]
        system_responses = []
        for i in range(0, batch_size, self.api_batch):
            system_responses += asyncio.run(self.rec_agent.llm.agenerate_response(system_forward_prompts[i:i + self.api_batch]))
        system_responses = [self.rec_agent.output_parser.parse(response['choices'][0]['message']['content']) for response in system_responses]
        system_selections, system_reasons = [], []
        for response in system_responses:
            system_selections.append(response[0])
            system_reasons.append(response[1])
        return system_selections, system_reasons


    def backward(self, system_reasons, batch_user, batch_pos_item, batch_neg_item):
        batch_size = len(batch_user)
        pos_item_descriptions_forward, neg_item_descriptions_forward, pos_item_titles, neg_item_titles, user_descriptions_forward = [], [], [], [], []
        for i, user in enumerate(batch_user):
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            pos_item_titles.append(pos_item_agent.role_description['item_title'])
            neg_item_titles.append(neg_item_agent.role_description['item_title'])
            pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
            user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

        user_backward_prompts = [self.user_agents[int(batch_user[i])].astep_backward(system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i], neg_item_descriptions_forward[i]) for i in range(batch_size)]
        user_update_descriptions = []
        for i in range(0, batch_size, self.chat_api_batch):
            user_update_descriptions += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(user_backward_prompts[i:i+self.chat_api_batch]))

        user_update_descriptions = [self.user_agents[0].output_parser.parse_update(response["choices"][0]["message"]["content"]) for response in user_update_descriptions]
        for i, user in enumerate(batch_user):
            self.user_agents[int(user)].update_memory.append(user_update_descriptions[i])
        print("*"*10 + "User Update Is Over!" + "*"*10 + '\n')

        item_backward_prompts = [
            self.item_agents[int(batch_pos_item[i])].astep_backward(
                system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i],
                neg_item_descriptions_forward[i], user_update_descriptions[i])
            for i in range(batch_size)]

        item_update_memories = []
        for i in range(0, batch_size, self.chat_api_batch):
            item_update_memories += asyncio.run(
                self.item_agents[0].llm_chat.agenerate_response(item_backward_prompts[i:i + self.chat_api_batch]))

        item_update_memories = [self.item_agents[0].output_parser.parse(response["choices"][0]["message"]["content"]) for response in item_update_memories]


        # 录入到item agent memory中
        for i in range(batch_size):
            if len(item_update_memories[i]) != 2:
                print("*" * 10 + "item update 出现 bug" + "*" * 10 + '\n')
            else:
                self.item_agents[int(batch_pos_item[i])].update_memory.append(item_update_memories[i][1])
                # TODO: whether to update negative item?
                if self.config['update_neg_item']: self.item_agents[int(batch_neg_item[i])].update_memory.append(item_update_memories[i][0])
        print("*" * 10 + "Item Update Is Over!" + "*" * 10 + '\n')
        self.logging_during_updation(batch_user, system_reasons, user_backward_prompts, pos_item_descriptions_forward, neg_item_descriptions_forward, user_update_descriptions, item_update_memories)



    def backward_true(self, system_reasons, batch_user, batch_pos_item, batch_neg_item, round_1):
        batch_size = len(batch_user)
        pos_item_descriptions_forward, neg_item_descriptions_forward, pos_item_titles, neg_item_titles, user_descriptions_forward = [], [], [], [], []
        for i, user in enumerate(batch_user):
            pos_item_agent = self.item_agents[int(batch_pos_item[i])]
            neg_item_agent = self.item_agents[int(batch_neg_item[i])]
            pos_item_titles.append(pos_item_agent.role_description['item_title'])
            neg_item_titles.append(neg_item_agent.role_description['item_title'])
            pos_item_descriptions_forward.append(pos_item_agent.update_memory[-1])
            neg_item_descriptions_forward.append(neg_item_agent.update_memory[-1])
            user_descriptions_forward.append(self.user_agents[int(user)].update_memory[-1])

        if round_1:
            user_backward_prompts = [self.user_agents[int(batch_user[i])].astep_backward_true(system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i], neg_item_descriptions_forward[i]) for i in range(batch_size)]
            user_update_descriptions = []
            for i in range(0, batch_size, self.chat_api_batch):
                user_update_descriptions += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(user_backward_prompts[i:i+self.chat_api_batch]))

            user_update_descriptions = [self.user_agents[0].output_parser.parse_update(response["choices"][0]["message"]["content"]) for response in user_update_descriptions]
            for i, user in enumerate(batch_user):
                self.user_agents[int(user)].update_memory.append(user_update_descriptions[i])

            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward_true(
                    system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i], user_update_descriptions[i])
                for i in range(batch_size)]
        else:
            item_backward_prompts = [
                self.item_agents[int(batch_pos_item[i])].astep_backward_true(
                    system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_item_descriptions_forward[i],
                    neg_item_descriptions_forward[i], user_descriptions_forward[i])
                for i in range(batch_size)]

        item_update_memories = []
        for i in range(0, batch_size, self.chat_api_batch):
            item_update_memories += asyncio.run(
                self.item_agents[0].llm_chat.agenerate_response(item_backward_prompts[i:i + self.chat_api_batch]))

        item_update_memories = [self.item_agents[0].output_parser.parse(response["choices"][0]["message"]["content"])
                                for response in item_update_memories]

        # for descriptions in item_update_memories:
        #     print(descriptions)
        #     print('\n\n')
        #     input()

        for i in range(batch_size):
            if len(item_update_memories[i]) != 2:
                print("*" * 10 + "item update 出现 bug" + "*" * 10 + '\n')
            else:
                self.item_agents[int(batch_pos_item[i])].update_memory.append(item_update_memories[i][1])
                # TODO: whether to update negative item?
                if self.config['update_neg_item']: self.item_agents[int(batch_neg_item[i])].update_memory.append(
                    item_update_memories[i][0])


    def convert_system_selections_to_accuracy(self, system_selections, pos_items, neg_items):
        """
        判断系统的推荐结果是否正确
        """
        accuracy = []
        for i, selection in enumerate(system_selections):
            pos_item_title = self.item_text[int(pos_items[i])]
            neg_item_title = self.item_text[int(neg_items[i])]
            matched_name, _ = process.extractOne(selection, [pos_item_title, neg_item_title])
            if matched_name == pos_item_title:
                accuracy.append(1)
            else:
                accuracy.append(0)
        return accuracy


    def calculate_loss(self, interaction):
        print(f"User ID is : {interaction[self.USER_ID]}")
        print(f"Item ID is : {interaction[self.ITEM_ID]}")
        batch_user = interaction[self.USER_ID]
        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]
        batch_size = batch_user.size(0)

        # have_recorded_idx = set()

        for i in range(self.config['all_update_rounds']):
            print("~"*20 + f"{i}-th round update!" + "~"*20 + '\n')
            first_time = set()
            # TODO: forward part i.e. candidate item selection
            user_forward_description, pos_item_forward_description, neg_item_forward_description = [], [], []
            for j in range(batch_size):
                user_forward_description.append(self.user_agents[int(batch_user[j])].update_memory[-1])
                pos_item_forward_description.append(self.item_agents[int(batch_pos_item[j])].update_memory[-1])
                neg_item_forward_description.append(self.item_agents[int(batch_neg_item[j])].update_memory[-1])
            system_selections, system_reasons = self.forward(batch_user, batch_pos_item, batch_neg_item)
            accuracy = self.convert_system_selections_to_accuracy(system_selections, batch_pos_item, batch_neg_item)
            print(f"Current accuracy is {sum(accuracy) / len(accuracy)}")

            backward_system_reasons, backward_user, backward_pos_item, backward_neg_item, backward_system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true = [], [], [], [], [], [], [], []
            for j, acc in enumerate(accuracy):
                if acc == 0: # record wrong choices
                    backward_pos_item.append(int(batch_pos_item[j]))
                    backward_neg_item.append(int(batch_neg_item[j]))
                    backward_user.append(int(batch_user[j]))
                    backward_system_reasons.append(system_reasons[j])
                else: # record right choices
                    if i == 0:
                        first_time.add(int(batch_user[j]))
                        backward_user_true.append(int(batch_user[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])
                    elif int(batch_user[j]) not in first_time:
                        backward_user_true.append(int(batch_user[j]))
                        backward_pos_item_true.append(int(batch_pos_item[j]))
                        backward_neg_item_true.append(int(batch_neg_item[j]))
                        backward_system_reasons_true.append(system_reasons[j])

            # if sum(accuracy) / len(accuracy) > 0.9: break
            print(f"the user who are about to be updated: {backward_user}")
            self.backward(backward_system_reasons, backward_user, backward_pos_item, backward_neg_item)
            if i == 0 and len(backward_user_true):
                self.backward_true(backward_system_reasons_true, backward_user_true, backward_pos_item_true, backward_neg_item_true, True)
        self.backward_true(backward_system_reasons_true, backward_user_true, backward_pos_item_true,
                               backward_neg_item_true, False)


        if self.config['evaluation'] == 'rag':
            system_reasons_embeddings = self.generate_embedding(system_reasons)
            for i, user in enumerate(batch_user):
                self.rec_agent.user_examples[int(user)][(user_forward_description[i], self.item_text[int(batch_pos_item[i])], self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i], neg_item_forward_description[i], accuracy[i], system_reasons[i])] = system_reasons_embeddings[i]
        else:
            for i, user in enumerate(batch_user):
                self.rec_agent.user_examples[int(user)][(user_forward_description[i], self.item_text[int(batch_pos_item[i])], self.item_text[int(batch_neg_item[i])], pos_item_forward_description[i], neg_item_forward_description[i], accuracy[i], system_reasons[i])] = None

        self.logging_after_updation(batch_user, batch_pos_item, batch_neg_item)
        batch_pos_item_descriptions = []
        batch_neg_item_descriptions = []
        for i in range(batch_size):
            self.user_agents[int(batch_user[i])].memory_1.append(self.user_agents[int(batch_user[i])].update_memory[-1])
            batch_pos_item_descriptions.append(self.item_agents[int(batch_pos_item[i])].update_memory[-1])
            batch_neg_item_descriptions.append(self.item_agents[int(batch_neg_item[i])].update_memory[-1])

        if self.config['evaluation'] == 'rag':
            batch_pos_item_descriptions_embeddings = self.generate_embedding(batch_pos_item_descriptions)
            batch_neg_item_descriptions_embeddings = self.generate_embedding(batch_neg_item_descriptions)
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = batch_pos_item_descriptions_embeddings[i]
                self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = batch_neg_item_descriptions_embeddings[i]
        else:
            for i in range(batch_size):
                self.item_agents[int(batch_pos_item[i])].memory_embedding[batch_pos_item_descriptions[i]] = None
                self.item_agents[int(batch_neg_item[i])].memory_embedding[batch_neg_item_descriptions[i]] = None

    def logging_during_updation(self, batch_user, system_explanations, user_backward_prompts, pos_item_descriptions_forward, neg_item_descriptions_forward, user_update_descriptions, item_update_memories):
        batch_size = len(batch_user)
        for i in range(batch_size):
            user_id = int(batch_user[i])
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'Updation during reflection' + '~' * 20 + '\n')
                f.write(
                    f'There are two candidate CDs. \n The positive CD has the following information: {pos_item_descriptions_forward[i]}. \n The negative CD has the following information: {neg_item_descriptions_forward[i]}\n\n')
                f.write(
                    f'The recommender system made unsuitable recommendation. \n Its reasons are as follows: {system_explanations[i]}\n\n'
                )
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f"The prompts to update the user's descriptions is as follows: {user_backward_prompts[i]}\n\n"
                )
                f.write(
                    f'The user updates his self-description as follows: {user_update_descriptions[i]}\n\n')
                if self.config['update_neg_item']:
                    f.write(
                        f'The two candidate CDs update their description. \n The first CD has the following updated information: {item_update_memories[i][1]}\n The second CD has the following updated information {item_update_memories[i][0]} \n\n')
                else:
                    f.write(
                        f'The positive CD has the following updated information: {item_update_memories[i][1]}\n\n')

    def logging_after_updation(self, batch_user, batch_pos_item, batch_neg_item):
        print("~" * 20 + f"loging in record_{self.record_idx}" + "~" * 20)
        batch_size = batch_user.size(0)
        for i, user in enumerate(batch_user):
            user_id = int(user)
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{str(user_id)}'), 'a') as f:
                f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                f.write(
                    f'There are two candidate CDs. \n The first CD has the following information: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}. \n The second CD has the following information: {list(self.item_agents[int(batch_neg_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]} \n\n')
                if self.config['update_neg_item']:
                    f.write(
                        f'The two candidate CDs update their description. \n The first CD has the following updated information: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n The second CD has the following updated information {self.item_agents[int(batch_neg_item[i])].update_memory[-1]} \n\n')
                else:
                    f.write(
                        f'The positive CD has the following updated information: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n\n')

        for i in range(batch_size):
            pos_item_id = int(batch_pos_item[i])
            user_id = int(batch_user[i])
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'item_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'item.{str(pos_item_id)}'), 'a') as f:
                f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                f.write(
                    f"You: {self.item_agents[pos_item_id].role_description['item_title']} and the other movie: {self.item_agents[int(batch_neg_item[i])].role_description['item_title']} are recommended to a user.\n\n")
                f.write(
                    f'You have the following description: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f'The other movie has the following description: {list(self.item_agents[int(batch_neg_item[i])].memory_embedding.keys())[-1]}\n\n')
                f.write(
                    f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                f.write(
                    f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]}\n\n')
                f.write(
                    f'You update your description as follows: {self.item_agents[int(batch_pos_item[i])].update_memory[-1]}\n\n')
                if self.config['update_neg_item']:
                    f.write(
                        f'The other item updates the  following description: {self.item_agents[int(batch_neg_item[i])].update_memory[-1]}\n\n')
            if self.config['update_neg_item']:
                neg_item_id = int(batch_neg_item[i])
                path = osp.join(self.config['record_path'], self.dataset_name, 'record',
                                f'item_record_{self.record_idx}')
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(osp.join(path, f'item.{str(neg_item_id)}'), 'a') as f:
                    f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
                    f.write(
                        f"You: {self.item_agents[neg_item_id].role_description['item_title']} and the other movie: {self.item_agents[pos_item_id].role_description['item_title']} are recommended to a user.\n\n")
                    f.write(
                        f'The other movie has the following description: {list(self.item_agents[int(batch_pos_item[i])].memory_embedding.keys())[-1]}\n\n')
                    f.write(
                        f"The user's previous self-description is as follows: {self.user_agents[user_id].memory_1[-1]}\n\n")
                    f.write(
                        f'The user updates his self-description as follows: {self.user_agents[user_id].update_memory[-1]}\n\n')
                    f.write(
                        f'You update your description as follows: {self.item_agents[int(batch_neg_item[i])].update_memory[-1]}\n\n')

    
    def full_sort_predict(self, interaction, idxs):
        """
        Main function to rank with LLMs

        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return: score
        """

        batch_size = idxs.shape[0]
        batch_pos_item = interaction[self.ITEM_ID]
        # TODO: load previously saved user and item agents' memories
        if self.config['loaded']:
            self.record_idx = self.config['saved_idx']
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}',)
            with open(f'{path}/user','r') as f:
                f.readline()
                for line in f:
                    user, user_description = line.strip().split('\t')
                    user_id = self.user_token_id[user]
                    self.user_agents[user_id].memory_1.append(user_description)
            batch_user = interaction[self.USER_ID]
            for i, user in enumerate(batch_user):
                self.user_agents[int(user)].historical_interactions = np.load(f'{path}/user_embeddings_{self.user_id_token[int(user)]}.npy',allow_pickle=True).item()
                self.rec_agent.user_examples[int(user)] = np.load(f'{path}/user_examples_{self.user_id_token[int(user)]}.npy', allow_pickle=True).item()
            for i, item in enumerate(range(self.n_items)):
                if os.path.exists(f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy'):
                    self.item_agents[int(item)].memory_embedding = np.load(f'{path}/item_embeddings_{self.item_id_token[int(item)]}.npy',allow_pickle=True).item()

        if self.config['saved'] and not self.config['loaded']:
            path = osp.join(self.config['data_path'], 'saved', f'{self.record_idx}',)
            if not os.path.exists(path):
                os.makedirs(path)
            for item_id, item_context in self.item_agents.items():
                np.save(f'{path}/item_embeddings_{self.item_id_token[item_id]}.npy',
                        item_context.memory_embedding)
            with open(f'{path}/user','w') as f:
                f.write('user_id:token\tuser_description:token_seq\n')
                for user_id, user_context in self.user_agents.items():
                    user_description = user_context.memory_1[-1]
                    f.write(str(self.user_id_token[user_id]) + '\t' + user_description.replace('\n',' ') + '\n')
                    np.save(f'{path}/user_embeddings_{self.user_id_token[user_id]}.npy', user_context.historical_interactions)
                    np.save(f'{path}/user_examples_{self.user_id_token[user_id]}.npy',
                            self.rec_agent.user_examples[int(user_id)])

        all_candidate_idxs = set(idxs.view(-1).tolist())
        untrained_candidates = []
        for item in range(1, self.n_items):
            if list(self.item_agents[item].memory_embedding.keys())[-1].startswith('The CD is called'):
                if item in all_candidate_idxs:
                    untrained_candidates.append(item)
        print(f"In the reranking stage, there are {len(set(all_candidate_idxs))} candidates in total. \n There are {len(untrained_candidates)} have not been trained.")
        print("!!!")


        batch_user = interaction['user_id']
        batch_user_descriptions = []
        for i in range(batch_size):
            batch_user_descriptions.append(self.user_agents[int(batch_user[i])].memory_1[-1])
        if self.config['evaluation'] == 'rag' and self.config['item_representation'] == 'rag':
            batch_user_embedding_description = self.generate_embedding(batch_user_descriptions)
        else:
            batch_user_embedding_description = None

        scores = torch.full((batch_user.shape[0], self.n_items), -10000.)
        user_descriptions, list_of_item_descriptions, candidate_texts, user_his_texts, batch_user_embedding_explanations, batch_user_his, batch_select_examples = [], [], [], [], [], [], None
        for i in range(batch_size):
            user_id = int(batch_user[i])
            user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description = self.get_batch_inputs(interaction, idxs, i, batch_user_embedding_description)
            user_descriptions.append(self.user_agents[user_id].memory_1[-1])
            user_his_texts.append(user_his_text)
            list_of_item_descriptions.append('\n\n'.join(candidate_text_order_description))
            candidate_texts.append(candidate_text)
            batch_user_his.append(list(self.rec_agent.user_examples[user_id].keys()))


        if self.config['evaluation'] == 'rag':
            batch_select_examples = []
            query_embeddings = self.generate_embedding(list_of_item_descriptions)
            for i in tqdm(range(batch_size)):
                user_his_descriptions = self.user_agents[int(batch_user[i])].memory_1[1:-1]
                user_his_description_embeddings = self.generate_embedding(user_his_descriptions)
                distances = distances_from_embeddings(query_embeddings[i], user_his_description_embeddings)
                index = indices_of_nearest_neighbors_from_distances(distances)[0]
                batch_select_examples.append(user_his_descriptions[index])
            np.save(os.path.join(path,'batch_select_examples.npy'), np.array(batch_select_examples))

        if self.config['evaluation'] != 'sequential':
            user_his_texts = None
        evaluation_prompts, messages = self.evaluation(batch_user, user_descriptions, user_his_texts, list_of_item_descriptions, batch_select_examples)



        # # TODO: logging
        # for i in range(batch_size):
        #     user_id = int(batch_user[i])
        #     pos_item = int(batch_pos_item[i])
        #     path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
        #     with open(osp.join(path, f'user.{user_id}'), 'a') as f:
        #         f.write('~' * 20 + 'New interaction' + '~' * 20 + '\n')
        #         f.write(
        #             f'The evaluation prompts are as follows: {evaluation_prompts[i]}\n\n')
        #
        #         f.write(f'The system results are as follows: {messages[i]} \n\n')
        #         f.write(
        #             f"The pos item is: {self.item_agents[pos_item].role_description['item_title']}. Its related descriptions is as follows: {list(self.item_agents[pos_item].memory_embedding.keys())[-1]} \n\n")

        batch_pos_item = interaction[self.ITEM_ID]
    
        self.parsing_output_text(scores, messages, idxs, candidate_texts,batch_pos_item)
        return scores




    def evaluation(self, batch_user, user_descriptions, user_his_texts, list_of_item_descriptions, batch_select_examples=None):
        batch_size = len(user_descriptions)
        if batch_select_examples != None:
            # retrieval mode:
            evaluation_prompts = [self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], [], list_of_item_descriptions[i], batch_select_examples[i]) for i in range(batch_size)]
        else:
            if self.config['evaluation'] == 'sequential':
                evaluation_prompts = [self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], user_his_texts[i],
                                                list_of_item_descriptions[i]) for i in range(batch_size)]
            else:
                evaluation_prompts = [
                    self.rec_agent.astep_evaluation(int(batch_user[i]), user_descriptions[i], [],
                                                    list_of_item_descriptions[i]) for i in range(batch_size)]

        messages = []
        for i in tqdm(range(0, batch_size, self.chat_api_batch)):
            messages += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(evaluation_prompts[i:i+self.chat_api_batch]))

        messages = [self.rec_agent.output_parser.parse_evaluation(response["choices"][0]["message"]["content"]) for response in messages]
        return evaluation_prompts, messages

    def get_batch_inputs(self, interaction, idxs, i, user_embedding):
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        real_his_len = min(self.max_his_len, user_his_len[i].item())
        user_his_text = [str(j+1) + '. ' + self.item_text[user_his[i, user_his_len[i].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]

        candidate_text = [self.item_text[idxs[i, j]]
                          for j in range(idxs.shape[1])]
        candidate_text_order = [str(j + 1) + '. ' + self.item_text[idxs[i, j].item()]
                                for j in range(idxs.shape[1])]

        if self.config['item_representation'] == 'direct':
            candidate_text_order_description = [str(j + 1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' +
                                                list(self.item_agents[idxs[i, j].item()].memory_embedding.keys())[-1]
                                                for j in range(idxs.shape[1])]
        elif self.config['item_representation'] == 'rag' and self.config['evaluation'] == 'rag':
            item_descriptions = []
            for item in idxs[i]:
                item = int(item)
                item_embeddings = list(self.item_agents[item].memory_embedding.values())
                distances = distances_from_embeddings(user_embedding, item_embeddings)
                indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)[0]
                item_descriptions.append(list(self.item_agents[item].memory_embedding.keys())[indices_of_nearest_neighbors])
            candidate_text_order_description = [str(j+1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' +
                                                item_descriptions[j] for j in range(idxs.shape[1])]





        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description
    

    def parsing_output_text(self, scores, messages, idxs, candidate_texts,batch_pos_item):
        all_recommendation_ranking_results = []
        for i, message in enumerate(messages):
            ranking_result = []
            candidate_text = candidate_texts[i]
            matched_names = []
            for j, item_detail in enumerate(message):
                if len(item_detail) < 1:
                    continue
                if item_detail.endswith('candidate movies:'):
                    continue
                pr = item_detail.find('. ')
                if item_detail[:pr].isdigit():
                    item_name = item_detail[pr + 2:].strip()
                else:
                    item_name = item_detail.strip()

                if self.config['match_rule'] == 'exact':
                    for id, candidate_text_single in enumerate(candidate_text):
                        if candidate_text_single in item_name:
                            item_id = idxs[i,id]
                            if scores[i, item_id] > -5000.: break # has been recommended
                            scores[i, item_id] = self.config['recall_budget'] - j
                            break
                elif self.config['match_rule'] == 'fuzzy':
                    matched_name, sim_score = process.extractOne(item_name, candidate_text)
                    matched_names.append(matched_name)
                    matched_idx = candidate_text.index(matched_name)
                    item_id = idxs[i,matched_idx]
                    if scores[i, item_id] > -5000.: continue # has been recommended
                    ranking_result.append(self.item_id_token[item_id])
                    scores[i, item_id] = self.config['recall_budget'] - j
            all_recommendation_ranking_results.append(ranking_result)


                        


    
