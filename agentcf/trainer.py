import os
import numpy as np
from tqdm import tqdm
import torch
from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color, early_stopping, dict2str,get_gpu_usage
from recbole.data.interaction import Interaction
from time import time

class LanguageLossTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.sampled_user_suffix = config['sampled_user_suffix']  # candidate generation model, by default, random
        self.recall_budget = config['recall_budget']                # size of candidate Sets, by default, 20
        self.fix_pos = config['fix_pos']                            # whether fix the position of ground-truth items in the candidate set, by default, -1
        self.user2sampled_item = self.load_selected_items(config, dataset)

    def load_selected_items(self, config, dataset):
        sampled_item_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.sampled_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        user2sampled_item = {}
        with open(sampled_item_file, 'r', encoding='utf-8') as file:
            for line in file:
                uid, iid_list = line.strip().split('\t')
                user2sampled_item[user_token2id[uid]] = [item_token2id[_] for _ in iid_list.split(' ')]
        return user2sampled_item        
        
    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        eval_func = self._full_sort_batch_eval
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            sampled_items = []
            for i in range(len(interaction)):
                user_id = int(interaction['user_id'][i].item())
                sampled_item = self.user2sampled_item[user_id]
                sampled_items.append(sampled_item)
            sampled_items = torch.LongTensor(sampled_items)

            if self.config['has_gt']:
                self.logger.info('Has ground truth')
                idxs = torch.LongTensor(sampled_items)
                for i in range(idxs.shape[0]):
                    if positive_i[i] in idxs[i]:
                        pr = idxs[i].cpu().numpy().tolist().index(positive_i[i].item())
                        idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])

                idxs = idxs[:,:self.recall_budget - 1]
                if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                    idxs = torch.cat([idxs, positive_i.unsqueeze(-1)], dim=-1).numpy()
                elif self.fix_pos == 0:
                    idxs = torch.cat([positive_i.unsqueeze(-1), idxs], dim=-1).numpy()
                else:
                    idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                    idxs = torch.cat([idxs_a, positive_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
            else:
                self.logger.info('Does not have ground truth.')
                idxs = torch.LongTensor(self.sampled_items)
                idxs = idxs[:,:self.recall_budget]
                idxs = idxs.numpy()

            if self.fix_pos == -1:
                self.logger.info('Shuffle ground truth')
                for i in range(idxs.shape[0]):
                    np.random.shuffle(idxs[i])
            idxs = torch.LongTensor(idxs)
            interaction, scores, positive_u, positive_i = eval_func(batched_data, idxs)
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result
        


    def _full_sort_batch_eval(self, batched_data, sampled_items):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device),sampled_items.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            # eval
            self._save_checkpoint(epoch_idx, verbose=verbose)
               

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            loss_func(interaction)
            # break
        return total_loss
    


class ITEMLanguageLossTrainer(LanguageLossTrainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model, dataset)
        self.item2sampled_item = self.load_selected_items(config, dataset)

    def load_selected_items(self, config, dataset):
        sampled_item_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.sampled_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        item2sampled_item = {}
        with open(sampled_item_file, 'r', encoding='utf-8') as file:
            for line in file:
                iid, iid_list = line.strip().split('\t')
                item2sampled_item[item_token2id[iid]] = [item_token2id[_] for _ in iid_list.split(' ')]
        return item2sampled_item

    @torch.no_grad()
    def evaluate(
            self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        eval_func = self._full_sort_batch_eval
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            sampled_items = []
            for i in range(len(interaction)):
                user_id = int(interaction['user_id'][i].item())
                item_id = int(interaction['item_id'][i].item())
                sampled_item = self.item2sampled_item[item_id]
                sampled_items.append(sampled_item)
            sampled_items = torch.LongTensor(sampled_items)

            if self.config['has_gt']:
                self.logger.info('Has ground truth')
                idxs = torch.LongTensor(sampled_items)
                for i in range(idxs.shape[0]):
                    if positive_i[i] in idxs[i]:
                        pr = idxs[i].cpu().numpy().tolist().index(positive_i[i].item())
                        idxs[i][pr:-1] = torch.clone(idxs[i][pr + 1:])

                idxs = idxs[:, :self.recall_budget - 1]
                if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                    idxs = torch.cat([idxs, positive_i.unsqueeze(-1)], dim=-1).numpy()
                elif self.fix_pos == 0:
                    idxs = torch.cat([positive_i.unsqueeze(-1), idxs], dim=-1).numpy()
                else:
                    idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                    idxs = torch.cat([idxs_a, positive_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
            else:
                self.logger.info('Does not have ground truth.')
                idxs = torch.LongTensor(self.sampled_items)
                idxs = idxs[:, :self.recall_budget]
                idxs = idxs.numpy()

            if self.fix_pos == -1:
                self.logger.info('Shuffle ground truth')
                for i in range(idxs.shape[0]):
                    np.random.shuffle(idxs[i])
            idxs = torch.LongTensor(idxs)
            interaction, scores, positive_u, positive_i = eval_func(batched_data, idxs)
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result




class SelectedUserTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.selected_user_suffix = config['selected_user_suffix']  # candidate generation model, by default, random
        self.recall_budget = config['recall_budget']                # size of candidate Sets, by default, 20
        self.fix_pos = config['fix_pos']                            # whether fix the position of ground-truth items in the candidate set, by default, -1
        self.selected_uids, self.sampled_items = self.load_selected_users(config, dataset)

    def load_selected_users(self, config, dataset):
        selected_users = []
        sampled_items = []
        selected_user_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.selected_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        with open(selected_user_file, 'r', encoding='utf-8') as file:
            for line in file:
                uid, iid_list = line.strip().split('\t')
                selected_users.append(uid)
                sampled_items.append([item_token2id[_] if (_ in item_token2id) else 0 for _ in iid_list.split(' ')])
        selected_uids = list([user_token2id[_] for _ in selected_users])
        return selected_uids, sampled_items

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        new_inter = {
            col: torch.stack([inter[col] for inter in selected_interactions]) for col in selected_interactions[0].columns
        }
        selected_interactions = Interaction(new_inter)
        selected_pos_i = torch.stack(selected_pos_i)
        selected_pos_u = torch.arange(selected_pos_i.shape[0])

        if self.config['has_gt']:
            self.logger.info('Has ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            for i in range(idxs.shape[0]):
                if selected_pos_i[i] in idxs[i]:
                    pr = idxs[i].numpy().tolist().index(selected_pos_i[i].item())
                    idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])

            idxs = idxs[:,:self.recall_budget - 1]
            if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                idxs = torch.cat([idxs, selected_pos_i.unsqueeze(-1)], dim=-1).numpy()
            elif self.fix_pos == 0:
                idxs = torch.cat([selected_pos_i.unsqueeze(-1), idxs], dim=-1).numpy()
            else:
                idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                idxs = torch.cat([idxs_a, selected_pos_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
        else:
            self.logger.info('Does not have ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            idxs = idxs[:,:self.recall_budget]
            idxs = idxs.numpy()

        if self.fix_pos == -1:
            self.logger.info('Shuffle ground truth')
            for i in range(idxs.shape[0]):
                np.random.shuffle(idxs[i])
        scores = self.model.predict_on_subsets(selected_interactions.to(self.device), idxs)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        self.eval_collector.eval_batch_collect(
            scores, selected_interactions, selected_pos_u, selected_pos_i
        )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result