import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset, Dataset
from collections import defaultdict

class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
        print(loaded_feat.shape)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        item2row_path = osp.join(self.config['data_path'], f'{self.dataset_name}_item_dataset2row.npy')
        item2row = np.load(item2row_path,allow_pickle=True).item()
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[item2row[int(token)]]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class VQRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = self.load_index()

    def load_index(self):
        import faiss
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        old_pq_codes, _, _, _ = self.parse_faiss_index(uni_index)
        old_code_num = old_pq_codes.shape[0]

        self.plm_suffix = self.config['plm_suffix']
        self.plm_size = self.config['plm_size']
        feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        uni_index.add(loaded_feat)
        all_pq_codes, centroid_embeds, coarse_embeds, opq_transform = self.parse_faiss_index(uni_index)
        pq_codes = all_pq_codes[old_code_num:]
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        # assert self.item_num == 1 + pq_codes.shape[0], pq_codes.shape

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(self.code_dim):
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        item2row_path = osp.join(self.config['data_path'], f'{self.dataset_name}_item_dataset2row.npy')
        item2row = np.load(item2row_path, allow_pickle=True).item()
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[item2row[int(token)]]
            
        self.plm_embedding = torch.FloatTensor(loaded_feat)
        return torch.LongTensor(mapped_codes)

    @staticmethod
    def parse_faiss_index(pq_index):
        import faiss
        vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
        assert isinstance(vt, faiss.LinearTransform)
        opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

        ivf_index = faiss.downcast_index(pq_index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)

        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), ivf_index.pq.M * ivf_index.pq.dsub)
        coarse_embeds = coarse_embeds.reshape(-1)

        return pq_codes, centroid_embeds, coarse_embeds, opq_transform
    
class BPRDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.user_group_order = config['user_group_order']


    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]
            # return datasets
            train_dataset, valid_dataset, test_dataset = datasets
            if self.user_group_order == 'user_first':
                print("user_first!!!!!!!!")
                return train_dataset, valid_dataset, test_dataset

            elif self.user_group_order == 'item_first':
                user_dataset = train_dataset['user_id'].numpy()
                item_dataset = train_dataset['item_id'].numpy()
                # print(user_dataset)
                # print(item_dataset)
                # print(len(user_dataset))
                # print(len(set(user_dataset)))
                # print(len(item_dataset))
                # print(len(set(item_dataset)))
                # item2count = defaultdict(int)
                # for item in item_dataset:
                #     item2count[item] += 1
                # num_1 = 0
                # for item,count in item2count.items():
                #     if count == 1: num_1 += 1
                # num_2 = 0
                # for item, count in item2count.items():
                #     if count == 2: num_2 += 1
                # num_3 = 0
                # for item, count in item2count.items():
                #     if count == 3: num_3 += 1
                # print(num_1)
                # print(num_2)
                # print(num_3)
                # print("!!!!")

                # print(train_dataset['item_id'].numpy())
                grouped_inter_feat_index = self._grouped_index(
                    train_dataset['item_id'].numpy()
                )
                print(grouped_inter_feat_index)
                selected_idxs = []
                for idxs in grouped_inter_feat_index:
                    selected_idxs.extend(idxs)


                train_dataset = self.copy(self.inter_feat[selected_idxs])
                # print(selected_idx)

                return train_dataset, valid_dataset, test_dataset

            elif self.user_group_order == 'item_sequential':
                grouped_inter_feat_index = self._grouped_index(
                    train_dataset['item_id'].numpy()
                )
                selected_idx = []
                while 1:
                    selected_item = []
                    for item, idxs in enumerate(grouped_inter_feat_index):
                        if len(idxs):
                            selected_idx.append(idxs[0])
                            idxs.pop(0)
                        else:
                            selected_item.append(item)
                    if len(selected_item) == len(grouped_inter_feat_index): break

                train_dataset = self.copy(self.inter_feat[selected_idx])
                return train_dataset, valid_dataset, test_dataset

            grouped_inter_feat_index = self._grouped_index(
                train_dataset['user_id'].numpy()
            )

            selected_idx = []

            while 1:
                selected_user = []
                for user, idxs in enumerate(grouped_inter_feat_index):
                    if len(idxs):
                        selected_idx.append(idxs[0])
                        idxs.pop(0)
                    else:
                        selected_user.append(user)
                if len(selected_user) == len(grouped_inter_feat_index): break

            train_dataset = self.copy(self.inter_feat[selected_idx])
            # print(selected_idx)

            return train_dataset, valid_dataset, test_dataset
    # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(
                f"The ordering_method [{ordering_args}] has not been implemented."
            )

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config["eval_args"]["group_by"]
        if split_mode == "RS":
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == "none":
                datasets = self.split_by_ratio(split_args["RS"], group_by=None)
            elif group_by == "user":
                datasets = self.split_by_ratio(
                    split_args["RS"], group_by=self.uid_field
                )
            else:
                raise NotImplementedError(
                    f"The grouping method [{group_by}] has not been implemented."
                )
        elif split_mode == "LS":
            datasets = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        else:
            raise NotImplementedError(
                f"The splitting_method [{split_mode}] has not been implemented."
            )

        return datasets


class ITEMBPRDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self.user_group_order = config['user_group_order']

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]
            # return datasets
            train_dataset, valid_dataset, test_dataset = datasets
            if self.user_group_order == 'user_first':
                return train_dataset, valid_dataset, test_dataset

            elif self.user_group_order == 'item_first':
                user_dataset = train_dataset['user_id'].numpy()
                item_dataset = train_dataset['item_id'].numpy()
                print(len(user_dataset))
                print(len(set(user_dataset)))
                item2count = defaultdict(int)
                for item in item_dataset:
                    item2count[item] += 1
                grouped_inter_feat_index = self._grouped_index(
                    train_dataset['item_id'].numpy()
                )
                # print(grouped_inter_feat_index)
                selected_idxs = []
                for idxs in grouped_inter_feat_index:
                    selected_idxs.extend(idxs)

                train_dataset = self.copy(self.inter_feat[selected_idxs])
                return train_dataset, valid_dataset, test_dataset
            elif self.user_group_order == 'item_sequential':
                grouped_inter_feat_index = self._grouped_index(
                    train_dataset['item_id'].numpy()
                )
                selected_idx = []
                while 1:
                    selected_item = []
                    for item, idxs in enumerate(grouped_inter_feat_index):
                        if len(idxs):
                            selected_idx.append(idxs[0])
                            idxs.pop(0)
                        else:
                            selected_item.append(item)
                    if len(selected_item) == len(grouped_inter_feat_index): break

                train_dataset = self.copy(self.inter_feat[selected_idx])
                # print(selected_idx)

                return train_dataset, valid_dataset, test_dataset

            grouped_inter_feat_index = self._grouped_index(
                train_dataset['user_id'].numpy()
            )

            selected_idx = []

            while 1:
                selected_user = []
                for user, idxs in enumerate(grouped_inter_feat_index):
                    if len(idxs):
                        selected_idx.append(idxs[0])
                        idxs.pop(0)
                    else:
                        selected_user.append(user)
                if len(selected_user) == len(grouped_inter_feat_index): break

            train_dataset = self.copy(self.inter_feat[selected_idx])
            # print(selected_idx)

            return train_dataset, valid_dataset, test_dataset
        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(
                f"The ordering_method [{ordering_args}] has not been implemented."
            )

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config["eval_args"]["group_by"]
        if split_mode == "RS":
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == "none":
                datasets = self.split_by_ratio(split_args["RS"], group_by=None)
            elif group_by == "user":
                datasets = self.split_by_ratio(
                    split_args["RS"], group_by=self.uid_field
                )
            else:
                raise NotImplementedError(
                    f"The grouping method [{group_by}] has not been implemented."
                )
        elif split_mode == "LS":
            datasets = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        else:
            raise NotImplementedError(
                f"The splitting_method [{split_mode}] has not been implemented."
            )

        return datasets