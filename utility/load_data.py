import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from utility.parser import parse_args
import torch
import torch.nn.functional as F
import pickle

args = parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.device = device

        train_file = path + '/train.json'
        val_file = path + '/val.json'
        test_file = path + '/test.json'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.n_val = 0
        self.neg_pools = {}
        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))


        for uid, items in train.items():
            if len(items) == 0: continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except: continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except: continue

        self.n_items += 1
        self.n_users += 1


        text_feats = np.load(args.data_path + args.dataset + '/text_feat.npy')
        self.text_feats = torch.tensor(text_feats, dtype=torch.float32).to(self.device)
        self.n_items = self.text_feats.shape[0]

        self.train_items, self.test_set, self.val_set = {}, {}, {}

        train_rows, train_cols = [], []
        for uid, items in train.items():
            if len(items) == 0: continue
            uid = int(uid)
            self.train_items[uid] = items
            train_rows.extend([uid] * len(items))
            train_cols.extend(items)
        for uid, items in train.items():

            items = [int(x) for x in items]
            self.train_items[int(uid)] = items

        data = np.ones_like(train_rows, dtype=np.float32)
        self.R = sp.csr_matrix((data, (train_rows, train_cols)), shape=(self.n_users, self.n_items))

        for uid, items in test.items():
            if len(items) > 0: self.test_set[int(uid)] = items
        for uid, items in val.items():
            if len(items) > 0: self.val_set[int(uid)] = items


        iu_dense = torch.tensor(self.R.toarray().T, dtype=torch.float32).to(self.device) # [I, U]
        self.co_occur = torch.matmul(iu_dense, iu_dense.T)
        self.co_occur.fill_diagonal_(0)
        print("Item co-occurrence matrix computed. Shape:", self.co_occur.shape)

        self.struct_support_topk = 5
        self.struct_co_theta = 5
        import os

        embedding_path = args.data_path + args.dataset + '/augmented_attribute_embedding_dict'
        try:

            if not os.path.exists(embedding_path):
                embedding_path = args.data_path + args.dataset + '/augmented_atttribute_embedding_dict'

            raw_emb_dict = pickle.load(open(embedding_path, 'rb'))
            print(f"Loaded semantic embeddings from: {embedding_path}")


            emb_dim = 0
            for attr_data in raw_emb_dict.values():
                if isinstance(attr_data, dict) and len(attr_data) > 0:
                    first_val = next(iter(attr_data.values()))
                    emb_dim = len(first_val)
                    break

            if emb_dim == 0:
                raise ValueError("Could not detect embedding dimension from dict.")

            print(f"Detected embedding dimension: {emb_dim}")


            self.semantic_emb_matrix = torch.zeros((self.n_items, emb_dim), dtype=torch.float32, device=self.device)
            self.valid_semantic_mask = torch.zeros(self.n_items, dtype=torch.bool, device=self.device)
            self.valid_semantic_item_ids = []

            all_potential_ids = set()
            attr_dicts = []
            for attr, content in raw_emb_dict.items():
                if isinstance(content, dict):
                    attr_dicts.append(content)
                    all_potential_ids.update(content.keys())
                else:
                    print(f"[Warning] Attribute '{attr}' is not a dict, skipping.")

            print(f"Found {len(all_potential_ids)} unique keys across all attributes.")


            count_loaded = 0
            for raw_key in all_potential_ids:
                try:
                    item_id = int(raw_key)
                except:
                    continue



                vectors = []
                for d in attr_dicts:
                    if raw_key in d and d[raw_key] is not None:
                        vectors.append(d[raw_key])

                if len(vectors) > 0:
                    vec_mean = np.mean(np.array(vectors), axis=0)

                    if item_id >= self.semantic_emb_matrix.shape[0]:
                        extra_rows = item_id - self.semantic_emb_matrix.shape[0] + 1
                        extra_matrix = torch.zeros((extra_rows, emb_dim), dtype=torch.float32, device=self.device)
                        self.semantic_emb_matrix = torch.cat([self.semantic_emb_matrix, extra_matrix], dim=0)
                        extra_mask = torch.zeros(extra_rows, dtype=torch.bool, device=self.device)
                        self.valid_semantic_mask = torch.cat([self.valid_semantic_mask, extra_mask], dim=0)

                    self.semantic_emb_matrix[item_id] = torch.tensor(vec_mean, dtype=torch.float32, device=self.device)
                    self.valid_semantic_mask[item_id] = True
                    self.valid_semantic_item_ids.append(item_id)
                    count_loaded += 1

            print(f"Successfully loaded {count_loaded} item embeddings (Merged & Mapped).")


            self.semantic_emb_norm = F.normalize(self.semantic_emb_matrix, p=2, dim=1)
            self.valid_semantic_item_ids_set = set(self.valid_semantic_item_ids)



        except Exception as e:
            print(f"Warning: Semantic embedding loading failed ({e}).")
            import traceback
            traceback.print_exc()
            self.semantic_emb_norm = torch.zeros((self.n_items, 64), device=self.device)
            self.valid_semantic_item_ids_set = set()
            self.valid_semantic_mask = torch.zeros(self.n_items, dtype=torch.bool, device=self.device)


        self.semantic_sample_num = 2
        self.semantic_topk =10

        self.semantic_neg_thresh = 0.3

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

            adj_mat = torch.tensor(adj_mat.toarray(), dtype=torch.float32).to(self.device)
            norm_adj_mat = torch.tensor(norm_adj_mat.toarray(), dtype=torch.float32).to(self.device)
            mean_adj_mat = torch.tensor(mean_adj_mat.toarray(), dtype=torch.float32).to(self.device)
        except:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', sp.csr_matrix(adj_mat.cpu().numpy()))
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', sp.csr_matrix(norm_adj_mat.cpu().numpy()))
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', sp.csr_matrix(mean_adj_mat.cpu().numpy()))
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        R_coo = self.R.tocoo()

        rows = np.concatenate([R_coo.row, R_coo.col + self.n_users])
        cols = np.concatenate([R_coo.col + self.n_users, R_coo.row])
        data = np.ones_like(rows, dtype=np.float32)

        adj_mat = sp.csr_matrix((data, (rows, cols)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = adj.sum(dim=1)
            d_inv = torch.pow(rowsum, -1).flatten()
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diag(d_inv)
            norm_adj = torch.matmul(d_mat_inv, adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj

        adj_mat_tensor = torch.tensor(adj_mat.toarray(), dtype=torch.float32).to(self.device)
        adj_mat_tensor += torch.eye(adj_mat_tensor.shape[0]).to(self.device)

        norm_adj_mat = normalized_adj_single(adj_mat_tensor)
        mean_adj_mat = normalized_adj_single(torch.tensor(adj_mat.toarray(), dtype=torch.float32).to(self.device))

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat_tensor, norm_adj_mat, mean_adj_mat

    def sample(self):

        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            if len(pos_items) >= num:
                return rd.sample(pos_items, num)
            else:

                return [rd.choice(pos_items) for _ in range(num)]


        def sample_neg_items_for_u(u, num):
            neg_items = []
            pos_set = set(self.train_items[u])
            while len(neg_items) < num:
                neg_id = rd.randint(0, self.n_items - 1)
                if neg_id not in pos_set:
                    neg_items.append(neg_id)
            return neg_items


        def sample_struct_pos_items_for_u(u, num):
            pos_items = self.train_items[u]


            if len(pos_items) <= num:

                needed = num - len(pos_items)
                fallback = sample_pos_items_for_u(u, needed)
                return pos_items + fallback


            pos_tensor = torch.tensor(pos_items, dtype=torch.long, device=self.device)
            sub_co = self.co_occur[pos_tensor][:, pos_tensor]
            support_scores = sub_co.sum(dim=1)

            _, indices = torch.topk(support_scores, k=num)
            candidates = pos_tensor[indices].cpu().tolist()

            return candidates


        def sample_struct_neg_items_for_u(u, num):
            pos_items = self.train_items[u]
            pos_tensor = torch.tensor(pos_items, dtype=torch.long, device=self.device)
            sub_co = self.co_occur[pos_tensor][:, pos_tensor]
            scores = sub_co.sum(dim=1)
            k_core = min(self.struct_support_topk, len(pos_items))
            _, indices = torch.topk(scores, k=k_core)
            core_pos_items = pos_tensor[indices]

            core_rows = self.co_occur[core_pos_items]

            mask_related = (core_rows >= self.struct_co_theta).any(dim=0)

            neg_candidates = []
            pos_set = set(pos_items)
            attempts = 0

            while len(neg_candidates) < num and attempts < 100:
                attempts += 1
                cand = rd.randint(0, self.n_items - 1)
                if cand not in pos_set and not mask_related[cand].item():
                    neg_candidates.append(cand)


            if len(neg_candidates) < num:
                needed = num - len(neg_candidates)
                fallback = sample_neg_items_for_u(u, needed)
                neg_candidates.extend(fallback)

            return neg_candidates[:num]


        def sample_semantic_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            valid_pos = [i for i in pos_items if i in self.valid_semantic_item_ids_set]

            valid_pos_tensor = torch.tensor(valid_pos, dtype=torch.long, device=self.device)
            user_center = self.semantic_emb_norm[valid_pos_tensor].mean(dim=0)
            user_center = F.normalize(user_center, p=2, dim=0)

            num_hist = num
            sampled_items = []
            pos_tensor_all = torch.tensor(pos_items, dtype=torch.long, device=self.device)
            pos_embs = self.semantic_emb_norm[pos_tensor_all]
            sims_hist = torch.mv(pos_embs, user_center)

            k_hist = min(self.semantic_topk, pos_embs.size(0))
            _, top_idx_hist = torch.topk(sims_hist, k=k_hist)
            hist_candidates = pos_tensor_all[top_idx_hist].cpu().tolist()

            if len(hist_candidates) >= num_hist:
                sampled_items.extend(rd.sample(hist_candidates, num_hist))
            else:
                sampled_items.extend(hist_candidates)

            if len(sampled_items) < num:
                sampled_items.extend(
                    sample_pos_items_for_u(u, num - len(sampled_items))
                )

            return sampled_items[:num]

        def sample_semantic_neg_items_for_u(u, num):
            pos_items = self.train_items[u]
            valid_pos = [i for i in pos_items if i in self.valid_semantic_item_ids_set]

            if len(valid_pos) == 0:
                return sample_neg_items_for_u(u, num)

            valid_pos_tensor = torch.tensor(valid_pos, dtype=torch.long, device=self.device)
            user_center = self.semantic_emb_norm[valid_pos_tensor].mean(dim=0)
            user_center = F.normalize(user_center, p=2, dim=0)

            all_sims = torch.mv(self.semantic_emb_norm, user_center)

            pos_tensor_all = torch.tensor(pos_items, dtype=torch.long, device=self.device)
            all_sims[pos_tensor_all] = float('inf')
            all_sims[~self.valid_semantic_mask] = float('inf')

            neg_candidates = []
            attempts = 0
            while len(neg_candidates) < num and attempts < 50:
                attempts += 1
                cand = rd.randint(0, self.n_items - 1)

                if (cand not in pos_items and
                        cand in self.valid_semantic_item_ids_set and
                        all_sims[cand].item() < self.semantic_neg_thresh):
                    neg_candidates.append(cand)


            if len(neg_candidates) < num:
                needed = num - len(neg_candidates)

                fallback = sample_neg_items_for_u(u, needed)
                neg_candidates.extend(fallback)

            return neg_candidates[:num]

        random_pos, random_neg = [], []
        struct_pos, struct_neg = [], []
        semantic_pos, semantic_neg = [], []

        for u in users:

            random_pos += sample_pos_items_for_u(u, self.semantic_sample_num)
            random_neg += sample_neg_items_for_u(u, self.semantic_sample_num)

            struct_pos += sample_struct_pos_items_for_u(u, self.semantic_sample_num)
            struct_neg += sample_struct_neg_items_for_u(u, self.semantic_sample_num)

            semantic_pos += sample_semantic_pos_items_for_u(u, self.semantic_sample_num)
            semantic_neg += sample_semantic_neg_items_for_u(u, self.semantic_sample_num)

        def to_gpu(lst):
            return torch.tensor(lst, dtype=torch.long, device=self.device)

        return (to_gpu(users),
                to_gpu(random_pos), to_gpu(random_neg),
                to_gpu(struct_pos), to_gpu(struct_neg),
                to_gpu(semantic_pos), to_gpu(semantic_neg))