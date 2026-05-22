from datetime import datetime
import math
import os
import random
import sys
from tqdm import tqdm
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utility.parser import parse_args
from model import MM_Model
from utility.batch_test import *
from utility.logging1 import Logger
from utility.norm import build_sim, build_knn_normalized_graph

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        self.device = device
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        self.ui_graph = self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat', 'rb'))

        # get user embedding
        augmented_user_init_embedding = pickle.load(
            open(args.data_path + args.dataset + '/augmented_user_init_embedding', 'rb'))
        augmented_user_init_embedding_list = []
        for i in range(len(augmented_user_init_embedding)):
            augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
        augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
        self.user_init_embedding = augmented_user_init_embedding_final

        # get separate embedding matrix
        if args.dataset == 'preprocessed_raw_MovieLens':
            augmented_total_embed_dict = {'title': [], 'genre': [], 'director': [], 'country': [], 'language': []}
        elif args.dataset == 'netflix':
            augmented_total_embed_dict = {'year': [], 'title': [], 'director': [], 'country': [], 'language': []}
        else:
            augmented_total_embed_dict = {'Year': [], 'Title': [], 'director': [], 'country': [], 'language': []}

        try:
            augmented_atttribute_embedding_dict = pickle.load(
                open(args.data_path + args.dataset + '/augmented_attribute_embedding_dict', 'rb'))
        except FileNotFoundError:

            augmented_atttribute_embedding_dict = pickle.load(
                open(args.data_path + args.dataset + '/augmented_atttribute_embedding_dict', 'rb'))

        for value in augmented_atttribute_embedding_dict.keys():
            for i in range(len(augmented_atttribute_embedding_dict[value])):
                augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])
            augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])

        self.item_attribute_embedding = augmented_total_embed_dict

        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T

        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self.model_mm = MM_Model(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                                 self.image_feats, self.text_feats, self.user_init_embedding,
                                 self.item_attribute_embedding)
        self.model_mm = self.model_mm.to("cuda")

        self.optimizer = optim.AdamW([{'params': self.model_mm.parameters()}], lr=self.lr)

    def cal_bpr_loss(self, users, pos_items, neg_items, user_emb, item_emb):
        current_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        pos_scores = torch.sum(current_user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(current_user_emb * neg_item_emb, dim=1)

        difference = pos_scores - neg_scores
        loss = -torch.mean(F.logsigmoid(difference + 1e-8))

        reg_loss = (1 / 2) * (current_user_emb.norm(2).pow(2) +
                              pos_item_emb.norm(2).pow(2) +
                              neg_item_emb.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss * self.decay

    def info_nce_loss_with_batch_neg(self, anchor_emb, positive_emb, negative_emb, pos_weight, neg_weight, tau=0.1):

        batch_size = anchor_emb.shape[0]


        anchor_norm = F.normalize(anchor_emb, p=2, dim=-1)  # [B, D]
        pos_norm = F.normalize(positive_emb, p=2, dim=-1)  # [B, 2, D]
        neg_norm = F.normalize(negative_emb, p=2, dim=-1)  # [B, 2, D]


        sim_pos = torch.matmul(anchor_norm.unsqueeze(1), pos_norm.transpose(1, 2)).squeeze(1)
        exp_pos = torch.exp(sim_pos / tau)  # [B, 2]


        sim_neg = torch.matmul(anchor_norm.unsqueeze(1), neg_norm.transpose(1, 2)).squeeze(1)
        exp_explicit_neg = torch.exp(sim_neg / tau) * neg_weight  # [B, 2]
        sum_exp_explicit_neg = exp_explicit_neg.sum(dim=1)  # [B]


        all_pos_candidates = pos_norm.view(-1, self.emb_dim)


        sim_batch = torch.matmul(anchor_norm, all_pos_candidates.t())
        exp_batch = torch.exp(sim_batch / tau)


        mask = torch.zeros((batch_size, 2 * batch_size), dtype=torch.bool, device=self.device)
        row_ids = torch.arange(batch_size, device=self.device)
        mask[row_ids, 2 * row_ids] = True
        mask[row_ids, 2 * row_ids + 1] = True

        exp_batch = exp_batch.masked_fill(mask, 0.0)
        sum_exp_batch_neg = exp_batch.sum(dim=1)


        denominator = exp_pos + sum_exp_explicit_neg.unsqueeze(1) + sum_exp_batch_neg.unsqueeze(1) + 1e-8

        log_prob = (sim_pos / tau) - torch.log(denominator)


        loss = -(log_prob * pos_weight).sum() / (pos_weight.sum() + 1e-8)


        loss_components = {}
        with torch.no_grad():
            p = torch.exp(sim_pos[:, 0] / tau)
            loss_components['struct_pos'] = (
                -torch.log(p / (p + sum_exp_explicit_neg + sum_exp_batch_neg + 1e-8))).mean().item()
            n = torch.exp(sim_neg[:, 0] / tau)
            loss_components['struct_neg'] = (-torch.log(1 / (1 + n + 1e-8))).mean().item()

        return loss, loss_components

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)
        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)
        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.device)

    def test(self, users_to_test, is_val):
        self.model_mm.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                                                self.image_iu_graph, self.text_ui_graph,
                                                                self.text_ui_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        now_time = datetime.now()
        run_time = datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        stopping_step = 0
        if args.dataset == 'netflix':
            n_batch = (data_generator.n_train // args.batch_size + 1)
        else:
            n_batch = (data_generator.n_train // args.batch_size + 1) // 10
        best_recall = 0
        if not hasattr(self, 'device'):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        multimodal_config = {"item_attr": (lambda x: item_prof_feat[x], 1)}


        sample_loss_accumulator = {
            'struct_pos': 0., 'semantic_pos': 0.,
            'struct_neg': 0., 'semantic_neg': 0.,
            'valid_batch_count': 0,
            'pos_intersection_count': 0,
            'neg_intersection_count': 0
        }


        INTERSECT_W = 3
        INTERSECT_W_1 = 10.0
        NORMAL_W = 1
        NORMAL_W_1 = 1

        for epoch in range(args.epoch):
            t1 = time()
            total_epoch_loss = 0.
            total_bpr_loss = 0.
            total_contrast = 0.


            for key in sample_loss_accumulator.keys():
                sample_loss_accumulator[key] = 0.

            sample_time = 0.

            for idx in tqdm(range(n_batch)):
                self.model_mm.train()
                sample_t1 = time()


                base_users, random_pos_base, random_neg_base, struct_pos_base, struct_neg_base, semantic_pos_base, semantic_neg_base = data_generator.sample()

                base_batch_size = len(base_users)
                if base_batch_size == 0:
                    continue
                sample_loss_accumulator['valid_batch_count'] += 1

                sample_time += time() - sample_t1
                total_loss_tensor = torch.tensor(0.0, device=self.device)


                user_presentation_h, item_presentation_h, image_i_feat, text_i_feat, image_u_feat, text_u_feat \
                    , user_prof_feat_pre, item_prof_feat_pre, user_prof_feat, item_prof_feat, user_att_feats, item_att_feats, i_mask_nodes, u_mask_nodes \
                    = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph,
                                    self.text_ui_graph, self.text_ui_graph)

                sample_time += time() - sample_t1


                ratio_pos = random_pos_base.shape[0] // base_batch_size
                ratio_neg = random_neg_base.shape[0] // base_batch_size

                bpr_pos_items = random_pos_base.view(base_batch_size, ratio_pos)[:, 0]
                bpr_neg_items = semantic_neg_base.view(base_batch_size, ratio_neg)[:, 0]

                bpr_loss, reg_loss = self.cal_bpr_loss(
                    base_users, bpr_pos_items, bpr_neg_items,
                    user_presentation_h, item_presentation_h
                )

                total_loss_tensor = total_loss_tensor + bpr_loss + reg_loss
                total_bpr_loss += bpr_loss.item()


                for modal_name, (get_emb_func, modal_weight) in multimodal_config.items():
                    try:
                        l1 = struct_pos_base.numel()
                        l2 = semantic_pos_base.numel()
                        l3 = struct_neg_base.numel()
                        l4 = semantic_neg_base.numel()

                        min_len = min(l1, l2, l3, l4)
                        struct_pos = struct_pos_base.view(-1)[:min_len].view(-1, 1)
                        semantic_pos = semantic_pos_base.view(-1)[:min_len].view(-1, 1)
                        struct_neg = struct_neg_base.view(-1)[:min_len].view(-1, 1)
                        semantic_neg = semantic_neg_base.view(-1)[:min_len].view(-1, 1)

                        pos_ids = torch.cat([struct_pos, semantic_pos], dim=1)
                        neg_ids = torch.cat([struct_neg, semantic_neg], dim=1)

                        anchor_base = get_emb_func(bpr_pos_items)

                        current_batch_size = anchor_base.size(0)


                        K = (min_len + current_batch_size - 1) // current_batch_size


                        anchor_emb_expanded = anchor_base.unsqueeze(1).repeat(1, K, 1).view(-1, self.emb_dim)


                        if anchor_emb_expanded.size(0) > min_len:
                            anchor_emb_expanded = anchor_emb_expanded[:min_len, :]


                        actual_batch_size = min_len


                        pos_weight = torch.full((actual_batch_size, 2), NORMAL_W, device=self.device)
                        neg_weight = torch.full((actual_batch_size, 2), NORMAL_W_1, device=self.device)



                        if min_len % current_batch_size == 0:

                            s_pos_matrix = struct_pos.view(current_batch_size, -1)  # [B, K]
                            m_pos_matrix = semantic_pos.view(current_batch_size, -1)  # [B, K]

                            pos_intersect_matrix = (s_pos_matrix.unsqueeze(2) == m_pos_matrix.unsqueeze(1))
                            user_has_pos_inter = pos_intersect_matrix.any(dim=1).any(dim=1)  # [B]

                            sample_loss_accumulator['pos_intersection_count'] += user_has_pos_inter.sum().item()


                            ratio = min_len // current_batch_size
                            pos_weight_mask = user_has_pos_inter.unsqueeze(1).repeat(1, ratio).view(-1)

                            if pos_weight_mask.any():
                                pos_weight[pos_weight_mask, 0] = INTERSECT_W
                                pos_weight[pos_weight_mask, 1] = INTERSECT_W


                            s_neg_matrix = struct_neg.view(current_batch_size, -1)
                            m_neg_matrix = semantic_neg.view(current_batch_size, -1)

                            neg_intersect_matrix = (s_neg_matrix.unsqueeze(2) == m_neg_matrix.unsqueeze(1))
                            user_has_neg_inter = neg_intersect_matrix.any(dim=1).any(dim=1)

                            sample_loss_accumulator['neg_intersection_count'] += user_has_neg_inter.sum().item()

                            neg_weight_mask = user_has_neg_inter.unsqueeze(1).repeat(1, ratio).view(-1)

                            if neg_weight_mask.any():
                                neg_weight[neg_weight_mask, 0] = INTERSECT_W_1
                                neg_weight[neg_weight_mask, 1] = INTERSECT_W_1

                        positive_emb = get_emb_func(pos_ids.view(-1)).view(actual_batch_size, 2, self.emb_dim)
                        negative_emb = get_emb_func(neg_ids.view(-1)).view(actual_batch_size, 2, self.emb_dim)

                        modal_total_loss, modal_loss_components = self.info_nce_loss_with_batch_neg(
                            anchor_emb_expanded, positive_emb, negative_emb,
                            pos_weight=pos_weight, neg_weight=neg_weight,
                            tau=args.tau
                        )
                        cl_loss = modal_total_loss
                        total_loss_tensor = total_loss_tensor + cl_loss

                        total_contrast += modal_total_loss.item()


                        for sample_type, loss_val in modal_loss_components.items():
                            if sample_type in sample_loss_accumulator:
                                if isinstance(loss_val, torch.Tensor):
                                    sample_loss_accumulator[sample_type] += loss_val.item()
                                else:
                                    sample_loss_accumulator[sample_type] += loss_val



                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error in CL loss calculation ({modal_name}): {e}")
                        continue


                if not torch.isnan(total_loss_tensor) and not torch.isinf(total_loss_tensor):
                    nn.utils.clip_grad_norm_(self.model_mm.parameters(), max_norm=1.0)
                    self.optimizer.zero_grad()
                    total_loss_tensor.backward(retain_graph=False)
                    self.optimizer.step()

                total_epoch_loss += total_loss_tensor.item()
                torch.cuda.empty_cache()


            valid_batch = sample_loss_accumulator['valid_batch_count']
            avg_sample_losses = {}
            for sample_type in ['struct_pos', 'semantic_pos', 'struct_neg', 'semantic_neg']:
                avg_sample_losses[sample_type] = sample_loss_accumulator[
                                                     sample_type] / valid_batch if valid_batch > 0 else 0.


            pos_int_rate = sample_loss_accumulator['pos_intersection_count'] / valid_batch if valid_batch > 0 else 0.
            neg_int_rate = sample_loss_accumulator['neg_intersection_count'] / valid_batch if valid_batch > 0 else 0.

            epoch_time = time() - t1
            perf_str = (
                    f'Epoch {epoch} [%.1fs]: Total=%.4f | BPR=%.4f | CL=%.4f | '
                    f'PosInterCount=%.1f, NegInterCount=%.1f | '
                    f'PosL[S=%.3f, M=%.3f] NegL[S=%.3f, M=%.3f]'
                    % (epoch_time, total_epoch_loss, total_bpr_loss,
                       total_contrast / valid_batch if valid_batch > 0 else 0.,
                       pos_int_rate, neg_int_rate,
                       avg_sample_losses['struct_pos'], avg_sample_losses['semantic_pos'],
                       avg_sample_losses['struct_neg'], avg_sample_losses['semantic_neg'])
            )
            self.logger.logging(perf_str)


            if math.isnan(total_epoch_loss):
                self.logger.logging(f'ERROR: Epoch {epoch} loss is nan!')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                training_time_list.append(epoch_time)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            self.model_mm.eval()
            with torch.no_grad():
                ret = self.test(users_to_test, is_val=False)
            training_time_list.append(t2 - t1)

            if args.verbose > 0:
                perf_str = (
                        f'Epoch {epoch} [%.1fs + %.1fs]: train=%.5f, recall=[%.5f, %.5f, %.5f, %.5f], '
                        f'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]'
                        % (t2 - t1, time() - t2, total_epoch_loss,
                           ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][-1],
                           ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1],
                           ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                           ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                )
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                with torch.no_grad():
                    test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging(
                    f"Test_Recall@{eval(args.Ks)[1]}: %.5f, precision=[%.5f], ndcg=[%.5f]"
                    % (test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1])
                )
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging(f'#####Early stopping steps: {stopping_step} #####')
            else:
                self.logger.logging('#####Early stop! #####')
                break

        self.logger.logging(f'Final best test result: {str(test_ret)}')
        return best_recall, run_time


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()