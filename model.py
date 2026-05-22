import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MM_Model(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats,
                 user_init_embedding, item_attribute_dict):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size


        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        self.user_trans = nn.Linear(user_init_embedding.shape[1], args.embed_size)
        if args.dataset == 'netflix':
            self.item_trans = nn.Linear(item_attribute_dict['title'].shape[1], args.embed_size)
        else:
            self.item_trans = nn.Linear(item_attribute_dict['Title'].shape[1], args.embed_size)

        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)
        nn.init.xavier_uniform_(self.user_trans.weight)
        nn.init.xavier_uniform_(self.item_trans.weight)


        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)


        self.image_feats = torch.tensor(image_feats).float().to(device)
        self.text_feats = torch.tensor(text_feats).float().to(device)
        self.user_feats = torch.tensor(user_init_embedding).float().to(device)
        self.item_feats = {}
        for key in item_attribute_dict.keys():
            self.item_feats[key] = torch.tensor(item_attribute_dict[key]).float().to(device)

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)

    def mm(self, x, y):
        if 1:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def forward(self, ui_graph, iu_graph):

        i_mask_nodes, u_mask_nodes = None, None

        u_perm = torch.randperm(self.n_users)
        u_num_mask_nodes = 0
        u_mask_nodes = u_perm[: u_num_mask_nodes]
        self.user_feats[u_mask_nodes] = self.user_feats.mean(0)

        image_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = self.dropout(self.text_trans(self.text_feats))
        user_feats = self.dropout(self.user_trans(self.user_feats.to(torch.float32)))

        item_feats = {}
        for key in self.item_feats.keys():
            item_feats[key] = self.dropout(self.item_trans(self.item_feats[key]))

        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        # aug item attribute
        user_feat_from_item = {}
        for key in self.item_feats.keys():
            user_feat_from_item[key] = self.mm(ui_graph, item_feats[key])
            item_feats[key] = self.mm(iu_graph, user_feat_from_item[key])

        # aug user profile
        item_prof_feat = self.mm(iu_graph, user_feats)
        user_prof_feat = self.mm(ui_graph, item_prof_feat)

        u_g_embeddings = self.user_id_embedding.weight
        i_g_embeddings = self.item_id_embedding.weight

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        u_g_embeddings = u_g_embeddings + args.model_cat_rate * F.normalize(image_user_feats, p=2,
                                                                            dim=1) + args.model_cat_rate * F.normalize(
            text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate * F.normalize(image_item_feats, p=2,
                                                                            dim=1) + args.model_cat_rate * F.normalize(
            text_item_feats, p=2, dim=1)
        # profile
        u_g_embeddings += args.user_cat_rate * F.normalize(user_prof_feat, p=2, dim=1)
        i_g_embeddings += args.user_cat_rate * F.normalize(item_prof_feat, p=2, dim=1)

        # attribute
        for key in self.item_feats.keys():
            u_g_embeddings += args.item_cat_rate * F.normalize(user_feat_from_item[key], p=2, dim=1)
            i_g_embeddings += args.item_cat_rate * F.normalize(item_feats[key], p=2, dim=1)

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, user_feats, item_feats, user_prof_feat, item_prof_feat, user_feat_from_item, item_feats, i_mask_nodes, u_mask_nodes
