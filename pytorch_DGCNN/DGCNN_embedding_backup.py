from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from hgnn_lib import HGNNLIB
from pytorch_util import weights_init, gnn_spmm


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU', latent_edge_feat_dim = None):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim
        
        
        latent_edge_feat_dim = 0  ###############################################################################33
        
        
        if latent_edge_feat_dim is None:
            latent_edge_feat_dim = num_node_feats

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + latent_edge_feat_dim, latent_dim[0]))
#         self.conv_params.append(nn.Linear(35, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i-1]))
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))
        
        self.conv_params.append(nn.Linear(latent_dim[-1], latent_dim[-1]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_edge_feat_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def forward(self, hypergraph_list, node_feat, edge_feat):
        hypergraph_sizes = [hypergraph_list[i].num_nodes for i in range(len(hypergraph_list))]
        node_hdegs = [torch.Tensor(hypergraph_list[i].hdegs) + 1 for i in range(len(hypergraph_list))]
        node_hdegs = torch.cat(node_hdegs).unsqueeze(1)
#         pdb.set_trace()
        hyperedge_sizes = [torch.Tensor(hypergraph_list[i].hyperedge_sizes) + 1 for i in range(len(hypergraph_list))]
        hyperedge_sizes = torch.cat(hyperedge_sizes).unsqueeze(1)

        n2f_sp, f2n_sp, subhg_sp = HGNNLIB.PrepareSparseMatrices(hypergraph_list ,True) 
        
        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2f_sp = n2f_sp.cuda()
            f2n_sp = f2n_sp.cuda()
            subhg_sp = subhg_sp.cuda()
            node_hdegs = node_hdegs.cuda()
            hyperedge_sizes = hyperedge_sizes.cuda()
        node_feat = Variable(node_feat)
        
        
        edge_feat=None                        #########################################################
        
        
        
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2f_sp = Variable(n2f_sp)
        f2n_sp = Variable(f2n_sp)
        subhg_sp = Variable(subhg_sp)
        node_hdegs = Variable(node_hdegs)
        hyperedge_sizes = Variable(hyperedge_sizes)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2f_sp, f2n_sp, subhg_sp, hypergraph_sizes, node_hdegs, hyperedge_sizes)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2f_sp, f2n_sp, subhg_sp, hypergraph_sizes, node_hdegs, hyperedge_sizes):
        ''' if exists edge feature, concatenate to node feature vector '''
        
        edge_feat=None           ###########################################################
        
        
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
#             input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(n2f_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < 2*len(self.latent_dim):
#             # OPTION 1
#             f2npool = gnn_spmm(f2n_sp, cur_message_layer)  # Y = S^T * X 
            
#             print(f2npool.shape)
#             print(f2n_sp.shape)
#             print(cur_message_layer.shape)
#             print(node_feat.shape)

#             node_linear = self.conv_params[lv](f2npool)  # Y = Y * W
#             normalized_linear = node_linear.div(hyperedge_sizes)  # Y = sizes^-1 * Y
#             cur_message_layer = torch.tanh(normalized_linear) # Z = tanh(Y)  
#             n2fpool = gnn_spmm(n2f_sp, cur_message_layer)  # Y = S * Z
#             node_linear = nn.Linear(cur_message_layer.shape[1], cur_message_layer.shape[1])(n2fpool)  # Z = Z * W
#             normalized_linear = node_linear.div(node_hdegs)  # Z = D_f^-1 * Z
#             cur_message_layer = torch.tanh(normalized_linear)
#             cat_message_layers.append(cur_message_layer)
#             lv += 1
            
            # OPTION 2
#             f2npool = gnn_spmm(f2n_sp, cur_message_layer)  # Y = S^T * X
#             node_linear = self.conv_params[lv](f2npool)  # Y = Y * W
#             normalized_linear = node_linear.div(hyperedge_sizes)  # Y = sizes^-1 * Y
#             cur_message_layer = torch.tanh(normalized_linear) # Z = tanh(Y)  
#             lv += 1
#             n2fpool = gnn_spmm(n2f_sp, cur_message_layer)  # Y = S * Z
#             node_linear = self.conv_params[lv](n2fpool)  # Z = Z * W
#             normalized_linear = node_linear.div(node_hdegs)  # Z = D_f^-1 * Z
#             cur_message_layer = torch.tanh(normalized_linear)
#             cat_message_layers.append(cur_message_layer)
#             lv += 1
            
            # OPTION 3
#             f2npool = gnn_spmm(f2n_sp, cur_message_layer)  # Y = S^T * X
#             node_linear = self.conv_params[lv](f2npool)  # Y = Y * W
#             normalized_linear = node_linear.div(hyperedge_sizes)  # Y = sizes^-1 * Y
#             node_linear = gnn_spmm(n2f_sp, normalized_linear) # Y = S * Y
#             normalized_linear = node_linear.div(node_hdegs) # Y = D_f^-1 * Z
#             cur_message_layer = torch.tanh(normalized_linear) # Z = tanh(Y)
#             cat_message_layers.append(cur_message_layer)
#             lv += 1

             
            # OPTION 4          
#             n2fpool = gnn_spmm(n2f_sp, cur_message_layer)  # Y = S * Z
            
#             print(n2fpool.shape)
#             print(n2f_sp.shape)
#             print(cur_message_layer.shape)
#             print(edge_feat.shape)
            
#             node_linear = self.conv_params[lv](n2fpool)  # Z = Z * W
#             normalized_linear = node_linear.div(node_hdegs)  # Z = D_f^-1 * Z
#             cur_message_layer = torch.tanh(normalized_linear)
#             cat_message_layers.append(cur_message_layer)
#             lv += 1
            
#             f2npool = gnn_spmm(f2n_sp, cur_message_layer)  # Y = S^T * X
#             node_linear = self.conv_params[lv](f2npool)  # Y = Y * W
#             normalized_linear = node_linear.div(hyperedge_sizes)  # Y = sizes^-1 * Y
#             cur_message_layer = torch.tanh(normalized_linear) # Z = tanh(Y)  
#             lv += 1

# OPTION 2
            f2npool = gnn_spmm(f2n_sp, cur_message_layer)  # Y = S^T * X
            node_linear = self.conv_params[lv](f2npool)  # Y = Y * W
            normalized_linear = node_linear.div(hyperedge_sizes)  # Y = sizes^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear) # Z = tanh(Y)  
            lv += 1
            n2fpool = gnn_spmm(n2f_sp, cur_message_layer)  # Y = S * Z
            node_linear = self.conv_params[lv](n2fpool)  # Z = Z * W
            normalized_linear = node_linear.div(node_hdegs)  # Z = D_f^-1 * Z
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1
        cur_message_layer = torch.cat(cat_message_layers, 1)

        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(hypergraph_sizes), self.k, self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(subhg_sp.size()[0]):
            to_sort = sort_channel[accum_count: accum_count + hypergraph_sizes[i]]
            k = self.k if self.k <= hypergraph_sizes[i] else hypergraph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += hypergraph_sizes[i]

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)

        to_dense = conv1d_res.view(len(hypergraph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)
