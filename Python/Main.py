import torch
import pdb
import numpy as np
import sys, copy, math, time, pdb
# import cPickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
from sklearn.decomposition import NMF
from joblib import Memory
data_path = '/storage2/home2/e1-313-15477/SEAL'
cachedir = os.path.join(data_path, 'seal_cache')
memory = Memory(cachedir, verbose=0)


parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data-name', default='USAir', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.5,
                    help='ratio of test links')                             # changed for small S 
# model settings
parser.add_argument('--hop', default=3, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-edge-embedding', action='store_true', default=False,
                    help='whether to use node2vec edge embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
parser.add_argument('--use-edge-attribute', action='store_true', default=False,
                    help='whether to use edge attributes')

args = parser.parse_args('')
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

# if args.train_name is None:
#     args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
#     data = sio.loadmat(args.data_dir)
#     net = data['net']
#     if 'group' in data:
#         # load node attributes (here a.k.a. node classes)
#         attributes = data['group'].toarray().astype('float32')
#     else:
#         attributes = None
#     # check whether net is symmetric (for small nets only)
#     if False:
#         net_ = net.toarray()
#         assert(np.allclose(net_, net_.T, atol=1e-8))
#     #Sample train and test links
#     train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num)
# else:
#     args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
#     args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
#     train_idx = np.loadtxt(args.train_dir, dtype=int)
#     test_idx = np.loadtxt(args.test_dir, dtype=int)
#     max_idx = max(np.max(train_idx), np.max(test_idx))
#     net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
#     net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
#     net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
#     #Sample negative train and test links
#     train_pos = (train_idx[:, 0], train_idx[:, 1])
#     test_pos = (test_idx[:, 0], test_idx[:, 1])
#     train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)

    
    
''' Reading Hypergraph Data '''

sys.path.append('/home2/e1-313-15477/hynetworkx/src')
from experimenter import *
from utils import *
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm_notebook
from tabulate import tabulate
from data_preparer import S_to_A,clean_train_hypergraph
mcm = mixed_combinations_map
if 'HDPa' in mcm['PA']:
    mcm['PA'].remove('HDPa')

data_names = [
              'email-Enron',
              'contact-high-school',
              'NDC-substances',
              'tags-math-sx',
              'threads-math-sx',
              'coauth-DBLP'
             ]
split_modes = [
               'structural',
               'temporal',
              ]
data_name = data_names[0]
print('DATASET: {}'.format(data_name))
split_mode = 'structural'
params = get_default_params()
params['data_params']['data_name'] = data_name
params['data_params']['split_mode'] = split_mode
params['data_params']['base_path'] = '/home2/e1-313-15477/'
dfs, train_dfs, test_dfs, overfit_dfs = [], [], [], []
params['iter_var'] = 0
data_params = params['data_params']
lp_data_params = params['lp_data_params']
lp_params = params['lp_params']
iter_var = params['iter_var']


@memory.cache
def initialize(data_params,lp_data_params,args):

    data_name, base_path, split_mode, max_size_limit = [data_params[x] for x in
                                                            ['data_name', 'base_path', 'split_mode', 'max_size_limit']]
    rho, neg_factor, neg_mode = [lp_data_params[x] for x in
                                    ['rho', 'neg_factor', 'neg_mode']]
    S, times, id_label_map = parse_S(data_name,
                                        base_path,
                                        split_mode,
                                        max_size_limit,
                                        *get_time_filter_params(data_name))
    
    my_net = S_to_A(S, False)
    train_pos, train_neg, test_pos, test_neg = sample_neg(my_net, args.test_ratio, args.max_train_num)
    S_train = clean_train_hypergraph(S, csr_matrix(([1]*len(test_pos[0]), (test_pos[0],test_pos[1])),
                                               shape= (S.shape[0],S.shape[0])))
    A = my_net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links

    node_information = None
    if args.use_embedding:
        embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
        node_information = embeddings
    if args.use_attribute and attributes is not None:
        if node_information is not None:
            node_information = np.concatenate([node_information, attributes], axis=1)
        else:
            node_information = attributes

    edge_information = None
    B = S_to_A(S.T, False)
    if args.use_edge_embedding:                                                         
        edge_embeddings = generate_node2vec_embeddings(B, 128, True, train_neg)
        edge_information = edge_embeddings
    if args.use_edge_attribute and edge_attributes is not None:
        if edge_information is not None:
            edge_information = np.concatenate([edge_information, edge_attributes], axis=1)
        else:
            edge_information = edge_attributes

    train_hypergraphs, test_hypergraphs, max_n_label, max_f_label = links2subgraphs(A, S_train,train_pos, train_neg, test_pos,                                                                                       test_neg, args.hop, args.max_nodes_per_hop,                                                                                               node_information,edge_information)

    return train_hypergraphs, test_hypergraphs, max_n_label, max_f_label,edge_information,node_information
    
    
train_hypergraphs, test_hypergraphs, max_n_label, max_f_label,edge_information,node_information = initialize(data_params,lp_data_params,args)     
    
    
    
# a = [0,0, 1,1, 2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6,    7,7,7, 8,8,8,8,    9,9,9,9]
# b = [0,9, 1,2, 3,4, 2,4,6, 5,6,7, 5,7,8, 10,11,12, 1,9,2, 3,10,13,14, 3,10,13,14]

# S = csr_matrix(([1]*len(a),(b,a)))
# print("S = ",S.todense())

# test_ratio = args.test_ratio
# max_train_num=args.max_train_num

# @memory.cache
# def initialize_1(S, test_ratio, max_train_num):
#     my_net = S_to_A(S, False)
#     train_pos, train_neg, test_pos, test_neg = sample_neg(my_net, test_ratio, max_train_num)
#     S_train = clean_train_hypergraph(S, csr_matrix(([1]*len(test_pos[0]), (test_pos[0],test_pos[1])),
#                                                shape= (S.shape[0],S.shape[0])))

#     return S_train,train_pos, train_neg, test_pos, test_neg,my_net

# S_train,train_pos, train_neg, test_pos, test_neg,my_net = initialize_1(S, test_ratio, max_train_num)
    

# k =  int(2*(S_train.shape[0]))
# print("nmf start")
# model = NMF(k)
# print(S_train.shape)
# S_train = model.fit_transform(S_train)
# print(S_train.shape)
# print("nmf done")
   
# '''Train and apply classifier'''
# A = my_net.copy()  # the observed network
# A[test_pos[0], test_pos[1]] = 0  # mask test links
# A[test_pos[1], test_pos[0]] = 0  # mask test links

# node_information = None
# if args.use_embedding:
#     embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
#     node_information = embeddings
# if args.use_attribute and attributes is not None:
#     if node_information is not None:
#         node_information = np.concatenate([node_information, attributes], axis=1)
#     else:
#         node_information = attributes
        
# edge_information = None
# B = S_to_A(S.T, False)
# if args.use_edge_embedding:                                                         
#     edge_embeddings = generate_node2vec_embeddings(B, 128, True, train_neg)
#     edge_information = edge_embeddings
# if args.use_edge_attribute and edge_attributes is not None:
#     if edge_information is not None:
#         edge_information = np.concatenate([edge_information, edge_attributes], axis=1)
#     else:
#         edge_information = edge_attributes

# hop = args.hop
# max_nodes_per_hop = args.max_nodes_per_hop

# @memory.cache
# def initialize_2(A, S_train, train_pos, train_neg, test_pos, test_neg,hop, 
#                  max_nodes_per_hop, node_information,edge_information):
    
#     train_hypergraphs, test_hypergraphs, max_n_label, max_f_label = links2subgraphs(A, S_train,train_pos, train_neg, test_pos,                                                                                       test_neg, hop, max_nodes_per_hop,                                                                                               node_information,edge_information)
    
#     return train_hypergraphs, test_hypergraphs, max_n_label, max_f_label

# train_hypergraphs, test_hypergraphs, max_n_label, max_f_label = initialize_2(A, S_train, train_pos, train_neg, test_pos,                                                                                      test_neg,hop, max_nodes_per_hop, node_information,
#                                                                              edge_information)
# pdb.set_trace()
# def graphs2hypergraphs(S, graphs):
#     hypergraphs = []
#     F = incidence_to_hyperedges(S)
#     for g in train_graphs:
#         nodes = set(g.nodes())
#         matching_F = {f for f in F if f.issubset(nodes)}
#         S_g = hyperedge_to_incidence(matching_F)
#         hypergraph = HGNNHypergraph(S_g, g.label, g.node_tags, g.node_features)
#         hypergraphs.append(hypergraph)
#     return hypergraphs

# train_hypergraphs, test_hypergraphs = graphs2hypergraphs(S_train, train_graphs), graphs2hypergraphs(S_train, test_graphs)

print('# train: %d, # test: %d' % (len(train_hypergraphs), len(test_hypergraphs)))

# DGCNN configurations
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [64, 32, 32, 16, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
# cmd_args.dropout_prob = 0.9
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.num_epochs = 40
cmd_args.learning_rate = 1e-4
cmd_args.batch_size = 5                                             # default was 50 
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.edge_feat_dim = max_f_label + 1
cmd_args.attr_dim = 0
cmd_args.edge_attr_dim = 0

if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
    
if edge_information is not None:
    cmd_args.edge_attr_dim = edge_information.shape[1]
    
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_hypergraphs + test_hypergraphs])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

#pdb.set_trace()
classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_hypergraphs)))
best_loss = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(train_hypergraphs, classifier, train_idxes, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

    classifier.eval()
    test_loss = loop_dataset(test_hypergraphs, classifier, list(range(len(test_hypergraphs))))
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

with open('acc_results.txt', 'a+') as f:
    f.write(str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(test_loss[2]) + '\n')

