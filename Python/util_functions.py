from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
#import cPickle as cp
import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph, HGNNHypergraph
import node2vec
import multiprocessing as mp
from collections import defaultdict
from scipy.sparse import csr_matrix
sys.path.append('/home2/e1-313-15477/hynetworkx/src')
from data_preparer import clean_train_hypergraph


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

def links2subgraphs(A, S, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, 
                    node_information=None, edge_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    max_f_label = {'value': 0}
    def helper(A, S, links, g_label):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, S, h, max_nodes_per_hop, node_information, edge_information) for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
#         g_list = [GNNGraph(g, g_label, n_labels, n_features) for g, _, n_labels, n_features in results]
        hyg_list = [HGNNHypergraph(S_g, g_label, n_labels, n_features, f_labels) for _, S_g, n_labels, f_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, _, n_labels, _, _ in results]), max_n_label['value'])
        max_f_label['value'] = max(max([max(f_labels) if len(f_labels)>0 else 0 for _, _, _, f_labels, _ in results]), max_f_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return hyg_list

    print('Enclosing subgraph extraction begins...')
#     train_graphs = helper(A, train_pos, 1) + helper(A, train_neg, 0)
#     test_graphs = helper(A, test_pos, 1) + helper(A, test_neg, 0)
    train_hypergraphs = helper(A, S, train_pos, 1) + helper(A, S, train_neg, 0)
    print("train_hypergraph_done")
    test_hypergraphs = helper(A, S, test_pos, 1) + helper(A, S, test_neg, 0)
    print("test_hypergraph_done")
    print(max_n_label, max_f_label)
    return train_hypergraphs, test_hypergraphs, max_n_label['value'], max_f_label['value']


def parallel_worker(x):
    return subgraph_extraction_labeling(*x)
    

def filter_hypergraph(S, nodes):
    S_cleaned = clean_train_hypergraph(S, csr_matrix(([1, 1], ([nodes[0], nodes[1]], [nodes[1], nodes[0]])), shape=(S.shape[0], S.shape[0])))
    F_dict = incidence_to_hyperedges(S_cleaned, _type=dict)
    matching_F_dict = {j: f for j, f in F_dict.items() if f.issubset(set(nodes))}
    I = nodes
    J = list(matching_F_dict)
    try:
        S_g = S_cleaned[I, :][:, J]
    except IndexError:
        print('Caught exception', type(I), type(J), S_cleaned.shape)
        raise (IndexError)
    return S_g


def subgraph_extraction_labeling(ind, A, S, h=1, max_nodes_per_hop=None, node_information=None, edge_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    S_g = filter_hypergraph(S, nodes)
    # apply node-labeling
    labels, f_labels = node_label(subgraph, S_g)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
#     if edge_information is not None:
#         features = edge_information[edges]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    # TODO: remove edge 0, 1 from S_g also
    return g, S_g, labels.tolist(), f_labels.tolist(), features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def S_to_bipartite(S):
    B = nx.bipartite.from_biadjacency_matrix(S)
    return B



def node_label(subgraph, S):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    
    u = 0
    v = 1
    B = S_to_bipartite(S)
    dist_to_u = {i-S.shape[0]: j/2 for i, j in nx.single_source_shortest_path_length(B, u).items() if i >= S.shape[0]}
    dist_to_v = {i-S.shape[0]: j/2 for i, j in nx.single_source_shortest_path_length(B, v).items() if i >= S.shape[0]}
    
    if len(dist_to_u) == 0:
#         print('empty distance')
        dist_to_u = np.ones(S.shape[1])*np.inf
    else:
#         print('non-empty distance')
        I, V = list(zip(*dist_to_u.items()))
        dist_to_u = np.array(csr_matrix((V, (I, [0]*len(I))), shape=(S.shape[1], 1)).todense()).ravel()
    if len(dist_to_v) == 0:
#         print('empty distance')
        dist_to_v = np.ones(S.shape[1])*np.inf
    else:
#         print('non-empty distance')
        I, V = list(zip(*dist_to_v.items()))
        dist_to_v = np.array(csr_matrix((V, (I, [0]*len(I))), shape=(S.shape[1], 1)).todense()).ravel()
    d = (dist_to_u + dist_to_v).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    f_labels = 1 + np.minimum(dist_to_u, dist_to_v).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
#     f_labels = np.concatenate((np.array([1, 1]), f_labels))
    f_labels[np.isinf(f_labels)] = 0
    f_labels[f_labels>1e6] = 0  # set inf labels to 0
    f_labels[f_labels<-1e6] = 0  # set -inf labels to 0
    return labels, f_labels


def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings



def incidence_to_hyperedges(S, silent_mode=True, _type=set):
    I, J = S.nonzero()
    hyperedges = defaultdict(set)
    indices = list(zip(I, J))
    if not silent_mode:
        print('Converting incidence matrix to hyperedge {} for faster processing...'.format(_type))
    for i, j in (tqdm(indices) if not silent_mode else indices):
        hyperedges[j].add(i)
    if _type == set:
        return set(map(frozenset, hyperedges.values()))
    elif _type == list:
        return set(map(frozenset, hyperedges.values()))
    elif _type == dict:
        return {i: frozenset(f) for i, f in hyperedges.items()}
    return hyperedges


def hyperedges_to_incidence(hyperedges, nV):
    nF = len(hyperedges)
    hyperedges = list(set(hyperedges))
    I = []
    J = []
    for j, f in enumerate(hyperedges):
        I.extend(f)
        J.extend([j] * len(f))
    S = csr_matrix(([1] * len(I), (I, J)), shape=(nV, nF))
    return S


def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

