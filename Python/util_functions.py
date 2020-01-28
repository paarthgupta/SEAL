from __future__ import print_function
def get_base_path():
    return '/home2/e1-313-15477/'
#     return '/content/drive/My Drive/Colab Notebooks/data/'
def get_library_path():
    return '/home2/e1-313-15477/'
#     return '/content/drive/My Drive/Colab Notebooks/libraries/'

import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
# import cPickle as cp
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
lib_path = get_library_path()
sys.path.append(os.path.join(lib_path, 'pytorch_DGCNN'))
from util import GNNGraph
sys.path.append(os.path.join(lib_path, 'SEAL/Python/software/node2vec/src'))
import node2vec
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix,hstack, vstack

sys.path.append('/home2/e1-313-15477/hynetworkx/src')
from data_preparer import S_to_A, A_to_S

default_mode = 'clique'

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

def get_Z(n_rows, n_cols):
    return csr_matrix(([], ([], [])), shape=(n_rows, n_cols))

def stack_quadrant(tl, tr, bl, br):
    if (tl is None and tr is None) or (bl is None and br is None) or \
       (tl is None and bl is None) or (tr is None and br is None):
        print('Warning: Unstackable! Size of zero matrices not known.')
        return None
    if tl is None:
        tl = get_Z(tr.shape[0], bl.shape[1])
    if tr is None:
        tr = get_Z(tl.shape[0], br.shape[1])
    if bl is None:
        bl = get_Z(br.shape[0], tl.shape[1])
    if br is None:
        br = get_Z(bl.shape[0], tr.shape[1])
    l = vstack([tl, bl])
    r = vstack([tr, br])
    return hstack([l, r]).tocsr()

def links2subgraphs(S, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, node_information=None, hyperedge_information = None, mode=default_mode):
    # extract enclosing subgraphs        
    max_n_label = {'value': 0}
    def helper(A, links, g_label, h, num_nodes, make_bip=False, node_information = None):
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
        results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])])
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
        g_list = [GNNGraph(g ,g_label, n_labels, n_features, num_nodes, make_bip) for g, n_labels, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ in results]), max_n_label['value'])
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    try:
        if mode != 'clique':
            node_information = node_information + hyperedge_information
    except TypeError:
        # If he_info is None and node_info not None, make node_info None.
        node_information = None
        
    if mode == 'clique':
        A = S_to_A(S, False)
        make_bip = False
    elif mode == 'uniclique':
        A_ = S_to_A(S, False)
        S_ = A_to_S(A_)
        A = stack_quadrant(None, S_, S_.T, None)
        make_bip = False
    elif mode == 'biclique':
        A_ = S_to_A(S, False)
        S_ = A_to_S(A_)
        A = stack_quadrant(None, S_, S_.T, None)
        make_bip = True
    elif mode == 'unistar':
        A = stack_quadrant(None, S, S.T, None)
        make_bip = False
    elif mode == 'bistar':
        A = stack_quadrant(None, S, S.T, None)
        make_bip = True
    elif mode == 'unistar-clique':
        A_ = S_to_A(S, False)
        A = stack_quadrant(A_, S, S.T, None)
        make_bip = False
    elif mode == 'bistar-clique':
        A_ = S_to_A(S, False)
        A = stack_quadrant(A_, S, S.T, None)
        make_bip = True
    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1, h, S.shape[0], make_bip, node_information) + helper(A, train_neg, 0, h, S.shape[0], make_bip, node_information)
    test_graphs = helper(A, test_pos, 1, h, S.shape[0], make_bip, node_information) + helper(A, test_neg, 0, h, S.shape[0], make_bip, node_information)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
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
    # apply node-labeling
    labels = node_label(subgraph)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    return g, labels.tolist(), features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph):
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
    return labels

    
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