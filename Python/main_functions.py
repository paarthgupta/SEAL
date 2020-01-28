import collections
from joblib import Memory
import os
import sys
from util_functions import get_library_path, get_base_path
library_path = get_library_path()
sys.path.append(os.path.join(library_path, 'hynetworkx/src/'))
from experimenter import *
from utils import *
from data_preparer import S_to_A, clean_train_hypergraph
from scipy.sparse import csr_matrix
from util_functions import *
sys.path.append(os.path.join(library_path, 'pytorch_DGCNN'))
from main import *

data_path = get_base_path()
cachedir = os.path.join(data_path, 'seal_cache/new/')
memory = Memory(cachedir, verbose=1)

modes = [
         'clique', # 0
         'uniclique', # 1
         'biclique', # 2
         'unistar', # 3
         'bistar', # 4
         'unistar-clique', # 5
         'bistar-clique', # 6
         'unistar-uniclique', # 7
         'bistar-uniclique', # 8
         'unistar-biclique', # 9
         'bistar-biclique', # 10
         'star-clique-tripartite' # 11
        ]
'''
clique:
c-------c
 \     /
  \   /
   \ /
    c

uniclique:
c---c---c
 \     /
  c   c
   \ /
    c


biclique:
c---s---c
 \     /
  s   s
   \ /
    c

unistar:
c   c
 \ /   
  c
  | 
  c

bistar:
c   c
 \ /   
  s
  | 
  c

unistar-clique:
c-------c
 \\   //
  \ c /
   \|/
    c

bistar-clique
c-------c
 \\   //
  \ s /
   \|/
    c

unistar-uniclique:
c---c---c
 \\   //
  c c c
   \|/
    c

bistar-uniclique:
c---c---c
 \\   //
  c s c
   \|/
    c

unistar-biclique:
c---s---c
 \\   //
  s c s
   \|/
    c

bistar-biclique:
c---s---c
 \\   //
  s s s
   \|/
    c

star-clique-tripartite:
c---s---c
 \\   //
  s t s
   \|/
    c
'''

Data = collections.namedtuple('Data', ['S'])
LPData = collections.namedtuple('LPData', ['S', 'node_emb', 'hyperedge_emb', 'train_pos', 'train_neg',
                                           'test_pos', 'test_neg'])
SubgraphData = collections.namedtuple('SubgraphData', ['train', 'test', 'label_dim'])
NNResults = collections.namedtuple('NNResults', ['train', 'test'])

def get_default_data():
    a = [0,0, 1,1, 2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6,    7,7,7, 8,8,8,8,    9,9,9,9]
    b = [0,9, 1,2, 3,4, 2,4,6, 5,6,7, 5,7,8, 10,11,12, 1,9,2, 3,10,13,14, 3,10,13,14]

    S = csr_matrix(([1]*len(a),(b,a)))
    data = Data(S)
    return data


@memory.cache
def read_data(data_params):
    data_name, base_path, split_mode, max_size_limit = [data_params[x] for x in
                                                            ['data_name', 'base_path', 'split_mode', 'max_size_limit']]
    S, times, id_label_map = parse_S(data_name,
                                    base_path,
                                    split_mode,
                                    max_size_limit,
                                    *get_time_filter_params(data_name))
    data = Data(S)
    return data

@memory.cache
def prepare_lp_data(data, lp_data_params):
    A = S_to_A(data.S, False)
    train_pos, train_neg, test_pos, test_neg = sample_neg(A, lp_data_params['rho'])
    S_train = clean_train_hypergraph(data.S, csr_matrix(([1]*len(test_pos[0]), (test_pos[0],test_pos[1])),
                                               shape= (data.S.shape[0], data.S.shape[0])))
    lp_data = LPData(S_train, None, None, train_pos, train_neg, test_pos, test_neg)
    return lp_data

@memory.cache
def extract_embeddings(lp_data, emb_params):
    node_information = None
    if emb_params['node_dim'] is not None:
        embeddings = generate_node2vec_embeddings(S_to_A(lp_data.S, False), emb_params['node_dim'], True, lp_data.train_neg)
        node_information = embeddings

    # TODO: hyperedge-embeddings need to be rectified!
    hyperedge_information = None
    if emb_params['hyperedge_dim'] is not None:
        hyperedge_embeddings = generate_node2vec_embeddings(S_to_A(lp_data.S.T, False), emb_params['hyperedge_dim'], True, lp_data.train_neg)
        hyperedge_information = hyperedge_embeddings
    lp_data = LPData(lp_data.S, node_information, hyperedge_information,
                     lp_data.train_pos, lp_data.train_neg, lp_data.test_pos, lp_data.test_neg)
    return lp_data

@memory.cache
def extract_subgraphs(lp_data, subgraph_params):
    
    train_graphs, test_graphs, max_n_label = \
        links2subgraphs(lp_data.S,
                        lp_data.train_pos, lp_data.train_neg, lp_data.test_pos, lp_data.test_neg,
                        subgraph_params['hop'], subgraph_params['max_nodes_per_hop'],
                        lp_data.node_emb, lp_data.hyperedge_emb, subgraph_params['mode'])
    subgraph_data = SubgraphData(train_graphs, test_graphs, max_n_label)
    return subgraph_data

def classify(subgraph_data, classifier_params, silent = True):
    cmd_args.gm = classifier_params['nn']
    cmd_args.sortpooling_k = classifier_params['sortpooling_k']
    cmd_args.latent_dim = classifier_params['latent_dim']
    cmd_args.hidden = classifier_params['hidden']
    cmd_args.out_dim = classifier_params['out_dim']
    cmd_args.dropout = classifier_params['dropout']
    # cmd_args.dropout_prob = 0.9
    cmd_args.num_class = classifier_params['num_class']
    cmd_args.mode = classifier_params['mode']
    cmd_args.num_epochs = classifier_params['num_epochs']
    cmd_args.learning_rate = classifier_params['learning_rate']
    cmd_args.batch_size = classifier_params['batch_size']
    cmd_args.feat_dim = int(subgraph_data.label_dim) + 1
    # cmd_args.edge_feat_dim = subgraph + 1
    cmd_args.attr_dim = classifier_params['attr_dim']
    cmd_args.edge_attr_dim = classifier_params['edge_attr_dim']

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in subgraph_data.train + subgraph_data.test])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    #pdb.set_trace()
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(subgraph_data.train)))
    best_loss = None
    train_metrics = []
    test_metrics = []
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(subgraph_data.train, classifier, train_idxes, optimizer=optimizer)
        train_metrics.append(dict(zip(['loss', 'acc', 'auc'], avg_loss)))
        if not silent:
            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        classifier.eval()
        test_loss = loop_dataset(subgraph_data.test, classifier, list(range(len(subgraph_data.test))))
        test_metrics.append(dict(zip(['loss', 'acc', 'auc'], test_loss)))
        if not silent:
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))
    results = NNResults(train_metrics, test_metrics)
    return results