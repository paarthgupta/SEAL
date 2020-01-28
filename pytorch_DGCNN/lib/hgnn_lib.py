import ctypes
import numpy as np
import os
import sys
import torch
import pdb
import scipy

class _hgnn_lib(object):

    def __init__(self):
        pass

    def PrepareSparseMatrices(self, hypergraph_list, use_S_=False):
        S_list = [hypergraph.S if not use_S_ else hypergraph.S_ for hypergraph in hypergraph_list]
        full_I = []
        full_J = []
        full_V = []
        I_prefix = 0
        J_prefix = 0
        list_I = []
        list_J = []
        for i, S in enumerate(S_list):
            n_rows, n_cols = S.shape
            I, J, V = scipy.sparse.find(S)
            full_I += list(I+I_prefix)
            full_J += list(J+J_prefix)
            full_V += list(V)
            list_I += [i]*n_rows
            list_J += list(range(I_prefix, I_prefix + n_rows))
            I_prefix += n_rows
            J_prefix += n_cols
        n2f_sp = torch.sparse.FloatTensor(torch.LongTensor([full_I, full_J]), torch.FloatTensor(full_V), torch.Size([I_prefix, J_prefix]))
        f2n_sp = torch.sparse.FloatTensor(torch.LongTensor([full_J, full_I]), torch.FloatTensor(full_V), torch.Size([J_prefix, I_prefix]))
        subhg_sp = torch.sparse.FloatTensor(torch.LongTensor([list_I, list_J]), torch.LongTensor([1]*len(list_I)), torch.Size([len(S_list), I_prefix]))
        return n2f_sp, f2n_sp, subhg_sp

HGNNLIB = _hgnn_lib()

