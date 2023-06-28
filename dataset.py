from functools import partial
from sys import prefix
from sklearn.inspection import partial_dependence
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate, random_split
from tqdm import tqdm
import dgl
import json
from gensim.models import Word2Vec, KeyedVectors
from copy import deepcopy
import numpy as np


class MisuseDataset(Dataset):
    def __init__(self, 
                 train_examples, # make sure that train_examples less than total number
                 train_file_path='train.txt'
                 ) -> None:
        super().__init__()

        self.size = train_examples
        self.filepath = train_file_path
        self.lines = open(self.filepath, 'r', encoding='utf-8').read().splitlines()[:train_examples]

        self.astnode_w2v = KeyedVectors.load('w2v_astnodes.bin')
        self.codetext_w2v = KeyedVectors.load('w2v_codetext.bin')

    def parAST_to_dgl_graph(self, par_ast):
        # TODO: transition partial ast to dgl.graph
        u = []
        v = []
        nodes_idx = [n['idx'] for n in par_ast]
        idx_type_dict = {}
        for node in par_ast:
            idx_type_dict[node['idx']] = node['type']
        
        for node in par_ast:
            if node.get("children") != None:
                for child_idx in node['children']:
                    if child_idx in nodes_idx:
                        u.append(node['idx'])
                        v.append(child_idx)

        g = dgl.graph((u, v))
        features = []
        for i in range(g.num_nodes()):
            if i in nodes_idx:
                features.append(self.astnode_w2v[idx_type_dict[i]])
            else:
                features.append(np.zeros(self.astnode_w2v.vector_size))

        g.ndata['annotation'] = torch.tensor(np.array(features))
        g.edata['type'] = torch.ones(g.num_edges(), dtype=torch.int32)
        return g

    def intercept_codetext_basedon_partialAST(self, codetext, ast_nodes):
        # TODO: Intercept the codetext, to make it corresponding to the partial ast.
        return codetext

    def __len__(self):
        return len(self.size)

    def __getitem__(self, index):
        data_dict = json.loads(self.lines[index])
        ast_nodes = data_dict['ast']
        next_node_type = data_dict['nxt']
        src_path = data_dict['src_path']

        return self.parAST_to_dgl_graph(ast_nodes), \
               self.intercept_codetext_basedon_partialAST(open(src_path, 'r', encoding='utf-8').read(), ast_nodes), \
               torch.tensor(self.astnode_w2v[next_node_type])
