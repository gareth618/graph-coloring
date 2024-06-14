import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os

from dgl.nn.pytorch import GraphConv
from itertools import chain



def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_adjacency_matrix(nx_graph, torch_device, torch_dtype):

    adj = nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device)

    return adj_


def parse_line(file_line, node_offset):
    x, y = file_line.split(' ')[1:]  # skip first character - specifies each line is an edge definition
    x, y = int(x)+node_offset, int(y)+node_offset  # nodes in file are 1-indexed, whereas python is 0-indexed
    return x, y


def build_graph_from_color_file(fname, node_offset=-1, parent_fpath=''):
    fpath = os.path.join(parent_fpath, fname)
    print(f'Building graph from contents of file: {fpath}')
    with open(fpath, 'r') as f:
        content = f.read().strip()
    start_idx = [idx for idx, line in enumerate(content.split('\n')) if line.startswith('p')][0]
    lines = content.split('\n')[start_idx:]  # skip comment line(s)
    nr_nodes=int(lines[0].split(' ')[2])
    print(nr_nodes)
    edges = [parse_line(line, node_offset) for line in lines[1:] if len(line) > 0]
    print(len(edges))
    nx_temp = nx.from_edgelist(edges)
    print(sorted(nx_temp.nodes()))
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(nr_nodes))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


class GNNConv(nn.Module):


    def __init__(self, g, in_feats, hidden_size, num_classes, dropout):


        super(GNNConv, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_size, activation=F.relu, allow_zero_in_degree=True))
        # output layer
        self.layers.append(GraphConv(hidden_size, num_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)
        self.in_feats = in_feats
        self.hidden_size = hidden_size

    def forward(self, features):


        h = features
        for i, layer in enumerate(self.layers):
            h_in = h  # Preserve the input for the residual connection
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i != len(self.layers) - 1:  # Add residual connection for hidden layers
                if h_in.shape == h.shape:
                    h = h + h_in
        return h


# Construct graph to learn on #
def get_gnn(g, n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):


    try:
        print(f'Function get_gnn(): Setting seed to {gnn_hypers["seed"]}')
        set_seed(gnn_hypers['seed'])
    except KeyError:
        print('!! Function get_gnn(): Seed not specified in gnn_hypers object. Defaulting to 0 !!')
        set_seed(0)

    model = gnn_hypers['model']
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']
    agg_type = gnn_hypers['layer_agg_type'] or 'mean'

    # instantiate the GNN
    print(f'Building {model} model...')
    if model == "GraphConv":
        net = GNNConv(g, dim_embedding, hidden_dim, number_classes, dropout)

    else:
        raise ValueError("Invalid model type input! Model type has to be in one of these two options: ['GraphConv', 'GraphSAGE']")

    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())

    print('Building ADAM-W optimizer...')
    optimizer = torch.optim.AdamW(params, **opt_params, weight_decay=1e-2)

    return net, embed, optimizer


def loss_func_mod(probs, adj_tensor):

    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2

    return loss_


def loss_func_color_hard(coloring, nx_graph):


    cost_ = 0
    for (u, v) in nx_graph.edges:
        cost_ += 1*(coloring[u] == coloring[v])*(u != v)

    return cost_

