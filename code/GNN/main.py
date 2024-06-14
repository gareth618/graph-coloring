import random
import torch
import warnings
import numpy as np
import networkx as nx
import os
import dgl
from datetime import datetime
from time import time
import matplotlib as plt
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv
from itertools import chain
from tqdm import tqdm
from utils import get_gnn, loss_func_color_hard,build_graph_from_color_file,loss_func_mod

def main():
    SEED_VALUE = 0
    # random.seed(SEED_VALUE)
    # np.random.seed(SEED_VALUE)
    # torch.manual_seed(SEED_VALUE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    print(f'Will use device: {device}, torch dtype: {dtype}')
    chromatic_numbers = {
    'jean.col': 10,
    'anna.col': 11,
    'huck.col': 11,
    'david.col': 11,
    'homer.col': 13,
    'myciel5.col': 6,
    'myciel6.col': 7,
    'myciel7.col': 7,
    'inithx.i.1.col':54,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 8,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
    'queen13_13.col': 13,
    }
    problem_file = 'huck.col'
    input_parent = './instances'
    input_fpath = os.path.join(input_parent, problem_file)

    hyperparameters = {
        'model': 'GraphConv',
        'dim_embedding': 64,
        'dropout': 0.1,
        'learning_rate': 0.0001,
        'hidden_dim': 64,
        'seed': SEED_VALUE
    }
    # hyperparameters = {
    #         'model': 'GraphSAGE',
    #         'dim_embedding': 77,
    #         'dropout': 0.3784,
    #         'learning_rate': 0.02988,
    #         'hidden_dim': 32,
    #         'seed': SEED_VALUE
    #     }
    solver_hyperparameters = {
        'graph_file': problem_file,
        'layer_agg_type': 'mean',
        'number_classes': chromatic_numbers[problem_file]
    }
    hyperparameters.update(solver_hyperparameters)

    tolerance=1e-3
    epochs=int(4e5)
    patience=20000
    print(hyperparameters)
    nx_graph = build_graph_from_color_file(input_fpath, node_offset=-1, parent_fpath='')
    dgl_graph = dgl.from_networkx(nx_graph)
    dgl_graph = dgl_graph.to(device)
    opt_hyperparameters = {
    'lr': hyperparameters.get('learning_rate', None)
    }
    lower_bound_colors=1
    upper_bound_colors=nx_graph.number_of_nodes()
    print(nx_graph.number_of_edges())
    print(upper_bound_colors)
    best_no_colors=upper_bound_colors
    best_coloring_complete=None

    while lower_bound_colors<=upper_bound_colors:
        no_colors=lower_bound_colors + (upper_bound_colors-lower_bound_colors)//2
        print('Searching for number of colors:',no_colors)
        founded=False
        tries=1
        hyperparameters['number_classes']=no_colors

        adj_matrix = torch.tensor(nx.linalg.graphmatrix.adjacency_matrix(nx_graph).todense()).type(dtype).to(device)

        for tryy in range(tries):
            print('Try no:',tryy)
            hyperparameters['seed']=int(datetime.now().timestamp())
            net, embed, optimizer = get_gnn(dgl_graph, nx_graph.number_of_nodes(), hyperparameters, opt_hyperparameters, device, dtype)

            t_start = time()
            inputs = embed.weight
            best_cost = torch.tensor(float('Inf'))  # high initialization
            best_loss = torch.tensor(float('Inf'))
            best_coloring = None

            prev_loss = 1.
            cnt = 0
            for epoch in range(epochs):

                logits = net(inputs)
                probs = F.softmax(logits, dim=1)
                loss = loss_func_mod(probs, adj_matrix)
                coloring = torch.argmax(probs, dim=1)
                cost_hard = loss_func_color_hard(coloring, nx_graph)
                if cost_hard < best_cost:
                    best_loss = loss
                    best_cost = cost_hard
                    best_coloring = coloring

                if (abs(loss - prev_loss) <= tolerance) | ((loss - prev_loss) > 0):
                    cnt += 1
                else:
                    cnt = 0
                prev_loss = loss

                if cost_hard==0:
                    founded=True
                    best_no_colors=no_colors
                    best_coloring_complete=coloring
                    break

                if cnt >= patience:
                    print(f'Stopping early on epoch {epoch}. Patience count: {cnt}')
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if founded==True:
                print('Solution found in try',tryy)
                break
            print('Results: Failed')

            print('Epoch %d | Final loss: %.5f' % (epoch, loss.item()))
            print('Epoch %d | Lowest discrete cost: %.5f' % (epoch, best_cost))
            final_loss = loss
            final_coloring = torch.argmax(probs, 1)
            print(f'Final coloring: {final_coloring}, soft loss: {final_loss}')
            runtime_gnn = round(time() - t_start, 4)
            print(f'GNN runtime: {runtime_gnn}s')
        if founded==True:
            print('Solution found for',no_colors,'colors')
            upper_bound_colors=no_colors-1
        else:
            print('No solution found for',no_colors,'colors')
            lower_bound_colors=no_colors+1
    print('Solution:')
    print(best_no_colors)
    print(best_coloring_complete)

main()