# %%
import sys, os
print (os.getcwd())

DIR_DATA = '/Users/pro/Documents/z_data'
DIR_DATA_OGB = f'{DIR_DATA}/dataset/'

# %%
from torch_geometric.datasets import TUDataset

root = f'{DIR_DATA}/enzymes'
name = 'ENZYMES'

# The ENZYMES dataset
pyg_dataset= TUDataset(root, 'ENZYMES')

# You can find that there are 600 graphs in this dataset
print(pyg_dataset)

# %%
def get_num_classes(pyg_dataset):
    # TODO: Implement this function that takes a PyG dataset object
    # and return the number of classes for that dataset.

    num_classes = 0

    ############# Your code here ############
    ## (~1 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful.
    num_classes = pyg_dataset.num_classes

    #########################################

    return num_classes


def get_num_features(pyg_dataset):
    # TODO: Implement this function that takes a PyG dataset object
    # and return the number of features for that dataset.

    num_features = 0

    ############# Your code here ############
    ## (~1 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful.
    num_features = pyg_dataset.num_features 
    #########################################

    return num_features

# You may find that some information need to be stored in the dataset level,
# specifically if there are multiple graphs in the dataset

num_classes = get_num_classes(pyg_dataset)
num_features = get_num_features(pyg_dataset)
print("{} dataset has {} classes".format(name, num_classes))
print("{} dataset has {} features".format(name, num_features))
# %%
def get_graph_class(pyg_dataset, idx):
    # TODO: Implement this function that takes a PyG dataset object,
    # the index of the graph in dataset, and returns the class/label 
    # of the graph (in integer).

    label = -1

    ############# Your code here ############
    ## (~1 line of code)
    label = pyg_dataset[idx].y
    #########################################

    return label

# Here pyg_dataset is a dataset for graph classification
graph_0 = pyg_dataset[0]
print(graph_0)

idx = 100
label = get_graph_class(pyg_dataset, idx)
print('Graph with index {} has label {}'.format(idx, label))

# %%
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

dataset_name = 'ogbn-arxiv'
# Load the dataset and transform it to sparse tensor
dataset = PygNodePropPredDataset(name=dataset_name,
                                 root=DIR_DATA_OGB,
                                 transform=T.ToSparseTensor())
print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

# Extract the graph
data = dataset[0]
print(data)
# %%
def graph_num_features(data):
    # TODO: Implement this function that takes a PyG data object,
    # and returns the number of features in the graph (in integer).

    num_features = 0

    ############# Your code here ############
    ## (~1 line of code)
    num_features = data.num_features
    #########################################

    return num_features

num_features = graph_num_features(data)
print('The graph has {} features'.format(num_features))
# %%
import torch
import torch.nn.functional as F
print(torch.__version__)

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# %%
dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name,
                                 root=DIR_DATA_OGB,
                                 transform=T.ToSparseTensor())
data = dataset[0]

# %%
# print (data.adj_t)

# %%

# Make the adjacency matrix to symmetric
data.adj_t = data.adj_t.to_symmetric()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If you use GPU, the device should be cuda
print('Device: {}'.format(device))

data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
# %%
