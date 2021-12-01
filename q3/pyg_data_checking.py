# %%
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

# %%
EPS = 1e-15

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath('__file__')), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
print('length of graph dataset:', len(dataset),'graph')
data = dataset[0]


print(data.edge_index)
print(data.edge_stores)
print('shape of x:', data.edge_stores[0]['x'].shape)
print('num of features:', data.num_features)
# %% Attributes Defines in 'data' 

"""
1. batch
 
2. edge_attr

3. edge_index

4. edge_stores

5. node_stores

6. num_edge_features
      Returns the number of features per edge in the graph.

7. num_faces
      Returns the number of faces in the mesh.

8. num_features
        Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`.

9. num_node_features
      Returns the number of features per node in the graph.
 
10. pos

11. stores

12. x

13. y
"""
# %%
print(data.batch)
# %% 
print(data.edge_attr)
# %%
print(data.edge_index)
# %%
print(data.edge_stores)

# %%
print(data.edge_stores[0]['x'] == data.x)

# %%
print(data.edge_stores[0]['edge_index'])
print('edge count:', data.edge_index.shape[1])
# %%
print(data.node_stores)
print(data.edge_stores == data.node_stores)
# %%
print(data.num_faces)


# %%
print(data.num_features)

print(data.num_features == data.num_node_features)

# %%
print(data.pos)

# %% 
print(data.stores == data.edge_stores)

# %%
print(data.y)
print(data.y.shape, data.x.shape)


