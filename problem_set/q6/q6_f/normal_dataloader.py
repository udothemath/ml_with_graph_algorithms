#%%
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from time import time
from torch.utils.data import DataLoader

EPS = 1e-15

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath('__file__')), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
#%%
data = torch.cat([data.x, data.y.unsqueeze(-1)], dim=-1)

#%%
class Model(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.layer1 = nn.Linear(input_shape, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32, output_shape)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(1433, 7).to(device)

    
#%%
def main():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(1, 3):
        train_loader = DataLoader(data, batch_size=256, num_workers=i, persistent_workers=True)
        start_time = time()
        for e in range(1, 51):
            model.train()
            t_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch[:, :-1].to(device))
                cri = nn.CrossEntropyLoss()
                loss = cri(out, batch[:, -1].long().to(device))
                t_loss += loss.item()
                loss.backward()
                optimizer.step()
        print('num_workers:' , i, 'total time consumed:', time() - start_time)
#%%
if __name__ == '__main__':
    main()


# %%
