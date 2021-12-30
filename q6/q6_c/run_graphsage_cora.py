# %%
import sys, os
print (os.getcwd())

# %%
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
print(torch.cuda.get_device_name(0))
print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')

# %%
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

from functools import wraps
import time

def elapsed_time(func):
    @wraps(func)
    def out(*args, **kwargs):
        init_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - init_time
        print(f'Elapsed time: {elapsed_time:6.2f} seconds for {func.__name__}')
    return out

EPS = 1e-15

dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

print(path)
# %%
class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, number_workers=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels, number_workers))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

def train(model, optimizer, x, train_loader):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / data.num_nodes

@torch.no_grad()
def test(model, optimizer, x, edge_index):
    model.eval()
    out = model.full_forward(x, edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_acc, test_acc

# %%
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SAGE(data.num_node_features, hidden_channels=64, num_layers=8, number_workers=2).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# x, edge_index = data.x.to(device), data.edge_index.to(device)

# @elapsed_time
# def run_epoch(num_of_epoch=10):
#     for epoch in range(1, num_of_epoch+1):
#         loss = train(model)
#         val_acc, test_acc = test(model)
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
#             f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# run_epoch(num_of_epoch=2)
# run_epoch(num_of_epoch=10)

# %%
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# %%
def test_number_of_workers(num_of_epoch=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    for i in [1, 2, 4, 8]:
        init_time = time.time()
        model = SAGE(data.num_node_features, hidden_channels=64, num_layers=8, number_workers=i).to(device)
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x, edge_index = data.x.to(device), data.edge_index.to(device)

        train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256, shuffle=True, num_nodes=data.num_nodes, num_workers=0)
        
        for epoch in range(1, num_of_epoch+1):
            loss = train(model, optimizer, x, train_loader)
            val_acc, test_acc = test(model, optimizer, x, edge_index)
            if epoch == num_of_epoch:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                    f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        # feed_in_model(model, num_of_epoch)
        elapsed_time = time.time() - init_time
        print(f'Number of workers: {i}. Elapsed time: {elapsed_time:6.2f} seconds.')
        del model 


if __name__ == '__main__':
    test_number_of_workers(num_of_epoch=10)
    print("Done!")
# %%
