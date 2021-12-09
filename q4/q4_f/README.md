## Pytorch建模基礎架構
- [以mnist為例](https://blog.csdn.net/qq_40211493/article/details/106580655)

## PyTorch Geometric Data 格式
- [torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)

## NeighborLoader與NeighborSampler
- [torch_geometric.loader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html)
- NeighborLoader可泛化在異質圖上，但NeighborSampler只能用在同質圖上。

## Unsupervised Graph Representation Learning
- 利用positive sample和negative sample，希望真實有連的點之間距離越近越好，沒有連的點之間距離越遠越好。
- 實務上會找到這些samples後，利用GNN找到他們的embedding，再去計算之間的距離當作loss

## GraphSage
- [docs: SAGEConv @PyG ](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)
    - [repo(example1): reddit.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py)
    - [repo(example2): ogbn_products_sage.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)
    - [repo(example3): graph_sage_unsup.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py)
