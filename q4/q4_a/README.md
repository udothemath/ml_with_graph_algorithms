## 目標
說明圖演算法中的GCN，GraphSage，以及GAT的程式碼

## PyTorch Geometric
- [repo: Implemented GNN Models @PyG](https://github.com/pyg-team/pytorch_geometric#implemented-gnn-models)
 
## GCN
- [repo: tkipf/gcn (tensorflow)](https://github.com/tkipf/gcn)
- [repo: tkipf/pygcn (pytorch)](https://github.com/tkipf/pygcn)

## GraphSage
- [docs: SageConV @PyG ](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)
    - [repo(example1): reddit.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py)
    - [repo(example2): ogbn_products_sage.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)
    - [repo(example3): graph_sage_unsup.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py)

## GAT


## 參考資料
- [GDC in PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html)
    - Processes the graph via Graph Diffusion Convolution (GDC) from the `"Diffusion Improves Graph Learning" <https://www.kdd.in.tum.de/gdc>` paper.
- [終於有人總結了圖神經網絡！](https://www.readfog.com/a/1639181535368286208)