## Objective
說明圖演算法中的GCN，GraphSage，以及GAT的程式碼

## PyTorch Geometric
- [repo: Implemented GNN Models @PyG](https://github.com/pyg-team/pytorch_geometric#implemented-gnn-models)

## Concept
- [nn](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch-geometric-nn)
    - Linear, Sequential, and Model layers
- [TORCH.NN.FUNCTIONAL.NLL_LOSS](https://pytorch.org/docs/1.9.0/generated/torch.nn.functional.nll_loss.html)
    - The negative log likelihood loss. It is useful to train a classification problem with C classes. Details in [NLLLoss](https://pytorch.org/docs/1.9.0/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss).
        - Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.

## Question
1. How to pick appropriate loss function?

## GCN
- [docs: GCNConv @ PyG](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py)
    - [repo: gcn.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py)
- [repo: tkipf/gcn (tensorflow)](https://github.com/tkipf/gcn)
- [repo: tkipf/pygcn (pytorch)](https://github.com/tkipf/pygcn)

## GraphSage
- [docs: SAGEConv @PyG ](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)
    - [repo(example1): reddit.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py)
    - [repo(example2): ogbn_products_sage.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)
    - [repo(example3): graph_sage_unsup.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py)

## GAT
- [docs: GATConv @PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)
    - [repo(example): gat.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py)

## Environment
- [Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
    ```bash== 
    $ python3 -m venv /Users/pro/Documents/env_torch
    ``` 
    ```bash== 
    $ source /Users/pro/Documents/env_torch/bin/activate
    ``` 


## Reference
- [GDC in PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html)
    - Processes the graph via Graph Diffusion Convolution (GDC) from the `"Diffusion Improves Graph Learning" <https://www.kdd.in.tum.de/gdc>` paper.
- [終於有人總結了圖神經網絡！](https://www.readfog.com/a/1639181535368286208)
- [Python Module: typing](https://myapollo.com.tw/zh-tw/python-typing-module/)
    - 該模組並非用來規定 Python 程式必須使用什麼型別，而是透過型別註釋(type annotations)讓開發者或協作者可以更加了解某個變數的型別，也讓第三方的工具能夠實作型別檢查器(type checker)。