## Objective
說明圖演算法中的GCN，GraphSage，以及GAT的程式碼

## Key Component
- [docs: nn functional](https://pytorch.org/docs/stable/nn.functional.html)
- [repo: Implemented GNN Models @PyG](https://github.com/pyg-team/pytorch_geometric#implemented-gnn-models)
- [docs: nn in torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch-geometric-nn)
    - Linear, Sequential, and Model layers
- [docs: TORCH.NN.FUNCTIONAL.NLL_LOSS](https://pytorch.org/docs/1.9.0/generated/torch.nn.functional.nll_loss.html)
    - The negative log likelihood loss. It is useful to train a classification problem with C classes. Details in [NLLLoss](https://pytorch.org/docs/1.9.0/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss).
        - Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
- Activation function
    - Relu, elu, selu, celu, ... search at [TORCH.NN.FUNCTIONAL](https://pytorch.org/docs/stable/nn.functional.html)
- [Deep learning basics — weight decay](https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9)
    - Why do we use weight decay?
        - To prevent overfitting
        - To keep the weights small and avoid exploding gradient
- [Loss func for classification problem](https://www.yanxishe.com/columnDetail/16855)
- [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)
    - This function doesn’t work directly with NLLLoss, which expects the Log to be computed between the Softmax and itself. Use log_softmax instead (it’s faster and has better numerical properties).

## Question
1. How to chhoose appropriate loss function?
    - [post: Deep Learning: Which Loss and Activation Functions should I use?](https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8)
2. How does message passing function work within graph based layers?
3. How to set up batch computation using multi-cores?
4. How to modify model architecture to improve performance?
5. How to apply GAT on inductive problem?

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
- [paper: GRAPH ATTENTION NETWORKS(ICLR 2018)](https://arxiv.org/pdf/1710.10903.pdf)
- [post: Graph Attention Network](https://petar-v.com/GAT/)
    - [repo: GAT using TensorFlow](https://github.com/PetarV-/GAT/blob/master/execute_cora.py)
- [post: 探討graph attention機制有效性 — Understanding Attention and Generalization in Graph Neural Networks(NIPS 2019)](https://medium.com/@qaz7821819/%E6%8E%A2%E8%A8%8Egraph-attention%E6%A9%9F%E5%88%B6%E6%9C%89%E6%95%88%E6%80%A7-understanding-attention-and-generalization-in-graph-neural-networks-d1ad54084bc0)
- [paper: ADAPTIVE STRUCTURAL FINGERPRINTS FOR GRAPH
ATTENTION NETWORKS(2020)](https://openreview.net/pdf?id=BJxWx0NYPr)


## Install package (with virtual environment)
- [Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
    ```bash== 
    $ python3 -m venv /Users/pro/Documents/env_torch
    ``` 
    ```bash== 
    $ source /Users/pro/Documents/env_torch/bin/activate
    ``` 
- Install torch
    ```bash== 
    $ pip install torch==1.10.0 -f https://download.pytorch.org/whl/torch_stable.html 
    ```
- Install torch relevant package
    ```bash== 
    $ sh requirements_torch.sh
    ```


## Reference
- [GDC in PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html)
    - Processes the graph via Graph Diffusion Convolution (GDC) from the `"Diffusion Improves Graph Learning" <https://www.kdd.in.tum.de/gdc>` paper.
- [終於有人總結了圖神經網絡！](https://www.readfog.com/a/1639181535368286208)
- [Python Module: typing](https://myapollo.com.tw/zh-tw/python-typing-module/)
    - 該模組並非用來規定 Python 程式必須使用什麼型別，而是透過型別註釋(type annotations)讓開發者或協作者可以更加了解某個變數的型別，也讓第三方的工具能夠實作型別檢查器(type checker)。
- [parse_args()報錯:ipykernel_launcher.py: error: unrecognized arguments](http://corettainformation.blogspot.com/2021/02/parseargsipykernellauncherpy-error.html)