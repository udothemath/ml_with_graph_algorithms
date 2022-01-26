## Note 
此README共有以下六大類。新增參考資料時，請一同附上連結內容的簡易說明。

- Discussion in General
- Explainability
- Temporal
- Survey of Scalable Graph Neural Networks
- Application Related
- Programming Framework

### Discussion in General
Q: Can GCN predict the unseen nodes?
Quick answer: It is possible, but not clear how to do it yet.
  - [repo: can gcn predict the unseen nodes?](https://github.com/tkipf/gcn/issues/103)
  - [post: Recent Advances in Graph Convolutional Network (GCN)](https://towardsdatascience.com/recent-advances-in-graph-convolutional-network-gcn-9166b27969e5)
  - [post: Graph Neural Networks](https://snap-stanford.github.io/cs224w-notes/machine-learning-with-networks/graph-neural-networks)

Q: Semi-supervised Learning 
  - [post: Training Better Deep Learning Models for Structured Data using Semi-supervised Learning](https://towardsdatascience.com/training-better-deep-learning-models-for-structured-data-using-semi-supervised-learning-8acc3b536319)

Q: How to set up dataloader for GraphSage algorithm?
  - [post: GraphSAGE for Classification in Python](https://antonsruberts.github.io/graph/graphsage/)
  - [post: 图神经网络(GNN)入门之旅(五)-GraphSAGE源码解析](https://zhuanlan.zhihu.com/p/354831060)
  - [github issue: Is the size always the same between number of layers in SAGE and len(size) in NeighborSampler?](https://github.com/pyg-team/pytorch_geometric/discussions/3799#discussioncomment-1913294)

Q: do we need deep GNN?
  - Blog: [Do we need "deep" graph neural networks?](https://towardsdatascience.com/do-we-need-deep-graph-neural-networks-be62d3ec5c59#dd10-e43ce4231f64)
  - Blog: [Over-smoothing issue in graph neural network](https://towardsdatascience.com/over-smoothing-issue-in-graph-neural-network-bddc8fbc2472) 
  - Note
    - GNN的核心概念是什麼?
      - (保留節點資訊以及(圖)結構資訊)。GNN  is a model that can build upon the information given in both: the nodes’ features and the local structures in our graph
    - GNN的實際作法?
      - (Embedding匯集節點本身以及鄰居資訊)。... to construct these embeddings (for each node) integrating both the nodes' initial feature vector and the information about the local graph structure that surrounds them.
    - Over-smoothing造成什麼問題?
      - (資訊被過度弱化)。This means nodes will have access to information from nodes that are far and may not be similar to them. On one hand, the message passing formalism tries to soften out the distance between neighbors nodes (smoothing ) to ease our classification later.  On the other hand, it can work in the other direction by making all our nodes embedding similar thus we will not be able to classify unlabeled nodes (over-smoothing ).
    - Over-smoothing 怎麼發生的? 
      - (匯集aggregate以及更新update時發生的)The message passing framework uses the two main functions introduced earlier Aggregate and Update, which gather feature vectors from neighbors and combine them with the nodes’ own features to update their representation. This operation works in a way that makes interacting nodes (within this process) have quite similar representations.
    - 不要太多學習層就可以解決over-smoothing嗎?
      - (看情況) One may think that reducing the number of layers will reduce the effect of over-smoothing. Yes, but this implies not exploiting the multi-hop information in the case of complex-structured data, and consequently not boosting our end-task performance.
    - 如何量化over-smoothing的程度?
      - 計算額外的學習層可以學到更多資訊還是雜訊。MAD and MADGap
      - 計算各群節點的距離比例。Group Distance Ratio
    - 資源需求更少的量化作法?
      - 新增一個模型層. This layer uses in one hand a trainable assignment matrix, thus it has feedback from our loss function, so it's guided to assign nodes in the perfect case to their true classes. On the other hand, we have also the shifting and scaling parameters which are also guided by our loss function.
    - 結論(及討論)
      - 其他圖資訊更新作法。all of these issues can be linked to the main mechanisms that we use to train our graph models which is Message-passing
  - Reference
    - Blog: [Do we need deep graph neural networks? by Michael Bronstein](https://towardsdatascience.com/do-we-need-deep-graph-neural-networks-be62d3ec5c59)

### Explainability
- [How to Explain Graph Neural Network — GNNExplainer](https://towardsdatascience.com/how-can-we-explain-graph-neural-network-5031ea127004)
  - https://github.com/dmlc/dgl/tree/master/examples/pytorch/gnn_explainer
- [Towards Explainable Graph Neural Networks](https://link.medium.com/qTCP69rXOgb)
- GAT
- TABNET
- https://towardsdatascience.com/explainable-graph-neural-networks-cb009c2bc8ea

### Temporal: 
- [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf)

### Survey of Scalable Graph Neural Networks
- Articles:
  - Sampling Large Graphs in Pytorch Geometric: https://towardsdatascience.com/sampling-large-graphs-in-pytorch-geometric-97a6119c41f9
  - Simple Scalable GNN: https://towardsdatascience.com/simple-scalable-graph-neural-networks-7eb04f366d07
  - Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour: https://arxiv.org/abs/1706.02677
- History:
  - 2017 [GraphSage](https://arxiv.org/abs/1706.02216) (Node Sampling Approach)
    - 摘要
      - 名稱由來： GraphSAGE (SAmple and aggreGatE)
      - 示意圖：
      - 應用場景：節點分類，分群，連結預測
        - These node embeddings can then be fed to downstream machine learning systems and aid in tasks such as node classification, clustering, and link prediction
      - 貢獻：對於沒有看過的節點，不需要重新執行整個訓練流程
        - ..., previous works have focused on embedding nodes from a single fixed graph, and many real-world applications require embeddings to be quickly generated for unseen nodes, or entirely new (sub)graphs
      - 關鍵字：transductive, inductive
      - 概念：訓練後得出可以產生節點embedding的函式，而不僅只是產生每個節點的embedding
        - Instead of training a distinct embedding vector for each node, we train a set of aggregator functions that learn to aggregate feature information from a node’s local neighborhood (Figure 1)
      - 理論：GraphSage使用節點的特徵，即可學習到圖的結構特徵
        - GraphSAGE can learn about graph structure, even though it is inherently based on features.
      - 精進方向：有向圖(考慮關聯的方向性)的預測
        - extending GraphSAGE to incorporate directed or multi-modal graphs. A particularly interesting direction for future work is exploring non-uniform neighborhood sampling functions, and perhaps even learning these functions as part of the GraphSAGE optimization.
      - 參考資料
        - [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
        - [GraphSAGE on neo4j](https://neo4j.com/docs/graph-data-science/current/algorithms/graph-sage/)
        - [Heterogeneous GraphSAGE (HinSAGE)](https://stellargraph.readthedocs.io/en/stable/hinsage.html)
        - [Graph Representation Learning](https://paperswithcode.com/task/graph-representation-learning)
        - [PyTorch geometric](https://github.com/rusty1s/pytorch_geometric)
      - 待辦事項
        - [實作SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv)
    - [課程介紹@Stanford](http://snap.stanford.edu/graphsage/)
    - 補充：
      - https://www.researchgate.net/post/What-is-the-model-architectural-difference-between-transductive-GCN-and-inductive-GraphSAGE
        - The main novelty of GraphSAGE is a neighborhood sampling step (but this is independent of whether these models are used inductively or transductively). You can think of GraphSAGE as GCN with subsampled neighbors.
        - In practice, both can be used inductively and transductively.
        - The title of the GraphSAGE paper ("Inductive representation learning") is unfortunately a bit misleading in that regard. The main benefit of the sampling step of GraphSAGE is scalability (but at the cost of higher variance gradients).
      - https://link.medium.com/VKLnyI0onhb
        - GraphSAGE is capable of predicting embedding of a new node, without requiring a re-training procedure. To do so, GraphSAGE learns aggregator functions that can induce the embedding of a new node given its features and neighborhood. This is called inductive learning.
    - Notes
      - It used neighbourhood sampling combined with mini-batch training to train GNNs on large graphs.
      - To compute the training loss on a single node with an L-layer GCN, only the L-hop neighbours of that node are necessary, as nodes further away in the graph are not involved in the computation.
      - For graphs of the “small-world” type, such as social networks, the 2-hop neighbourhood of some nodes may already contain millions of nodes, making it too big to be stored in memory [9].
        - GraphSAGE tackles this problem by sampling the neighbours up to the L-th hop: starting from the training node, it samples uniformly with replacement [10] a fixed number k of 1-hop neighbours, then for each of these neighbours it again samples k neighbours, and so on for L times.
      - A notable drawback of GraphSAGE is that sampled nodes might appear multiple times, thus potentially introducing a lot of redundant computation.
  - [2018 Graph Attention Network(GATs)](https://arxiv.org/pdf/1710.10903.pdf)
    - Quick Brief Intro of Graph
      - A graph G consists of two types of elements: vertices and edges (Source)
      - Source from Stanford
    - Petar et al. ICLR 2018
    - Abstract
      - ... implicitly specifying different weights to different nodes in a neighborhood
      - ... readily applicable to inductive as well as transductive problem
    - Summary of GATs
      - The research described in the paper Graph Convolutional Network (GCN), indicates that combining local graph structure and node-level features yields good performance on node classification tasks. However, the way GCN aggregates is structure-dependent, which can hurt its generalizability. One workaround is to simply average over all neighbor node features as described in the research paper GraphSAGE. However, Graph Attention Network proposes a different type of aggregation. GAT uses weighting neighbor features with feature dependent and structure-free normalization, in the style of attention.
    - Key Concept
      - Attention mechanism
      - Equation
      - Four Steps
      - Architecture
      - Properties
    - Conclusion
      - ... (implicitly) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront
      - A t-SNE plot of the computed feature representations of a pre-trained GAT model’s
    - Implementation
      - PyTorch
        - PyTorch_geometric
        - Algorithm
      - DGL
        - Equation
      - Math derivation
        - Blog: Graph Attention Networks Under the Hood
    - Reference
      - Visualization: https://towardsdatascience.com/why-you-should-not-rely-on-t-sne-umap-or-trimap-f8f5dc333e59
      - Website: Petar's website
      - Blog: Do we need deep graph neural networks? by Michael Bronstein
      - YouTube: Deep learning on graphs: successes, challenges, and next steps | Graph Neural Networks
      - Tutorial: Graph attention networks by DGL
      - YouTube: Transformer by Hung-yi Lee
      - Blog: Graph Attention Networks Under the Hood
      - Blog: The Illustrated Transformer
  - 2019: Cluster-GCN: https://arxiv.org/abs/1905.07953 (Graph Sampling Approach) - Improving Efficiency of Mini-Batching
    - First clustering the graph. Then, at each batch, the model is trained on one cluster. This allows the nodes in each batch to be as tightly connected as possible.
    - Make sure that these subgraphs preserve most of the original edges and still present a meaningful topological structure.
  - 2019: GraphSAINT: https://arxiv.org/abs/1907.04931 (Propose a general probabilistic graph sampler)
    - Strategies:
      - Uniform node sampling
      - Uniform edge sampling
      - “importance sampling” by using random walks to compute the importance of nodes and use it as the probability distribution for sampling
    - Note:
      - During training, sampling acts as a sort of edge-wise dropout, which regularises the model and can help the performance.
      - Graph Sampling reduce the bottleneck and the "over-squashing phenominon": https://arxiv.org/pdf/2006.05205.pdf (Because we use a smaller sub-graph)
  - 2020: SIGN: simple, sampling-free architectures
    - Decompose graph into 1-hop neighbor graph, 1-hop neighbor graph, ..., and apply a single-layer GNN on each of the graph to avoid multi-hop unscalability of multi-layer GNN.
- In the field of recommendation [Graph Neural Networks in Recommender Systems: A Survey]: #Next-Up
  - Sampling large graph:
    - 2017: GraphSage [28]
    - 2018: PinSage  -  Graph Convolutional Neural Networks for Web-Scale Recommender Systems [145]
  - Reconstruct small-scale subgraph (mostly for knowledge graph augmented graph):
    - 2019: Attentive Knowledge Graph Embedding for Personalized Recommendation [91]
    - 2020: ATBRG: Adaptive Target-Behavior Relational Graph Network for Effective Recommendation [18]
  - Decouple the operations of non-linearities and collapsing weight matrices between consecutive layers.
    - 2019: Simplifying graph convolutional networks [129]
    - 2020: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation [31]
    - 2020: SIGN: Scalable Inception Graph Neural Networks. [20]
### Application Related
- 推薦系統:  #RecSys
  - Graph Neural Networks in Recommender Systems: A Survey #Next-Up
    - Evernote Link : https://www.evernote.com/shard/s89/nl/19207065/9546232e-dac3-4e82-9547-03bcece15d84
    - Dynamic Graph Neural Networks for Sequential Recommendation
  - Cross-Domain Recommendation (CDR):
    - Must Read:
    - 越屬於以下特性的paper越需要研讀: Multi-Target / User-level relevance (certain degree of shared users)
      - (2021) Cross-Domain Recommendation: Challenges, Progress, and Prospects # Cross-Domain Recommendation (CDR) 的Survey，可以先從這篇出發了解CDR的各面向與學術發展
      - (2020) HeroGRAPH: A Heterogeneous Graph Framework for Multi-Target Cross-Domain Recommendation # 這篇是最符合我們未來跨售情境的Paper (跨產品線顧客僅部分重疊)，且是使用Graph來解決此問題，值得參考。
      - (2020) Graphical and Attentional Framework for Dual-Target Cross-Domain Recommendation # 雖然只考慮兩個產品線，但是也如同我們未來跨售情境，兩產品線顧客僅部分重疊，也值得參考。
      - (2019) Cross Domain Recommendation via Bi-directional Transfer Graph Collaborative Filtering Networks # 符合成大產學目標要求的Paper (跨產品線顧客完全重疊)，雖只考慮了兩個產品線，但是也值得給成大團隊參考。
        - Cross Domain Recommendation via Bi-directional Transfer Graph Collaborative Filtering Networks.pdf
    - Source: Evernote 論文整理 [Survey of Cross-Domain Recommendation]
  - From Spotify to Fund Recommendation https://www.evernote.com/shard/s89/sh/5e762cd5-51f5-cb48-03c3-e9d285e9187b/e63acfbb97ee6d154200b339dd149b44
- 金流相關: #Loan #RegTech 
  - Linking bank clients using graph neural networks powered by rich transactional data.pdf
- #RegTech (信用風險外的風險模型)
### Programming Frameworks:   
- Pytorch Geometric 
  - [Survey of Pytorch Geometric](https://www.evernote.com/shard/s89/nl/19207065/825fff45-5289-84b2-3af1-08a1376bdbd5)
  - [Survey of Pytorch Geometric Temporal](https://www.evernote.com/shard/s89/nl/19207065/93a91816-8705-ee18-5756-4d29fc22e169)
- Deep Graph Library 知識整理
  - [Deep Graph Library(DGL), message passing and financial relations modelling](https://link.medium.com/jgVSSRX2Pgb)
  - [Training DGL GraphSAGE with PyTorch Lightning](https://github.com/dmlc/dgl/pull/2878)
  - [DGL-RecSys](https://github.com/dmlc/dgl#dgl-for-domain-applications)
  - [Handling large graph: Exact Offline Inference on Large Graphs](https://docs.dgl.ai/guide/minibatch-inference.html#guide-minibatch-inference)
- [Sampling Large Graphs in PyTorch Geometric](https://towardsdatascience.com/sampling-large-graphs-in-pytorch-geometric-97a6119c41f9)
- Connecting Graph ML with Graph DB
  - [Neo4j & DGL — a seamless integration](https://link.medium.com/YL4McKj1Rgb)
  - Amazon have done it: https://link.medium.com/3JHAtVXVRgb
- ETL Visualization Tool
  - https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/DesignNewPreprocessModule.ipynb

