# q7 論文整理(GNNAutoScale: Scalable and Expressive Graph Neural Networks via Historical Embeddings)
### 重點摘要
- Present GNNAutoScale (GAS), a framework for scaling arbitrary message-passing GNNs to **large graphs**.
- For a given minibatch of nodes, GAS prunes the GNN computation graph so that only nodes inside the **current mini-batch** and their direct **1-hop neighbors** are retained, **independent of GNN depth**
- Historical embeddings act as an offline storage and are used to accurately fill in the inter-dependency information of out-of-mini-batch nodes
- 示意圖: 參考paper Figure1
- Achieve this by providing **approximation error bounds of historical embeddings** and show how to tighten them in practice.
- Show that the **practical realization** of our framework, PyGAS, an easy-to-use extension for PYTORCH GEOMETRIC, is both fast and memory-efficient.
- Github: https://github.com/rusty1s/pyg_autoscale

### Challenge of previous work 
- Difficult to scale them to large graphs (neighbor explosion)
- Cannot be applied to any GNN

### Model performance comparison
- 參考paper Figure3, Table1

### GPU memory consumption comparsion
- 參考paper Table3
