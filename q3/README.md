## 目標
建構完整圖神經網絡開發流程(暫時不需要關注成效)

## 版本
- q3_a: 測試GraphGym套件
- q3_c: 使用基金交易資料建立heterogeneous graph，並成功串接到Hetero Graph (參考資料: https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html) 
- q3_e: 使用Recsys Challenge 2015資料(歐洲電商網站)，以一次user visit中所造訪的item建一張graph(item為node, item間的順序為edge)，預測是否購買該item(node classification)。
- q3_f: PyG的github上複製下來的[範例](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py)，目標是在做node classification（某篇paper屬於哪種領域的論文）



## 參考資料
- [PyTorch - Neural networks with nn modules](https://jhui.github.io/2018/02/09/PyTorch-neural-networks/)
- [Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric] (https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8)
- [Recsys2015 data source] (https://www.kaggle.com/chadgostopp/recsys-challenge-2015?select=yoochoose-clicks.dat#)
