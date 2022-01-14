- 自由定義題目

### 題目 1: pytorch改寫成pytorch lightning


### 題目 2: PyG_AutoScale Paper 研讀與example導入研發雲 

Paper: arxiv.org/abs/2106.05609

Github Repo: https://github.com/rusty1s/pyg_autoscale

安裝方法: 

1. 使用研發雲Image: 

選擇有GPU的instance

image: Esun Notebook - Opiton GPU Cuda 10 (GPU) ->內建pytorch 1.9.0 cuda 10.2版本

2. 安裝 pytorch geometric 

因pytorch_geometric的GPU版本需於pip install時提供url，但由於aicloud無法連外網，因此須將pytorch_geometric的相關whl檔於外網進行下載後移進aicloud進行安裝:

- 下載pytorch_geometric dependencies的whl檔
因image中現有的pytorch為1.9.0 cuda 10.2版本，因此需致以下連結尋找pytorch_geometric的whl檔進行下載:https://pytorch-geometric.com/whl/torch-1.9.0%2Bcu102.html

- 需下載以下whls:
```
torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
torch_sparse-0.6.10-cp37-cp37m-linux_x86_64.whl
torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
```
放到aicloud
透過C170_檔案上傳區把檔案放到桌面雲，再從桌面雲上傳到aicloud。



(目前這些whl檔也可以到 exchanging-pool/to奕勳/PYG 裡面找，但不保證永遠存在lol) 

- 安裝指令:
```
pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl

pip install torch_sparse-0.6.10-cp37-cp37m-linux_x86_64.whl

pip install torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl

pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl

pip install torch-geometric
```

3. 安裝pyg_autoscale套件: 

pip install git+https://github.com/rusty1s/pyg_autoscale.git


4. 測試小資料examples:
```
git clone https://github.com/rusty1s/pyg_autoscale.git
cd pyg_autoscale/examples
```
Train GCN on Cora:
```
python train_gcn.py --root=/tmp/datasets --device=0
```
Train GCN2 on Cora:
```
python train_gcn2.py --root=/tmp/datasets --device=0
```
Train GIN on Cluster (目前資料源因防火牆尚無法下載，需手動移入資料) 
```
python train_gin.py --root=/tmp/datasets --device=0
```

5. 測試小資料benchmark
```
cd pyg_autoscale/small_benchmark
python main.py model=gcn dataset=cora root=/tmp/datasets device=0
```

You can choose between the following models and datasets:

Models: gcn, gat, appnp, gcn2
Datasets: cora, citesser, pubmed, coauthor-cs, coauthor-physics, amazon-computers, amazon-photo, wikics

### 題目2未來展望: 

- 確認large scale benchmark可以在研發雲跑，目前large scale 資料源因防火牆無法自動下載，需手動移入研發雲進行測試
- 將此框架整合到edu_framework，變成一個可以長出各種Scalable版本GNN的框架 (不過此框架有很多cpu/gpu之間溝通的客製化，會與pytorch lightning底層有衝突，要思考如何在盡量不改動edu_framework框架的情況下，解決這些衝突) 





