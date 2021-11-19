## 任務
將資料轉換成圖演算法可以使用的格式

## 執行
### Step1: 
- install package
```bash
$ pip install -r requirements.txt
```
### Step 2
- run pyg 
    ```bash==
    $ python -u run.py  
    ``` 
- (Optional) run igraph
    ```bash== 
    $ python -u run_igraph.py
    ```
### Note: [python -u explanation](https://stackoverflow.com/questions/14258500/python-significance-of-u-option)

## 套件需求
```bash
$ pip install -r requirements.txt
```
or
```bash==
$ pip install numpy==1.21.4
$ pip install pandas==1.3.4
$ pip install torch==1.10.0
$ pip install torch-geometric==2.0.2
$ pip install torch-scatter==2.0.9
$ pip install torch-sparse==0.6.12
$ pip install sentence-transformers==2.1.0
$ pip install igraph==0.9.8
```
## 工具
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [igraph安裝介紹](https://igraph.org/python/)

## 資料集
- [CiteSeer](https://networkrepository.com/citeseer.php)
- [MovieLens](https://grouplens.org/datasets/movielens/)

## 參考資料
- [LOADING GRAPHS FROM CSV](https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html)
- [Visualising Graph Data with Python-igraph](https://towardsdatascience.com/visualising-graph-data-with-python-igraph-b3cc81a495cf)
- Visualize a graph online: [GraphVis](https://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/citeseer.edges)
