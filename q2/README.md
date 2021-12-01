## 任務
新增節點(Node)以及連線關係(Edge)的屬性

## 執行步驟
請依照執行環境(行內或是行外)選擇安裝流程
### Step 1 行外空間(aicloud @esun)
- install package
```bash
$ pip install -r requirements.txt
```
### Step 1 行內空間
- install package
```bash
$ pip install -r requirements_basic.txt
```
and then

```bash
$ sh requirements_pytorch_bash.sh
```
附註：由於pytorch套件有相容性的問題，請選擇有GPU的instance確保可以正確執行

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

## 模型說明
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

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
