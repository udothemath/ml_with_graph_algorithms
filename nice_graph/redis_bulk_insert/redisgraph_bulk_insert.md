# Intro 

Describe how edge list csv can be bulk insert to redisgraph. 

Basically, we follow the usage of redisgraph-bulk-loader: https://github.com/RedisGraph/redisgraph-bulk-loader.

Two approaches: 

1. Approach I - Bulk Insert: CSV with Cypher-HEADER + One time insert for both all edges and nodes command.

> 如果有空值，並且希望redisgraph吃進特定型態的property，要用Approch I。對於空值的欄位該Node會直接忽略該欄位作為node的property。

2. Approach II - Bulk Update: CSV without Cypher-HEADER + Node-by-Node-Edge-by-Edge insert with Cypher command.
     
> 如果有空值，Approach I 在塞值的時候可能會出錯，要解這個問題可能要搭配特殊的cypher語法來解決。

# Installation

pip install redisgraph-bulk-loader

# 範例csv

Nodes資料表: Person.csv / Country.csv
Edges資料表: KNOWS.csv / VISITS.csv


# 操作流程demo 

## 1) 啟動redis-stack-server :
```bash
redis-stack-server
```
>>
```output
                _._                                                  
           _.-``__ ''-._                                             
      _.-``    `.  `_.  ''-._           Redis 7.0.0 (00000000/0) 64 bit
  .-`` .-```.  ```\/    _.,_ ''-._                                  
 (    '      ,       .-`  | `,    )     Running in standalone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 6379
 |    `-._   `._    /     _.-'    |     PID: 1920
  `-._    `-._  `-./  _.-'    _.-'                                   
 |`-._`-._    `-.__.-'    _.-'_.-'|                                  
 |    `-._`-._        _.-'_.-'    |           https://redis.io       
  `-._    `-._`-.__.-'_.-'    _.-'                                   
 |`-._`-._    `-.__.-'    _.-'_.-'|                                  
 |    `-._`-._        _.-'_.-'    |                                  
  `-._    `-._`-.__.-'_.-'    _.-'                                   
      `-._    `-.__.-'    _.-'                                       
          `-._        _.-'                                           
              `-.__.-'                                               

```
## 2) Approach 1: Bulk Insert 

目標: 創建由USERS和FOLLOWS組成的SOCIAL network

## 2.1) 準備nodes, edges資料

Users.csv
```csv
:ID(User), name:STRING, rank:INT
0, "Jeffrey", 5
1, "Filipe", 8
```
FOLLOWS.csv
```csv
:START_ID(User), :END_ID(User), reaction_count:INT
0, 1, 25
1, 0, 10
```
## 2.2) 把nodes,edges塞進redis
```bash
redisgraph-bulk-insert SocialGraph --enforce-schema --nodes User.csv --relations FOLLOWS.csv
```
>>
```
User  [####################################]  100%
2 nodes created with label 'User'
FOLLOWS  [####################################]  100%
2 relations created for type 'FOLLOWS'
Construction of graph 'SocialGraph' complete: 2 nodes created, 2 relations created in 0.010385 seconds
```

NOTE: :ID(User)這個欄位若是給string，會被自動加上編號!!該編號是一個以0,1,2..往下的編號

## 3) Approach 2: Bulk Update 

### 3.1) Update nodes using cypher 
```bash
redisgraph-bulk-update SocialGraph --csv User.csv --query "MERGE (:User {id: row[0], name: row[1], rank: row[2]})"
```
>>
```
SocialGraph  [####################################]  100%
Labels added: 1.0
Nodes created: 2.0
Properties set: 6.0
Cached execution: 1.0
internal execution time: 0.561191
Update of graph 'SocialGraph' complete in 0.008437 seconds
```
### 3.2) Update edges using cypher 
```bash
redisgraph-bulk-update SocialGraph --csv FOLLOWS.csv --query "MATCH (start {id: row[0]}), (end {id: row[1]}) MERGE (start)-[f:FOLLOWS]->(end) SET f.reaction_count = row[2]"
```
>>
```
redisgraph-bulk-update SocialGraph --csv FOLLOWS.csv --query "MATCH (start {id: row[0]}), (end {id: row[1]}) MERGE (start)-[f:FOLLOWS]->(end) SET f.reaction_count = row[2]"
SocialGraph  [####################################]  100%
Properties set: 2.0
Relationships created: 2.0
Cached execution: 1.0
internal execution time: 0.701094
Update of graph 'SocialGraph' complete in 0.008778 seconds
```

# 4) 自己來 

## 4.1) 資料介紹

Nodes -> Person.csv / Country.csv 

Edges -> KNOWS.csv / VISITS.csv

## 4.2) 用 cypher把點和edges加入進去
```bash
redisgraph-bulk-update TourGraph --csv Person.csv --query "MERGE (:Person {name: row[0], age: row[1], gender: row[2], status: row[3]})"
```
>>
```
TourGraph  [####################################]  100%
Labels added: 1.0
Nodes created: 14.0
Properties set: 56.0
Cached execution: 1.0
internal execution time: 1.477279
Update of graph 'TourGraph' complete in 0.009369 seconds
```
```bash
redisgraph-bulk-update TourGraph --csv Country.csv --query "MERGE (:Country {name: row[0]})"
```
>>
```
TourGraph  [####################################]  100%
Labels added: 1.0
Nodes created: 13.0
Properties set: 13.0
Cached execution: 1.0
internal execution time: 0.689067
Update of graph 'TourGraph' complete in 0.008437 seconds
```
```bash
redisgraph-bulk-update TourGraph --csv KNOWS.csv --query "MATCH (start:Person {name: row[0]}), (end:Person {name: row[1]}) MERGE (start)-[f:KNOWS]->(end) SET f.relation = row[2]"
```
>>
```
TourGraph  [####################################]  100%
Properties set: 13.0
Relationships created: 13.0
Cached execution: 1.0
internal execution time: 1.440969
Update of graph 'TourGraph' complete in 0.008845 seconds
```

```bash
redisgraph-bulk-update TourGraph --csv VISITS.csv --query "MATCH (start:Person {name: row[0]}), (end:Country {name: row[1]}) MERGE (start)-[f:VISITS]->(end) SET f.purpose = row[2]"
```
>>
```
TourGraph  [####################################]  100%
Properties set: 35.0
Relationships created: 35.0
Cached execution: 1.0
internal execution time: 2.791477
Update of graph 'TourGraph' complete in 0.012179 seconds
```

## Using redisinsight to query the graph:

### Query Nodes: 
```
GRAPH.QUERY TourGraph "MATCH (a) RETURN a"
```
### Query Nodes & Edges 1:
```
GRAPH.QUERY TourGraph "MATCH (a)-[b]-() RETURN a, b"
```
### Query Nodes & Edges 2:
```
GRAPH.QUERY TourGraph "MATCH p=()-[]-() RETURN p"
```

# 5) 存入AVM Graph

## 5.1) 將AVM的一百萬筆還沒有合併上opendata/poi的交易資料csv (41個欄位) 進行Header的改寫

```python
import pandas as pd
table = pd.read_parquet('df1_before_opendata_added.parquet')
mapping_of_pandas_type_to_cypher_type = {
    'str': 'STRING',
    'int64': 'INT',
    'float64': 'Float',
    'Timestamp': 'DateTime'
}

header_cols = []
for i, col in enumerate(table.columns):
    header_str = f'{col}:{mapping_of_pandas_type_to_cypher_type[type(table[col].iloc[0]).__name__]}'
    header_cols.append(header_str)
    print(col, '->', header_str)
header_cols[0] = ':ID(House)'
table.columns = header_cols
table.to_csv('avm_node_table.csv', index=False)
```

## 5.2) 用bulk insert指令把csv資料存進去
```bash
redisgraph-bulk-insert AVMGraph --enforce-schema --nodes /home/jovyan/if-graph-ml/esb21375/data/avm_node_table.csv
```
>>
```
avm_node_table  [####################################]  100%          
959656 nodes created with label 'avm_node_table'
Construction of graph 'AVMGraph' complete: 959656 nodes created, 0 relations created in 94.030649 seconds
```
- 會發現用掉了1.2Gb的RAM，轉成dump.rds是0.7Gb

# 6. 依據業務邏輯建立edges再刪掉

## 6.1 先啟動redis-cli:

> 127.0.0.1:6379> [] 

## 6.2 篩選點出來看

```
GRAPH.QUERY AVMGraph "MATCH (n) RETURN n LIMIT 10"
```

## 6.3 在距離相距為10內的點之間建立edges

只先看南門里的部分: 南門里 - '\xe5\x8c\x97\xe6\x96\x97\xe6\x9d\x91' (共228個點)
```
GRAPH.QUERY AVMGraph "MATCH (a) MATCH (b) WHERE a.VILLNAME = '\xe5\x8c\x97\xe6\x96\x97\xe6\x9d\x91' AND b.VILLNAME = '\xe5\x8c\x97\xe6\x96\x97\xe6\x9d\x91' AND id(a) <> id(b) AND (a.xx-b.xx)^2+(a.yy-b.yy)^2 < 100 CREATE (a)-[:IS_NEAR]->(b)"
```
>>
1) 1) "Relationships created: 608"
   2) "Cached execution: 0"
   3) "Query internal execution time: 50248.657546 milliseconds"
(50.25s)

## 6.4 查詢完整的edges
```
GRAPH.QUERY AVMGraph "MATCH (a)-[r]->(b) WHERE a.VILLNAME = '\xe5\x8c\x97\xe6\x96\x97\xe6\x9d\x91' RETURN a, r, b"
```

<img width="392" alt="擷取" src="https://user-images.githubusercontent.com/6816755/194280727-3788a3dd-793b-4a49-82ff-92ec33f32979.PNG">
