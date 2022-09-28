# Intro 

Describe how edge list csv can be bulk insert to redisgraph. 

Basically, we follow the usage of redisgraph-bulk-loader: https://github.com/RedisGraph/redisgraph-bulk-loader.

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
## 2) Bulk Insert DEMO

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

## 3) Bulk Update DEMO

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
