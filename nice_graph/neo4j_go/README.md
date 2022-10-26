# go for neo4j
## off-line
1. Create virtual environment
	- ```$ python -m venv <name_of_venv_with_path> ```
2. Activate virtual environment
	- ```$ source <name_of_venv_with_path>/bin/activate ```
3. Install python package
	- ```$ pip install -r requirements_offline.txt ```
4. Initialize neo4j (graph database). Add $ to run at background.
	- ```$ neo4j start &```
4. Open browser with port forwarding
	- ```http://localhost:7474```
5. Execute by cypher code

## aicloud
###  Connect neo4j
0. Ensure that you pick the correct image (esun_graph)
1. Reset the neo4j setting
	- replace neo4j.conf file
	```bash 
	$ source reset_neo4j_config.sh
	```
	- Note: include the following setting in neo4j.conf
		- comment out import setting
			```bash=
			#dbms.directories.import=/var/lib/neo4j/import
			```
		- uncomment following line to import by desired
			```bash
			dbms.security.allow_csv_import_from_file_urls=true
			```
		- disable authentication (no need for user and password)
			```bash
			dbms.security.auth_enabled=false
			```
2. Connect neo4j
	- run neo4j console at background
	```bash
	$ neo4j console &
	```
3. There are three ways to communicate with neo4j
	- shell: 
		```bash 
		$ cypher-shell 
		```
	- python script:
		```bash
		$ python main.py
		```
	- through browser with port forward 
		```bash
		localhost:7474
		```
### Check the neo4j config setting
- default import path
	> Call dbms.listConfig() YIELD name, value
				WHERE name='dbms.directories.import'
				RETURN name, value;

	#### Note: Default: /var/lib/neo4j/import
- customized import directory
	> Call dbms.listConfig() YIELD name, value
				WHERE name='dbms.security.allow_csv_import_from_file_urls'
				RETURN name, value;
	#### Note: Default: true
- Note: Ensure comment out "dbms.directories.import" setting in neo4j.conf to import csv data from preferred directory

### Load csv file 
- from import directory
	> load csv with headers from 'file:///artists.csv' as row return count(row);
- from preferred directory
	> load csv with headers from 'file:////home/jovyan/ml_with_graph_algorithms/nice_graph/neo4j_go/artists_with_header.csv' as row return count(row);


## Thoughts
### 圖的應用
- [Graph DB到底有哪些應用啊？＠webcomm](https://www.webcomm.com.tw/web/tw/neo4j/)
    - ![](https://i.imgur.com/WlhKmdI.png)
- [https://neo4j.com/developer/](https://neo4j.com/cloud/platform/aura-graph-database/?ref=nav-get-started-cta)
- [Bite-Sized Neo4j for Data Scientists](https://github.com/cj2001/bite_sized_data_science)
- [Announcing NODES 2022! Nov. 16-17](https://neo4j.com/blog/nodes-2022/)
- [Videos from NODES 2021](https://neo4j.com/video/nodes-2021/)
    - [9 – Building an ML Pipeline in Neo4j Link Prediction Deep Dive](https://www.youtube.com/watch?v=qdbhCG-Yn74&t=72s&ab_channel=Neo4j)
        - ![](https://i.imgur.com/qvIH5DD.jpg)
    - [8 – Data Warehouse to Graph with Apache Spark](https://www.youtube.com/watch?v=rNKLgqU_mu0&ab_channel=Neo4j)
        - 使用Spark，將巢狀資料解構為可以輸入neo4j的格式 
### 如何增加圖的資訊？  
Q: 如何增加User節點在圖資料庫上的資訊？  
A: 有以下三種可能的方式。  
1. 新增User節點的屬性(property)
2. 新增不同於User節點的新節點種類(異質圖)，如Type節點，並建立User節點與Type節點的連線關係
3. 具有相同屬性的User節點，建立User節點間的連線關係 

備註：一點感想。盡量不要用第三種方法建立關係，因為少量的節點就可能會產生過多(不需要)的連線，如下圖。
<img width="443" alt="image" src="https://user-images.githubusercontent.com/10674490/195478800-c392de39-b61a-40e8-9897-9232c967d84b.png">


## Reference
- [Create a graph database in Neo4j using Python](https://towardsdatascience.com/create-a-graph-database-in-neo4j-using-python-4172d40f89c4)
 
 
