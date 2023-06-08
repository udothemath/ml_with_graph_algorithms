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
		- please ensure that you have following two port forwarding
			- localhost:7474 (step1: for neo4j connection)
			- localhost:7687 (step2: for graph db connection)

4. Once the setting of forwarded ports is ready, you can go to browser and make connection with neo4j using ``` http://localhost:7474/browser/ ```             

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
Q: 如何增加User節點在圖資料庫上的資訊？  
A: 有以下三種可能的方式。  
1. 新增User節點的屬性(property)
2. 新增不同於User節點的新節點種類(異質圖)，如Type節點，並建立User節點與Type節點的連線關係
3. 具有相同屬性的User節點，建立User節點間的連線關係 

備註：一點感想。盡量不要用第三種方法建立關係，因為少量的節點就可能會產生過多(不需要)的連線，如下圖。
<img width="443" alt="image" src="https://user-images.githubusercontent.com/10674490/195478800-c392de39-b61a-40e8-9897-9232c967d84b.png">




## Reference
- [Create a graph database in Neo4j using Python](https://towardsdatascience.com/create-a-graph-database-in-neo4j-using-python-4172d40f89c4)