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
Q: 如何增加圖的資訊？  
A: 有以下三種可能的方式。  
1. 新增節點的屬性(property)
2. 新增節點種類(異質圖)，並依照節點的屬性建立關係
3. 具有相同屬性的節點，建立節點與節點間關係連線 

備註：一點感想。盡量不要用第三種方法建立關係連線，因為可能會造成過多的連線，如下圖。
<img width="443" alt="image" src="https://user-images.githubusercontent.com/10674490/195478800-c392de39-b61a-40e8-9897-9232c967d84b.png">




## Reference
- [Create a graph database in Neo4j using Python](https://towardsdatascience.com/create-a-graph-database-in-neo4j-using-python-4172d40f89c4)
 
 
