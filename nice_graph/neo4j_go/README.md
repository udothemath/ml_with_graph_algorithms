# go for neo4j
## off-line
1. Set up virtual environment
   - $python -m venv <name_of_venv_with_path>
   - source <name_of_venv_with_path>/bin/activate
2. Install package
   - pip install -r requirements.txt
3. Initialize neo4j (graph database)
4. Collect info for database connection
5. Execute

## aicloud
###  Connect neo4j
0. Ensure that you pick the correct image (esun_graph)
1. Reset the neo4j setting
	- $source reset_neo4j_config.sh
	- Note: including the following setting in neo4j.conf
		- comment out import setting
			- '# dbms.directories.import=/var/lib/neo4j/import'
		- uncomment following line to import by desired
			- 'dbms.security.allow_csv_import_from_file_urls=true'
		- disable authentication (no need for user and password)
			- 'dbms.security.auth_enabled=false'
2. Connect neo4j: $neo4j console &
3. There are three ways to communicate with neo4j
	- shell: $cypher-shell
	- python script: $python main.py
	- through browser: localhost:7474
### Check the neo4j config setting
- default import path
	>>> $ Call dbms.listConfig() YIELD name, value
				WHERE name='dbms.directories.import'
				RETURN name, value;
	>>> Default: /var/lib/neo4j/import
- customized import directory
>>> $ Call dbms.listConfig() YIELD name, value
			WHERE name='dbms.security.allow_csv_import_from_file_urls'
			RETURN name, value;
>>> Default: true
- Note: Ensure comment out "dbms.directories.import" setting in neo4j.conf to import csv data from directory

### Load csv file 
- from import directory
>>> $ load csv with headers from 'file:///artists.csv' as row return count(row);
- from preferred directory
>>> $ load csv with headers from 'file:////home/jovyan/ml_with_graph_algorithms/nice_graph/neo4j_go/artists_with_header.csv' as row return count(row);
 
 