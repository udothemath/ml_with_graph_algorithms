cypher_conf = '''
CALL dbms.listConfig()
YIELD name, value
WHERE name STARTS WITH 'dbms.default'
RETURN name, value
ORDER BY name
LIMIT 3;
'''

cypher_clean = '''
MATCH (n) DETACH DELETE n
'''

DIR_DATA='Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go/data'
FILENAME = 'artists.csv'

cypher_csv_cnt_from_pro = f'''
LOAD CSV WITH HEADERS FROM 'file:///{DIR_DATA}/{FILENAME}' AS row 
RETURN count(row);
'''

# cypher_csv_create_from_pro = f'''
# LOAD CSV FROM 'file:///{DIR_DATA}/{FILENAME}' AS row 
# CREATE (:Artist {name: row[1], year: toInteger(row[2])})
# '''


cypher_csv_cnt_import = f'''
LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row 
RETURN count(row);
'''

cypher_csv_limit_import = f'''
LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row WITH row LIMIT 3 RETURN row;
'''

# https://neo4j.com/developer/desktop-csv-import/
# https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/#load-csv-import-data-from-a-csv-file
