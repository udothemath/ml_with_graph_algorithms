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

FILENAME = 'artists_with_header.csv'

# https://neo4j.com/developer/desktop-csv-import/
# https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/#load-csv-import-data-from-a-csv-file


cypher_csv_cnt_from_dir = f'''
LOAD CSV WITH HEADERS FROM 'file:///Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go/artists.csv' AS row 
RETURN count(row);
'''


cypher_csv_cnt_import = f'''
LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row 
RETURN count(row);
'''

cypher_csv_limit_import = f'''
LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row WITH row LIMIT 3 RETURN row;
'''
