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

FILENAME = 'artists_with_header'+'.csv'

# https://neo4j.com/developer/desktop-csv-import/
# https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/#load-csv-import-data-from-a-csv-file
cypher_csv_limit = f'''
LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row WITH row LIMIT 3 RETURN row;
'''

cypher_csv_limit_path = f'''
LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row WITH row LIMIT 3 RETURN row;
'''

cypher_csv_cnt = '''
LOAD CSV WITH HEADERS FROM 'file:///companies.csv' AS row 
RETURN count(row);
'''
#
