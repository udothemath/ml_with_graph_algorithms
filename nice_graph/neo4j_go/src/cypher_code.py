cypher_clean = '''
MATCH (n) DETACH DELETE n
'''

# https://neo4j.com/developer/desktop-csv-import/
cypher_csv_limit = '''
LOAD CSV WITH HEADERS FROM 'file:///companies.csv' AS row WITH row LIMIT 3 RETURN row;
'''

cypher_csv_cnt = '''
LOAD CSV WITH HEADERS FROM 'file:///companies.csv' AS row 
RETURN count(row);
'''
#
