cypher_clean = '''
MATCH (n) DETACH DELETE n
'''

cypher_conf = '''
CALL dbms.listConfig()
YIELD name, value
WHERE name STARTS WITH 'dbms.default'
RETURN name, value
ORDER BY name
LIMIT 3;
'''

cypher_info = '''
CALL dbms.listConnections() YIELD connectionId, connectTime, connector, username, userAgent, clientAddress
'''

cypher_usage = '''
CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes
RETURN attributes.HeapMemoryUsage.value.properties.max/1024/1024 as maxMemoryMB,
attributes.HeapMemoryUsage.value.properties.used/1024/1024 as usedMemoryMB,
attributes.HeapMemoryUsage.value.properties.committed/1024/1024 as committedMemoryMB
'''

cypher_node = '''MATCH (n) return count(n) '''

# DIR_DATA = 'Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go/data'
# FILENAME = 'artists_with_header.csv'

# HTML_CSV = 'https://gist.githubusercontent.com/jvilledieu/c3afe5bc21da28880a30/raw/a344034b82a11433ba6f149afa47e57567d4a18f/Companies.csv'

# cypher_html_csv = f'''
# LOAD CSV WITH HEADERS FROM '{HTML_CSV}' AS row Return count(row);
# '''

# load_csv_as_row = f'''LOAD CSV WITH HEADERS FROM 'file:///{DIR_DATA}/{FILENAME}' AS row '''

# cypher_csv_cnt_from_pro = f'''
# {load_csv_as_row} RETURN count(row);
# '''

# cypher_csv_create_from_pro = f'''
# LOAD CSV FROM 'file:///{DIR_DATA}/{FILENAME}' AS row
# CREATE (:Artist {{name: row[1], year: toInteger(row[2])}})
# '''


# cypher_csv_cnt_import = f'''
# LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row
# RETURN count(row);
# '''

# cypher_csv_limit_import = f'''
# LOAD CSV WITH HEADERS FROM 'file:///{FILENAME}' AS row WITH row LIMIT 3 RETURN row;
# '''

# MATCH (a:USER), (b:USER)
# WHERE a.num_hash = row.num_hash
# CREATE (:USER {prop: row.ind_cluster_life_car_trading_total_count_in_1m})
# RETURN count(*) as cnt

#     MATCH (a:NUM_ID), (b:NUM_ID)
#     WHERE a.info_yp_categ = b.info_yp_categ
#     AND a <> b
#     CREATE (a)-[r:YP_CATEG]->(b)
#     RETURN count(r) as cnt

# query = '''
#     UNWIND $rows as row
#     MERGE (:Ind_v2 {
#         num:row.num_hash,
#         ind_ct_total:row.ind_cluster_life_car_trading_total_count_in_1m,
#         ind_ct_qc: row.ind_cluster_life_car_trading_qc_count_in_1m})
#     return count(*)
# '''


# https://neo4j.com/developer/desktop-csv-import/
# https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/#load-csv-import-data-from-a-csv-file
