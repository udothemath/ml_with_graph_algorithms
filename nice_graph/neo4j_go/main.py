# %%
%load_ext autoreload
%autoreload 2
# %%
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time

from src.neo4j_conn import Neo4jConnection

path_bolt="bolt://127.0.0.1:7687"

conn = Neo4jConnection(uri=f"{path_bolt}", user="neo4j", pwd="1234")

cypher_code = '''
LOAD CSV WITH HEADERS from 'file:///csv_mdsjr_hist2022_mini1000_asof202206.csv' as row 
with row.oba as id_from, row.psu_acn1 as id_to, 
row.amt as amt, row.txd as txd
match (node_1:ID {id: id_from})
match (node_2:ID {id: id_to})
merge (node_1)-[rel:TRANSFER_TO {amount: toInteger(amt), date: txd}]->(node_2)
'''

cypher_code = '''
CREATE (p:Person {name:"AAA"})-[r:SAYS]->(message:Message {name:"Hello World!"}) 
RETURN p, message, r

'''

conn.query(cypher_code)



# if __name__=="__main__":
#     print("hello")
#     conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="1234")
#     print("good bye")

# %%


# %%
