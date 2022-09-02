# %%
import json
import time

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from src.cypher_code import (cypher_clean, cypher_conf,
                             cypher_csv_cnt_from_pro, cypher_csv_cnt_import)
from src.neo4j_conn import Neo4jConnection

# %load_ext autoreload
# %autoreload 2
PATH_BOLT = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "1239994"

conn = Neo4jConnection(uri=PATH_BOLT, user=USER, pwd=PASSWORD)


# print(conn.query(cypher_conf))

print(conn.query(cypher_csv_cnt_from_pro))
# print(conn.query(cypher_csv_cnt_import))


# %%
# cypher_code = '''
# MATCH (n) return n limit 4
# '''
# print(conn.query(cypher_code))


# cypher_code = '''
# MATCH (n:Company) return n limit 4
# '''
# conn.query(cypher_code)

# if __name__=="__main__":
#     print("hello")
#     conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="1234")
#     print("good bye")

# %%
