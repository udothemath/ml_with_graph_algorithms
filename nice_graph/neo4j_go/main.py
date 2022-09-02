# %%
import json
import time

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf,
                             cypher_csv_cnt_from_pro, cypher_csv_cnt_import)
from src.neo4j_conn import Neo4jConnection

PATH_BOLT = "bolt://localhost:7687"
conn = Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD)

def main():
    # print(conn.query(cypher_conf))
    print(conn.query(cypher_csv_cnt_from_pro))
    # print(conn.query(cypher_csv_cnt_import))

if __name__=="__main__":
    print(f"{'-'*20}")
    main()
    print(f"{'-'*20}")

# %%
