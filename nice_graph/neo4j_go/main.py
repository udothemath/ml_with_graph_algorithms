# %%
import json
import time

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf,
                             cypher_csv_cnt_from_pro, cypher_csv_cnt_import,
                             cypher_html_csv, cypher_node, load_csv_as_row)
from src.neo4j_conn import Neo4jConnection

PATH_BOLT = "bolt://localhost:7687"
conn = Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD)

def create_node():
    arg = "CREATE (a:Artist {name: row.Name, year: toInteger(row.Year)}) Return linenumber()-1 AS number, a.name, a.year"
    cypher_create_node = f'''
    {load_csv_as_row} {arg} 
    '''
    print(conn.query(cypher_create_node))


def main():
    # print(conn.query(cypher_conf))
    print(conn.query(cypher_clean))
    # create_node()
    print(conn.query(cypher_html_csv))
    # print(conn.query(cypher_csv_cnt_from_pro))
    try:   
        print(conn.query(cypher_csv_create_from_pro))
    except:
        print('U don\'t have cypher_csv_create_from_pro')
    print('done')

if __name__=="__main__":
    print(f"{'-'*20}")
    main()
    print(f"{'-'*20}")
