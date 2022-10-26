# %%
from neo4j import GraphDatabase, basic_auth
import json
import time
from asyncore import read
from var import col_var, col_var_one
import numpy as np
import pandas as pd
from IPython.display import display
from neo4j import GraphDatabase
from tqdm import tqdm

from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf,
                             cypher_csv_cnt_from_pro, cypher_csv_cnt_import,
                             cypher_csv_create_from_pro, cypher_html_csv,
                             cypher_info, cypher_node, load_csv_as_row)
from src.neo4j_conn import Neo4jConnection
pd.set_option('display.max_columns', 9999)

PATH_BOLT = "bolt://localhost:7687"
DATA_SOURCE = '/Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go/data'
FILE_NAME = f'{DATA_SOURCE}/6000set2_2022_04-05/2022-05-01/info.csv'


def read_csv_as_chunk(file_name: str, sample_size: int, chunk_size=1000):
    reader = pd.read_csv(file_name, header=0, nrows=sample_size,
                         iterator=True, low_memory=False)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Finish reading csv. Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac


def save_df_to_csv(input_df: pd.DataFrame(), to_filename: str, to_path=DATA_SOURCE) -> None:
    file_with_path = f"{to_path}/{to_filename}"
    try:
        input_df.to_csv(f"{file_with_path}", index=False)
        print(f"U have successfully save file {file_with_path}")
        aa
    except Exception as e:
        print("Fail to save csv file")
        raise e


def gen_cypher(file_name: str = FILE_NAME):

    data_path = f"{file_name}"
    cypher_csv_create_from_pro = f'''
        LOAD CSV FROM 'file:///{data_path}' AS row
        CREATE (:USER {{num_hash: row[0], search_dt: row[1]}})
    '''
    print(cypher_csv_create_from_pro)
    return cypher_csv_create_from_pro


def gen_cypher_info():

    file_name = f'{DATA_SOURCE}/6000set2_2022_04-05/2022-05-01/info.csv'

    df_all = pd.read_csv(file_name, header=0)
    print(df_all.shape)
    print(df_all.columns)
    # info size: 5485, 9

    # df = read_csv_as_chunk(file_name, sample_size=1000, chunk_size=1000)

    col_list = ['num_hash', 'search_dt', 'status_verify_yn', 'info_yp_categ',
                'info_yp_tag', 'info_yp_source', 'info_special_tag',
                'info_special_categ', 'info_spam_categ']

    # df_keep = df[col_list][:3]
    # display(df_keep)
    # display(df_keep[['num_hash','search_dt','status_verify_yn']])

    # save_df_to_csv(df_keep, to_filename='test_df_v2')

    data_path = f"{file_name}"
    cypher_task = f'''
        LOAD CSV FROM 'file:///{data_path}' AS row_life
        CREATE (:NUM_ID {{num_hash: row_life[0], info_yp_categ: row_life[3]}})
        CREATE (:SPECIAL_ID {{info_special_tag: row_life[6], info_special_categ: row_life[7]}})
    '''
    print(cypher_task)
    return cypher_task


def main():
    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        print(driver.query(cypher_clean))
        print(driver.query(cypher_conf))
        # ## print(driver.query(cypher_csv_cnt_from_pro))
        print(driver.query(gen_cypher()))
        # print(driver.query(gen_cypher_info()))

        # cypher_csv_create_rel = f'''
        #     MATCH (a:NUM_ID), (b:NUM_ID)
        #     WHERE a.info_yp_categ = b.info_yp_categ
        #     AND a <> b
        #     CREATE (a)-[r:YP_CATEG]->(b)
        #     RETURN count(r) as cnt
        # '''
        # print(driver.query(cypher_csv_create_rel))
        # # ## print(driver.query(cypher_html_csv))

# %%

# if __name__=="__main__":
#     print(f"{'-'*20}")
#     gen_cypher()
#     # main()
#     print(f"{'-'*20}")

# %%
