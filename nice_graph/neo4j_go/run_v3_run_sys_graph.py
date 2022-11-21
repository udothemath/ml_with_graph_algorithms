# %%
import json
import time
from asyncore import read
from var import col_var, col_var_one
import numpy as np
import pandas as pd
from IPython.display import display
from neo4j import GraphDatabase
from tqdm import tqdm
import os
import pyarrow

# from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf,
                             cypher_csv_cnt_from_pro, cypher_csv_cnt_import,
                             cypher_csv_create_from_pro, cypher_html_csv,
                             cypher_info, cypher_node, load_csv_as_row)
from src.neo4j_conn import Neo4jConnection
pd.set_option('display.max_columns', 9999)

PATH_BOLT = "bolt://localhost:7687"

ON_PRO = True

if ON_PRO:
    DATA_SOURCE = '/Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go/data'
else:
    DATA_SOURCE = 'Path_on_aicloud'

print(DATA_SOURCE)

file_comp = 'graph_node_analysis_v1_comp'
file_pm = 'graph_node_analysis_v1_pm'


def excel_to_csv(filename: str):
    xlsx_with_path = os.path.join(
        DATA_SOURCE, f"{filename}.xlsx")

    filename_as_csv = os.path.join(DATA_SOURCE, f"csv_{filename}.csv")
    df = pd.read_excel(xlsx_with_path, index_col=None,
                       header=0, engine='openpyxl')
    df.to_csv(filename_as_csv, index=None, header=True)
    print(f"U have saved {filename_as_csv}")


excel_to_csv(filename=file_comp)
excel_to_csv(filename=file_pm)

# %%
# %%


df_comp = pd.read_excel(os.path.join(DATA_SOURCE, file_comp),
                        index_col=None, header=0, engine='openpyxl')

df_pm = pd.read_excel(os.path.join(DATA_SOURCE, file_pm),
                      index_col=None, header=0, engine='openpyxl')


def print_info(df: pd.DataFrame, info_string: str, verbose: bool = False):
    print(f"{'-'*20} Info of {info_string} {'-'*20}")
    display(df[:3])
    print("Shape:", df.shape)
    print("Number of unique type in columns: ")
    for i in df.columns:
        print(f"    {i}", df[i].nunique())
    if verbose:
        print(df.value_counts())
        for i in df.columns:
            print(df[i].value_counts())


print_info(df_comp, 'comp', False)
print_info(df_pm, 'pm', False)
# %%

# %%
# cond = (df_one.columns.str.contains('info'))

col_info = [
    'info_yp_categ',
    'info_yp_tag',
    'info_yp_source',
    'info_special_tag',
    'info_special_categ',
    'info_spam_categ']

display(df_one[col_info].describe())
print(df_one.shape)

# df_check = df_one[df_one.columns[cond]]
# display(df_check[:3])

# df[df.columns[df.columns.str.contains("spike|spke")]]

print("Done")

# check_whos_parquet()

# %%
# %%


def read_csv_as_chunk(fname, sample_size, chunk_size=1000):
    reader = pd.read_csv(fname, header=0, nrows=sample_size,
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


# gen_cypher()
# gen_cypher_info()

# df = check_data(file_name=FILE_NAME)


def df_for_relation(df):
    print(df.shape)
    display(df[:3])

    # def add_ind(df):
    col_keep = ['num_hash']
    for curr_i in col_var_one:
        var_mean = df[curr_i].mean(axis=0)
        curr_i_ind = f'ind_{curr_i}'
        col_keep.append(curr_i_ind)
        df[curr_i_ind] = np.where(df[curr_i] > var_mean, 1, 0)

    rows = df[col_keep]

    display(rows[:3])
    print(rows.shape)
    query = f'''
        UNWIND $rows as row
        CREATE (xxx:USER {{
            num_hash: row.num_hash,
            ind_type1: row.ind_cluster_life_car_trading_total_count_in_1m,
            ind_type2: row.ind_cluster_life_car_trading_qc_count_in_1m 
            }})
        MERGE (yyy:REL_TYPE2 {{ind_type2: row.ind_cluster_life_car_trading_qc_count_in_1m}})
        WITH xxx, yyy
        MATCH (a:USER), (b:USER)
        WHERE a.ind_type2 = 1
        and b.ind_type2 = 1
        and a <> b
        CREATE (a)-[r:r_ind_type2]->(b) 
        with r
        RETURN count(r) 
    '''

    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        print(driver.query(cypher_clean))
        # print(driver.query(gen_cypher()))
        print(driver.query(query, parameters={
              'rows': rows.to_dict('records')}))

    print("done")


# df = check_data(sample_size=1000, file_name=FILE_NAME)
# df_for_relation(df)
# main()

# %%

# if __name__=="__main__":
#     print(f"{'-'*20}")
#     gen_cypher()
#     # main()
#     # data_csv_process(sample_size=1000)
#     print(f"{'-'*20}")

# %%


# %%
