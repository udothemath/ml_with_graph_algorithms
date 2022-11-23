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

from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf,
                             cypher_csv_cnt_from_pro, cypher_csv_cnt_import,
                             cypher_csv_create_from_pro, cypher_html_csv,
                             cypher_info, cypher_node, load_csv_as_row)
from src.neo4j_conn import Neo4jConnection
pd.set_option('display.max_columns', 9999)

PATH_BOLT = "bolt://localhost:7687"

ON_PRO = False

if ON_PRO:
    DATA_SOURCE = '/Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go/data'
else:
    DATA_SOURCE = '/home/jovyan/ml_with_graph_algorithms/nice_graph/neo4j_go/data'

print(DATA_SOURCE)

print(f"NEO4J user:{NEO4J_USER}. password: {NEO4J_PASSWORD}")
# %%

file_comp = 'graph_node_analysis_v1_comp'
file_pm = 'graph_node_analysis_v1_pm'


def excel_to_csv(filename: str):
    xlsx_with_path = os.path.join(
        DATA_SOURCE, f"{filename}.xlsx")

    filename_as_csv = os.path.join(DATA_SOURCE, f"csv2_{filename}.csv")
    df = pd.read_excel(xlsx_with_path, index_col=None,
                       header=0, engine='openpyxl')

    for i in df.columns[df.dtypes == object].tolist():
        df[i] = df[i].str.strip()

    df.to_csv(filename_as_csv, index=None, header=True)
    print(f"U have saved {filename_as_csv}")


# excel_to_csv(filename=file_comp)
# excel_to_csv(filename=file_pm)

# %%
df_comp = pd.read_excel(os.path.join(DATA_SOURCE, f"{file_comp}.xlsx"),
                        index_col=None, header=0, engine='openpyxl')

df_pm = pd.read_excel(os.path.join(DATA_SOURCE, f"{file_pm}.xlsx"),
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

def main():
    csv_comp = os.path.join(DATA_SOURCE, f"csv2_{file_comp}.csv")
    csv_pm = os.path.join(DATA_SOURCE, f"csv2_{file_pm}.csv")

    load_csv_comp_conn = \
        f'''
        LOAD CSV WITH HEADERS FROM 'file:///{csv_comp}' AS row 
        MERGE (n:Sys_comp {{sys_comp: row.元件}})    
        MERGE (m:Sys_lead {{sys_lead: row.元件負責人}}) 
        MERGE (o:Proj {{proj: row.專案}})  
        MERGE (n)-[rel_comp_lead:Comp_Lead]->(m)
        MERGE (n)-[:Comp_Proj]->(o)
        return count(rel_comp_lead)
        '''

    load_csv_pm_conn = \
        f'''
        LOAD CSV WITH HEADERS FROM 'file:///{csv_pm}' AS row
        MERGE (o:Proj {{proj: row.專案}})  
        MERGE (r:PM {{pm: row.PM}})
        MERGE (o)-[rel_proj_pm:Proj_PM]->(r)
        return count(rel_proj_pm)
        '''

    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        print(driver.query(cypher_clean))
        print(driver.query(cypher_conf))
        print(driver.query(cypher_info)) 
        print(driver.query(load_csv_comp_conn))
        print(driver.query(load_csv_pm_conn))

main()
# %%
# %%

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


# %%

# if __name__=="__main__":
#     print(f"{'-'*20}")
#     gen_cypher()
#     # main()
#     # data_csv_process(sample_size=1000)
#     print(f"{'-'*20}")

# %%
