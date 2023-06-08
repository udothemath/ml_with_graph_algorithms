# %%
# from var import col_var, col_var_one
import numpy as np
import pandas as pd
from IPython.display import display
from neo4j import GraphDatabase
from tqdm import tqdm
import os
import pyarrow
import csv
import random

from setting import NEO4J_PASSWORD, NEO4J_USER
# from src.cypher_code import (cypher_clean, cypher_conf,
#                              cypher_csv_cnt_from_pro, cypher_csv_cnt_import,
#                              cypher_csv_create_from_pro, cypher_html_csv,
#                              cypher_info, cypher_node, load_csv_as_row)
from src.neo4j_conn import Neo4jConnection
pd.set_option('display.max_columns', 9999)

PATH_BOLT = "bolt://localhost:7687"
DATA_SOURCE = '/home/jovyan/socialnetwork-info/OCR/graph/'
NEO4j_DIR = '/home/jovyan/socialnetwork_info_TFS/go_neo4j'

print(DATA_SOURCE)

# print(f"NEO4J user:{NEO4J_USER}. password: {NEO4J_PASSWORD}")

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

# %%


class gen_csv_file:
    def __init__(self, id_prefix='ab', num_rows=10, path=NEO4j_DIR):
        """
            generate sample csv file
        """
        self.__id_prefix = id_prefix
        self.__num_rows = num_rows
        self.__path = path

    def gen_node(self, num_cols=2) -> None:
        """
        generate csv file for node

        Parameters:
        num_rows (int): number of rows
        num_cols (int): number of columns

        Returns:
        None. The csv file will be generated at desired path

        Raises:
        TypeError: If num_rows or num_cols is not an integer
        """
        if not isinstance(self.__num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("Input must be a number.")

        file_with_path = os.path.join(
            self.__path, f"data_node_r{self.__num_rows}_c{num_cols}.csv")
        col_name = []
        for i in range(num_cols):
            col_name.append(f'Column_{i}')

        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # Write the data rows
            for i in range(self.__num_rows):
                row = [f'{self.__id_prefix}_{i}']
                for j in range(1, self.__num_rows):
                    row.append(random.randint(0, 100))
                csvwriter.writerow(row)

        print(
            f"Ur file with rows:{self.__num_rows} and cols:{num_cols} is ready.")

    def gen_relation(self, num_cols=2) -> None:
        """
        generate csv file for relation

        Parameters:
        num_rows (int): number of rows
        num_cols (int): number of columns

        Returns:
        None. The csv file will be generated at desired path

        Raises:
        TypeError: If num_rows or num_cols is not an integer
        """
        if not isinstance(self.__num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("Input must be a number.")

        file_with_path = os.path.join(self.__path,
                                      f"data_relation_r{self.__num_rows}_c{num_cols}.csv")
        col_name = ['from', 'to']
        half_index = int(self.__num_rows/2)
        from_id = [f"{self.__id_prefix}_{2*i}" for i in range(half_index)]
        to_id = [f"{self.__id_prefix}_{2*i+1}" for i in range(half_index)]
        rows = zip(from_id, to_id)

        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        print(f"Ur file of relation with rows:{self.__num_rows} is ready.")


# %%
def go_test(test_n_rows=1000):
    print("number pf rows:", test_n_rows)
    a = gen_csv_file(id_prefix='abc', num_rows=test_n_rows)
    # a.gen_node()
    a.gen_relation()
    csv_relation = os.path.join(
        NEO4j_DIR, f"data_relation_r{test_n_rows}_c2.csv")
    cypher_csv_relation = f'''
    USING PERIODIC COMMIT 100000
    LOAD CSV WITH HEADERS FROM 'file:///{csv_relation}' AS row 
    MERGE (id_from:The_id {{id: row.from}}) 
    MERGE (id_to:The_id {{id: row.to}}) 
    MERGE (id_from)-[:Relation]->(id_to)
    '''
    cypher_usage = '''
    CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes
    RETURN attributes.HeapMemoryUsage.value.properties.max/1024/1024 as maxMemoryMB,
    attributes.HeapMemoryUsage.value.properties.used/1024/1024 as usedMemoryMB,
    attributes.HeapMemoryUsage.value.properties.committed/1024/1024 as committedMemoryMB
    '''
    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        print(1, driver.query(cypher_clean))
        # print(2, driver.query(cypher_path))
        # print(3, driver.query(cypher_info))
        print(4, driver.query(cypher_csv_relation))
        # print(5, driver.query(cypher_f1))
        # print(6, driver.query(cypher_f2))
        # print(7, driver.query(cypher_node))
        print(8, driver.execute_t2(cypher_usage, the_style='df'))
        print(9, driver.execute_t2(cypher_usage,
              the_style='abc')[0]['usedMemoryMB'])
        # print(9, driver.execute(the_piece))
        # print(10, driver.execute_t2(the_piece))

    # if a:
    #     del a
    print("Done")


go_test(test_n_rows=1000)


##############
# %%
def main():
    csv_f0 = os.path.join(NEO4j_DIR, 'test_file.csv')
    csv_f1 = os.path.join(DATA_SOURCE, 'demo0', 'node', "customers.csv")
    csv_f2 = os.path.join(DATA_SOURCE, 'demo0', 'node', "fathers.csv")

    csv_node = os.path.join(NEO4j_DIR, "data_node_r20_c2.csv")
    csv_relation = os.path.join(NEO4j_DIR, "data_relation_r20_c2.csv")

    cypher_csv_node = f'''
    LOAD CSV WITH HEADERS FROM 'file:///{csv_node}' AS row 
    CREATE (:the_id {{id_value: row[0], number: row[1]}})
    '''

    cypher_csv_relation = f'''
    :aoto USING PERIODIC COMMIT 10000
    LOAD CSV WITH HEADERS FROM 'file:///{csv_relation}' AS row 
    MERGE (id_from:The_id {{id: row.from}}) 
    MERGE (id_to:The_id {{id: row.to}}) 
    MERGE (id_from)-[:Relation]->(id_to)
    '''

    # load_csv_comp_conn = \
    #     f'''
    #     LOAD CSV WITH HEADERS FROM 'file:///{csv_comp}' AS row
    #     MERGE (n:Sys_comp {{sys_comp: row.元件}})
    #     MERGE (m:Sys_lead {{sys_lead: row.元件負責人}})
    #     MERGE (o:Proj {{proj: row.專案}})
    #     MERGE (n)-[rel_comp_lead:Comp_Lead]->(m)
    #     MERGE (n)-[:Comp_Proj]->(o)
    #     return count(rel_comp_lead)
    #     '''

    cypher_f0 = f'''
    LOAD CSV FROM 'file:///{csv_f0}' AS row 
    CREATE (:test_id {{id_value: row[0], number: row[1]}})
    '''

    cypher_f1 = f'''
    LOAD CSV FROM 'file:///{csv_f1}' AS row 
    CREATE (:id {{id_value: row[0], name: row[1]}})
    '''

    cypher_f2 = f'''
    LOAD CSV FROM 'file:///{csv_f2}' AS row 
    CREATE (:father {{name: row[1], etl_dt: row[2]}})
    '''



    cypher_rel1 = f'''
    LOAD CSV FROM 'file:///{csv_f1}' AS row 
    CREATE (:id {{id_value: row[0], name: row[1]}})
    '''

    cypher_usage = '''
    CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes
    RETURN attributes.HeapMemoryUsage.value.properties.max/1024/1024 as maxMemoryMB,
    attributes.HeapMemoryUsage.value.properties.used/1024/1024 as usedMemoryMB,
    attributes.HeapMemoryUsage.value.properties.committed/1024/1024 as committedMemoryMB

    '''

    # CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes
    # WHERE attributes.Name = 'HeapMemoryUsage'
    # RETURN attributes.used / 1024 / 1024 AS usedMemoryMB,
    # attributes.committed / 1024 / 1024 AS committedMemoryMB,
    # attributes.max / 1024 / 1024 AS maxMemoryMB

    # load_csv_comp_conn = \
    #     f'''
    #     LOAD CSV WITH HEADERS FROM 'file:///{csv_comp}' AS row
    #     MERGE (n:Sys_comp {{sys_comp: row.元件}})
    #     MERGE (m:Sys_lead {{sys_lead: row.元件負責人}})
    #     MERGE (o:Proj {{proj: row.專案}})
    #     MERGE (n)-[rel_comp_lead:Comp_Lead]->(m)
    #     MERGE (n)-[:Comp_Proj]->(o)
    #     return count(rel_comp_lead)
    #     '''

    # load_csv_pm_conn = \
    #     f'''
    #     LOAD CSV WITH HEADERS FROM 'file:///{csv_pm}' AS row
    #     MERGE (o:Proj {{proj: row.專案}})
    #     MERGE (r:PM {{pm: row.PM}})
    #     MERGE (o)-[rel_proj_pm:Proj_PM]->(r)
    #     return count(rel_proj_pm)
    #     '''

    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        print(1, driver.query(cypher_clean))
        # print(2, driver.query(cypher_path))
        # print(3, driver.query(cypher_info))
        print(4, driver.query(cypher_csv_relation))
        # print(5, driver.query(cypher_f1))
        # print(6, driver.query(cypher_f2))
        # print(7, driver.query(cypher_node))
        print(8, driver.execute_t2(cypher_usage, the_style='df'))
        print(9, driver.execute_t2(cypher_usage,
              the_style='abc')[0]['usedMemoryMB'])
        # print(9, driver.execute(the_piece))
        # print(10, driver.execute_t2(the_piece))


main()


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
