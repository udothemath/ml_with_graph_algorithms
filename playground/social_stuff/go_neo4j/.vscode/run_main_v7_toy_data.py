# %%
import time
# from var import col_var, col_var_one
import numpy as np
import pandas as pd
from IPython.display import display
from neo4j import GraphDatabase
from tqdm import tqdm
import os
import pyarrow
import csv
import logging
from random import randrange
import statistics
from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf, cypher_info, cypher_usage, cypher_node)
from src.neo4j_conn import Neo4jConnection
from src.utils import wrap_log
from src.generate_csv import GenCSVfromDB
from src.table_info import FileInfo
pd.set_option('display.max_columns', 9999)

PATH_BOLT = "bolt://localhost:7687"
DATA_SOURCE = '/home/jovyan/socialnetwork-info/OCR/graph/'
NEO4j_DIR = '/home/jovyan/socialnetwork_info_TFS/go_neo4j'
FOLDER_DATA = os.path.join(NEO4j_DIR, "data")
FOLDER_LOG = os.path.join(NEO4j_DIR, "log")

dict_setting = {
    "NEO4j_DIR": NEO4j_DIR,
    "FOLDER_DATA": FOLDER_DATA,
    "FOLDER_LOG": FOLDER_LOG,   
}

print(f"Setting: {dict_setting}")

logging.basicConfig(
    filename=f'{FOLDER_LOG}/logs_main.log',
    filemode='a+',
    format='%(asctime)s: %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info(
    "===============================================================================")
logger.info("Starting Application Logging")
# %%
class go_main:
    def __init__(self, load_data=None, logger=None):
        self.load_data = load_data
        self.logger = logger

    @wrap_log
    def go_test_read_data(self, n_of_hops=1, repeated_time=2):
        curr_time = time.time()
        id_list = []
        mem_list = []
        for _ in range(repeated_time):
            a_test_id = f"abc_{str(randrange(0, 200))}"
            id_list.append(a_test_id)
            cypher_check_hops = f'''
                match (m {{id:'{a_test_id}'}})-[r*1..{n_of_hops}]-> (n) return count(n) 
            '''
            with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
                print(6, driver.query(cypher_check_hops))
                # print(8, driver.execute_t2(cypher_usage, the_style='df'))
                used_memory = driver.execute_t2(cypher_usage, the_style='abc')[
                    0]['usedMemoryMB']
                mem_list.append(used_memory)

        elapsed_time = time.time()-curr_time
        print(id_list)
        print(mem_list)
        print(f'''
            Elapsed time {elapsed_time:.4f} seconds with {repeated_time} runs. 
            Each run: {elapsed_time/repeated_time:.4f} seconds.
            Ave memory usage: {statistics.mean(mem_list)}, stdev: {statistics.stdev(mem_list):.2f}
            '''
              )

    def cypher_info_company(self):
        cypher_company_constraint = '''
            CREATE CONSTRAINT uniq_company IF NOT EXISTS 
            FOR (n:The_company_id) REQUIRE n.to IS UNIQUE
        '''
        file_id_company = os.path.join(
            NEO4j_DIR, f"data_l2_link_works_sample.csv")
        cypher_file_id_company = f'''
            USING PERIODIC COMMIT 1000
            LOAD CSV WITH HEADERS FROM 'file:///{file_id_company}' AS row 
            MERGE (id_from:The_id_number {{身分證字號: row.from}}) 
            MERGE (id_company:The_company_id {{公司名: row.to}}) 
        '''
        return cypher_company_constraint, cypher_file_id_company

    @wrap_log
    def go_load_data(self):

        csv_node_relation = self.load_data

        cypher_id_constraint = '''
        CREATE CONSTRAINT unique_the_id IF NOT EXISTS 
        FOR (m:The_id) REQUIRE m.from IS UNIQUE
        '''

        cypher_id_to_constraint = '''
        CREATE CONSTRAINT unique_the_id_to IF NOT EXISTS 
        FOR (n:The_id) REQUIRE n.to IS UNIQUE
        '''

        cypher_link_constraint = '''
        CREATE CONSTRAINT unique_the_link IF NOT EXISTS 
        FOR (r:The_link) REQUIRE r.link_type IS UNIQUE
        '''

        cypher_csv_node_and_relation = f'''
        USING PERIODIC COMMIT 2000
        LOAD CSV WITH HEADERS FROM 'file:///{csv_node_relation}' AS row 
        MERGE (id_from:The_id {{ID: row.from}}) 
        MERGE (id_to:The_id {{ID: row.to}}) 
        MERGE (link:The_link {{ID: row.link_type}}) 
        MERGE (id_from)-[:The_link{{Relation: row.link_type}}]->(id_to)
        '''


        query_list = [
            cypher_clean,
            # cypher_path,
            # cypher_info,
            cypher_id_constraint,
            cypher_id_to_constraint,
            cypher_link_constraint,
            cypher_csv_node_and_relation,
            cypher_node
        ]

        query_memory = [cypher_usage]

        self.logger.info("in go_test")
        with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
            for i, curr_query in enumerate(query_list, start=1):
                print(i, driver.query(curr_query))

            for i, curr_query in enumerate(query_memory, start=100):
                print(i, driver.execute_t2(curr_query))
        print("Done")


if __name__ == "__main__":
    start_time = time.time()
    print(f"{'-'*20}")
    file_1 = FileInfo(table_name="all_links", save_dir=FOLDER_DATA, 
                    save_file_prefix="aaa", size_limit=None)
    file_with_path = file_1.get_path
    if os.path.isfile(file_with_path):
        print(f"u already have file: {file_with_path}")
    else:
        print(f"file doesn't exists. Creating file...")
        GenCSVfromDB(file_1, logger=logger).create_csv_from_df()
        print(f"file is ready. {file_with_path}")

    a_script = go_main(file_with_path, logger)
    a_script.go_load_data()

    # for i in range(1, 7):
    #     print(f"{'*'*20} - number of hop: {i}")
    #     a_script.go_test_read_data(n_of_hops = i, repeated_time = 10)
    # print(f"{'-'*20}")
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print("---Done---")

# %%
# %%
