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
# from src.cypher_code import (cypher_clean, cypher_conf,
#                              cypher_csv_cnt_from_pro, cypher_csv_cnt_import,
#                              cypher_csv_create_from_pro, cypher_html_csv,
#                              cypher_info, cypher_node, load_csv_as_row)
from src.neo4j_conn import Neo4jConnection
from src.utils import wrap_log
pd.set_option('display.max_columns', 9999)

PATH_BOLT = "bolt://localhost:7687"
DATA_SOURCE = '/home/jovyan/socialnetwork-info/OCR/graph/'
NEO4j_DIR = '/home/jovyan/socialnetwork_info_TFS/go_neo4j'

print(DATA_SOURCE)

logging.basicConfig(
    filename=f'{NEO4j_DIR}/logs_main.log',
    filemode='a+',
    format='%(asctime)s: %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)
logger.info(
    "===============================================================================")
logger.info("Starting Application Logging")


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

cypher_usage = '''
CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes
RETURN attributes.HeapMemoryUsage.value.properties.max/1024/1024 as maxMemoryMB,
attributes.HeapMemoryUsage.value.properties.used/1024/1024 as usedMemoryMB,
attributes.HeapMemoryUsage.value.properties.committed/1024/1024 as committedMemoryMB
'''

# query_list = [
#     'match (n) return count(n)',
#     'match (n) return count(n)'
# ]

# def query_on_graph(graph_cypher_query_list: list):
#     with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
#         for i, j in enumerate(graph_cypher_query_list, 1):
#             print(i, driver.query(j))
#             logger.info(f"cyphyer query {i:4d}: {j}")
#             print("--- Done ---")

# query_on_graph(query_list)
# match (m {id: 'abc_151'})-[r*1..5]->(n) return n limit 100;

# %%
class go_main:
    def __init__(self, logger):
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
    def go_read_ocr_data(self, read_size=500):

        csv_node_relation = os.path.join(NEO4j_DIR, f"ocr_size{read_size}.csv")
        # csv_relation = os.path.join(NEO4j_DIR, f"data_relation_r{test_n_rows}_c2.csv")
        # print(csv_node_relation)

        cypher_id_constraint = '''
        CREATE CONSTRAINT unique_the_id IF NOT EXISTS FOR (n:The_id) REQUIRE n.name IS UNIQUE
        '''

        cypher_csv_node_and_relation = f'''
        USING PERIODIC COMMIT 100000
        LOAD CSV WITH HEADERS FROM 'file:///{csv_node_relation}' AS row 
        MERGE (id:The_id {{ 姓名: COALESCE(row.name, "None"),
                            身分證字號: COALESCE(row.node_id, "None"),
                            地址: COALESCE(row.address1, "None"),
                            生日: COALESCE(row.date_of_birth, "None")}}) 
        MERGE (id_f:The_id {{姓名: row.father}}) 
        MERGE (id_m:The_id {{姓名: row.mother}}) 
        MERGE (id_s:The_id {{姓名: row.spouse}}) 
        MERGE (id_f)-[:is_father{{關係: "父子"}}]->(id)
        MERGE (id_m)-[:is_mother{{關係: "母子"}}]->(id)
        MERGE (id_s)-[:is_spouse{{關係: "配偶"}}]->(id)
        '''

        csv_sibling_relation = os.path.join(NEO4j_DIR, f"data_link_sibling_ocr.csv")

        cypher_add_rel_sibling = f'''
            MATCH
            (id:The_id {{身分證字號: 'A122774626'}}),
            (id_s:The_id {{身分證字號: 'A121122913'}})
            CREATE (id)-[:is_sibling{{關係: "兄弟姊妹"}}]->(id_s)
        '''

        cypher_csv_rel_sibling = f'''
            USING PERIODIC COMMIT 100
            LOAD CSV WITH HEADERS FROM 'file:///{csv_sibling_relation}' AS row 
            MATCH
            (id:The_id {{身分證字號: COALESCE(row.from, "None") }}),
            (id_to:The_id {{身分證字號: COALESCE(row.to, "None")}})            
            CREATE (id)-[:is_sibling{{關係: "兄弟姊妹"}}]->(id_to)

        '''

        csv_company_relation = os.path.join(NEO4j_DIR, f"data_sub_l2_link_works.csv")

        cypher_company = f'''
        USING PERIODIC COMMIT 10
        LOAD CSV WITH HEADERS FROM 'file:///{csv_company_relation}' AS row 
        MERGE (company:The_company {{ 公司統編: COALESCE(row.to, "None")}})
        '''

        # cypher_rel_id_comp = f'''
        #     MATCH
        #     (id:The_id {{身分證字號: 'A122774626'}}),
        #     (company:The_company {{公司統編: '31218691'}})
        #     CREATE (id)-[:in_charge_of{{關係: "負責人"}}]->(company)
        # '''

        cypher_csv_rel_id_comp = f'''
            USING PERIODIC COMMIT 100
            LOAD CSV WITH HEADERS FROM 'file:///{csv_company_relation}' AS row 
            MATCH
            (id:The_id {{身分證字號: COALESCE(row.from, "None") }}),
            (company:The_company {{公司統編: COALESCE(row.to, "None")}})
            CREATE (id)-[:in_charge_of{{關係: "負責人"}}]->(company)
        '''

        # csv_sibling_relation = os.path.join(NEO4j_DIR, f"data_sub_link_sibling_ocr.csv")

        # cypher_constraint_sib = '''
            # CREATE CONSTRAINT unique_id_sibling IF NOT EXISTS FOR (n:The_id_sibling) REQUIRE n.to IS UNIQUE
        # '''

        # cypher_sibling_rel = f'''
        # USING PERIODIC COMMIT 10000
        # LOAD CSV WITH HEADERS FROM 'file:///{csv_sibling_relation}' AS row 
        # MERGE (id_from:The_id_sibling {{身分證字號: COALESCE(row.from, "None")}}) 
        # MERGE (id_to:The_id_sibling {{身分證字號: COALESCE(row.to, "None")}}) 
        # MERGE (id_f)-[:is_father{{關係: "父子"}}]->(id)
        # '''


        # cypher_csv_relation = f'''
        # USING PERIODIC COMMIT 100000
        # LOAD CSV WITH HEADERS FROM 'file:///{csv_relation}' AS row
        # MERGE (id_from:The_id {{id: row.from}})
        # MERGE (id_to:The_id {{id: row.to}})
        # MERGE (id_from)-[:Relation]->(id_to)
        # '''

        query_list = [
            cypher_clean,
            # cypher_path,
            # cypher_info,
            cypher_id_constraint,
            cypher_csv_node_and_relation,
            # cypher_create,
            cypher_add_rel_sibling,
            cypher_csv_rel_sibling,
            cypher_company,
            # cypher_rel_id_comp,
            cypher_csv_rel_id_comp,
            # cypher_constraint_sib,
            # cypher_sibling_rel,
        ]

        # a, b = self.cypher_info_company()

        query_memory = [cypher_usage]

        self.logger.info("in go_test")
        with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
            for i, curr_query in enumerate(query_list, start=1):
                print(i, driver.query(curr_query))

            for i, curr_query in enumerate(query_memory, start=100):
                print(i, driver.execute_t2(curr_query))
        print("Done")


if __name__ == "__main__":
    print(f"{'-'*20}")
    a_script = go_main(logger)
    a_script.go_read_ocr_data(read_size=2000)

    # for i in range(1, 7):
    #     print(f"{'*'*20} - number of hop: {i}")
    #     a_script.go_test_read_data(n_of_hops = i, repeated_time = 10)
    # print(f"{'-'*20}")

# %%
# %%
