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


# %%
class gen_csv_file:
    def __init__(self, id_prefix='ab', num_rows=10, path=NEO4j_DIR, logger=None):
        """
            generate sample csv file
        """
        self.__id_prefix = id_prefix
        self.__num_rows = num_rows
        self.__path = path

        if logger is None:
            logging.basicConfig(
                filename=f'{NEO4j_DIR}/logs1.log',
                filemode='a',
                format='%(asctime)s: %(name)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO
            )

            logger = logging.getLogger(__name__)
            logger.info(
                "===============================================================================")
            logger.info("Starting Application Logging")

            # logger.info( "Logging Config Imported in Second Script" )

        self.logger = logger

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
        col_name = ['id', 'value']
        id_list = [f"{self.__id_prefix}_{i}" for i in range(self.__num_rows)]
        value_list = [randrange(0, 200) for i in range(self.__num_rows)]
        rows = zip(id_list, value_list)

        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        print(
            f"Ur file with rows:{self.__num_rows} and cols:{num_cols} is ready.")

    @wrap_log
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

        self.logger.info(
            f"Ur file of relation with rows:{self.__num_rows} is ready.")

    @wrap_log
    def gen_relation_random(self, num_cols=2, repeated_rel=1) -> None:
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
        from_id_all = []
        to_id_all = []
        for _ in range(repeated_rel):
            from_id_all.extend(
                f"{self.__id_prefix}_{i}" for i in range(self.__num_rows))
            to_id_all.extend(
                f"{self.__id_prefix}_{randrange(self.__num_rows-1)}" for i in range(self.__num_rows))
        rows = zip(from_id_all, to_id_all)
        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        self.logger.info(
            f"Ur file of relation with rows:{self.__num_rows} is ready.")

    @wrap_log
    def gen_relation_random_with_prop(self, num_cols=2, repeated_rel=1) -> None:
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
        col_name = ['from', 'to', 'from_value', 'to_value']
        from_id_all = []
        to_id_all = []
        from_value_all = []
        to_value_all = []
        for _ in range(repeated_rel):
            from_id_all.extend(
                f"{self.__id_prefix}_{i}" for i in range(self.__num_rows))
            to_id_all.extend(
                f"{self.__id_prefix}_{randrange(self.__num_rows-1)}" for i in range(self.__num_rows))
            from_value_all.extend(randrange(0, 200)
                                  for i in range(self.__num_rows))
            to_value_all.extend(randrange(0, 200)
                                for i in range(self.__num_rows))
        rows = zip(from_id_all, to_id_all, from_value_all, to_value_all)
        # Open a new CSV file for writing
        with open(file_with_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(col_name)

            # for word in the_list:
            for row in rows:
                csvwriter.writerow(row)

        self.logger.info(
            f"Ur file of relation with rows:{self.__num_rows} is ready.")


# %%
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


query_list = [
    'match (n) return count(n)',
    'match (n) return count(n)'
]


def query_on_graph(graph_cypher_query_list: str):
    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        for i, j in enumerate(graph_cypher_query_list, 1):
            print(i, driver.query(j))
            logger.info(f"cyphyer query {i:4d}: {j}")
            print("--- Done ---")


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

    @wrap_log
    def go_test_load_data(self, test_n_rows=1000):
        print("number pf rows:", test_n_rows)
        a = gen_csv_file(id_prefix='abc', num_rows=test_n_rows)
        a.gen_node()
        a.gen_relation_random(repeated_rel=5)
        csv_node = os.path.join(NEO4j_DIR, f"data_node_r{test_n_rows}_c2.csv")
        csv_relation = os.path.join(
            NEO4j_DIR, f"data_relation_r{test_n_rows}_c2.csv")

        cypher_id_constraint = '''
        CREATE CONSTRAINT unique_the_id IF NOT EXISTS FOR (n:The_id) REQUIRE n.id IS UNIQUE
        '''

        cypher_csv_node = f'''
        USING PERIODIC COMMIT 100000
        LOAD CSV WITH HEADERS FROM 'file:///{csv_node}' AS row 
        MERGE (id:The_id {{id: row.id, value: row.value}}) 

        '''

        cypher_csv_relation = f'''
        USING PERIODIC COMMIT 100000
        LOAD CSV WITH HEADERS FROM 'file:///{csv_relation}' AS row 
        MERGE (id_from:The_id {{id: row.from}}) 
        MERGE (id_to:The_id {{id: row.to}}) 
        MERGE (id_from)-[:Relation]->(id_to)
        '''

        self.logger.info("in go_test")
        with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
            print(1, driver.query(cypher_clean))
            # # print(2, driver.query(cypher_path))
            # # print(3, driver.query(cypher_info))
            print(3, driver.query(cypher_id_constraint))
            print(4, driver.query(cypher_csv_node))
            print(5, driver.query(cypher_csv_relation))
            # print(7, driver.query(cypher_f2))
            print(8, driver.execute_t2(cypher_usage, the_style='df'))
            print(9, driver.execute_t2(cypher_usage,
                  the_style='abc')[0]['usedMemoryMB'])
            # print(9, driver.execute(the_piece))
            # print(10, driver.execute_t2(the_piece))

        print("Done")


if __name__ == "__main__":
    print(f"{'-'*20}")
    a_script = go_main(logger)
    # a_script.go_test_load_data(test_n_rows = 1_000_000)

    # for i in range(1, 7):
    #     print(f"{'*'*20} - number of hop: {i}")
    #     a_script.go_test_read_data(n_of_hops = i, repeated_time = 10)
    # print(f"{'-'*20}")

# %%
