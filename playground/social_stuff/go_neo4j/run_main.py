# %%
import time
import numpy as np
import pandas as pd
from IPython.display import display
from neo4j import GraphDatabase
from tqdm import tqdm
import os
import pyarrow
import logging
import yaml
from dataclasses import dataclass
from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (
    cypher_clean, cypher_conf, cypher_info, cypher_usage, cypher_node)
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


class RunNeo4j:
    def __init__(self, table_info=None, logger=None):
        """
        Run graph db: neo4j
        Args:
            table_info (dataclass): table info
            logger (str): Logging
        Returns:
            type: Description of the return value.
        Raises:
            ExceptionType: Description of when this exception is raised.
        """
        self.table_info = table_info
        self.logger = logger

    @property
    def get_constraint_dict(self):
        return self.table_info.constraint

    @wrap_log
    def run(self):
        """
            Read cypher from table_info
        """
        query_list = [value for key, value in self.get_constraint_dict.items()]
        print(query_list)
        self.logger.info("in go_test")
        with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
            print(0, driver.query(cypher_clean))
            for i, curr_query in enumerate(query_list, start=1):
                print(i, driver.query(curr_query))
            print(999, driver.query(cypher_node))
        print("Done with neo4j")


def create_csv_file():
    """
    Create csv file and save to desired directory
    Args:
        table_info (dataclass): table info
        logger (str): Logging
    Returns:
        None
    """
    file_1 = FileInfo(table_name="all_links", save_dir=FOLDER_DATA,
                      save_file_prefix="aaa", size_limit=30)
    file_with_path = file_1.get_path
    if os.path.isfile(file_with_path):
        print(f"u already have file: {file_with_path}")
    else:
        print(f"file doesn't exists. Creating file...")
        GenCSVfromDB(file_1, logger=logger).create_csv_from_df()
        print(f"file is ready. {file_with_path}")


def read_yaml():
    """
    Read yaml for neo4y syntax
    Args:
        yaml_filename(yaml): yaml file
    Returns:
        table_info(dataclass): the information of the table
    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    @dataclass
    class TableInfo:
        table: str
        col: dict
        constraint: str

    yaml_filename = 'table_neo4j.yaml'
    with open(yaml_filename, 'r') as file:
        data = yaml.safe_load(file)

    table_info = TableInfo(**data)
    return table_info


if __name__ == "__main__":
    start_time = time.time()
    print(f"{'-'*20}")
    create_csv_file()
    table_info = read_yaml()
    print(table_info)
    a_script = RunNeo4j(table_info, logger)
    a_script.run()
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print("---Done---")
