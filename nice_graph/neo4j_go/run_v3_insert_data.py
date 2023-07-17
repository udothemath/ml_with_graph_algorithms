# %%
import os
from dataclasses import dataclass
from src.utils import create_csv_file, logger_decorator
from src.neo4j_conn import Neo4jConnection
from setting import NEO4J_PASSWORD, NEO4J_USER
from src.cypher_code import (cypher_clean, cypher_conf)

PATH_BOLT = "bolt://localhost:7687"
MAIN_PATH = "/Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go"
DATA_PATH = os.path.join(MAIN_PATH, 'data')
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class FileInfo:
    file_prefix: str
    num_rows: int
    type_size: int
    file_path: str

    @property
    def filename_path(self):
        return os.path.join(self.file_path, f'{self.file_prefix}_size{self.num_rows}.csv')

def check_file(file_info:FileInfo) -> None:
    filename_path  =  file_info.filename_path
    print(f"file: {filename_path}")
    if os.path.isfile(filename_path):
        print("u already have csv file. Do nothing...")
    else:
        print("u don't have csv file. Create in progress...")
        create_csv_file(a_csv)

@logger_decorator
def main(the_file):

    the_cypher = f'''
        LOAD CSV WITH HEADERS FROM 'file:///{the_file}' AS row
        RETURN count(row);
    '''

    with Neo4jConnection(uri=PATH_BOLT, user=NEO4J_USER, pwd=NEO4J_PASSWORD) as driver:
        print(driver.query(cypher_conf))
        print(driver.query(the_cypher))
        # print(driver.query(cypher_html_csv))


if __name__ == "__main__":
    print("run main")
    # step 1:
    a_csv = FileInfo('sample', 12, 5, DATA_PATH)
    a_path = a_csv.filename_path
    check_file(a_csv)
    main(a_path)


    print("done in main")
# %%

