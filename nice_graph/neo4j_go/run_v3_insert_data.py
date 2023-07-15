# %%
import os
from src.utils import create_csv_file

MAIN_PATH = "/Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go"
DATA_PATH = os.path.join(MAIN_PATH, 'data')
print(DATA_PATH)


if __name__ == "__main__":
    create_csv_file('example', 123, type_size=5, file_path=DATA_PATH)
    print("done")
# %%
