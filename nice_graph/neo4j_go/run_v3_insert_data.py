# %%
import os
from dataclasses import dataclass
from src.utils import create_csv_file

MAIN_PATH = "/Users/pro/Documents/ml_with_graph_algorithms/nice_graph/neo4j_go"
DATA_PATH = os.path.join(MAIN_PATH, 'data')

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


if __name__ == "__main__":
    # step 1:
    a_csv = FileInfo('sample', 12, 5, DATA_PATH)
    check_file(a_csv)


    print("done in main")
# %%

