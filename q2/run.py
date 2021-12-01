# %%
import os
import pandas as pd
from IPython.display import display
import torch
# from sentence_transformers import SentenceTransformer

from pathlib import Path
from torch_geometric.data import download_url, extract_zip

QUESTION = 'q2'
PATH = f"/Users/pro/Documents/ml_with_graph_algorithms/{QUESTION}"
PATH_DATA = f"{PATH}/data"

os.chdir(PATH)
print(f"Current directory: {os.getcwd()}")

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

proj_name = url.rsplit('/', 1)[1].rsplit('.', 1)[0]
path_for_filename = f"{PATH_DATA}/{proj_name}"
Path(f"{path_for_filename}").mkdir(parents=True, exist_ok=True)
movie_path = f'{path_for_filename}/movies.csv'
rating_path = f'{path_for_filename}/ratings.csv'

# print(movie_path, rating_path)

# %%
def download_dataset(url, path_data):
    '''
    download dataset and put it in path_data
    '''    
    Path(f"{path_data}").mkdir(parents=True, exist_ok=True)
    extract_zip(download_url(url, path_data), path_data)
    print("Exit from downloading files")

# %%
def download_dataset(url, path_data):
    '''
    download dataset and put it in path_data
    '''    
    Path(f"{path_data}").mkdir(parents=True, exist_ok=True)
    extract_zip(download_url(url, path_data), path_data)
    print("Exit from downloading files")

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

class GenresEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

def check_indicator(input_string):
    print (f"{'-'*10} {input_string} {'-'*10}")

def check_first_few_items(the_dict, n):
    _dict = {k: the_dict[k] for k in list(the_dict)[:n]}
    print(_dict)

# if __name__ == "__main__":
#   pass
download_dataset(url, path_data=PATH_DATA)

# %%
# check_indicator("Check few data rows")
# df_movie = pd.read_csv(movie_path)
# df_rating = pd.read_csv(rating_path)

# print(f"Movie info size: {df_movie.shape}")
# print(f"User and movie rating size: {df_rating.shape}")

# display(df_movie.head())
# display(df_rating.head())

# check_indicator("Generate node index")
# # _, movie_mapping = load_node_csv(movie_path, index_col='movieId')       
# _, user_mapping = load_node_csv(rating_path, index_col='userId')
# ## Note: There is no additional feature information for users present in this dataset. As such, we do not define any encoders

# movie_x, movie_mapping = load_node_csv(
#     movie_path, index_col='movieId', encoders={
#         'title': SequenceEncoder(),
#         'genres': GenresEncoder()
#     })    

# check_indicator("Check node index")
# print(f"Total number of movie: {len(movie_mapping)}")
# print(f"Total number of user: {len(user_mapping)}")

# check_first_few_items(movie_mapping, 5)
# check_first_few_items(user_mapping, 5)

# # %%
# check_indicator("Check relation attribute")
# edge_index, edge_label = load_edge_csv(
#     rating_path,
#     src_index_col='userId',
#     src_mapping=user_mapping,
#     dst_index_col='movieId',
#     dst_mapping=movie_mapping,
#     encoders={'rating': IdentityEncoder(dtype=torch.long)},
# )

# # %%
# check_indicator("Check edge index")
# print(f"Size of edge by edge_label: {len(edge_label)}")
# print(f"Few example of edge_label: {edge_label[:5]}")
# # %%
# check_indicator("Check movie features")
# print(f"Shape of movie_x: {movie_x.shape}")
# print(f"Example of movie_x: {movie_x[:5]}")

# # %% 
# check_indicator("Run example")

# x = torch.arange(6)

# print(x.view(3, -1))
# print(x.view(1, -1))
# print(x.view(-1, 1))
# # %%
# x = torch.arange(12)
# print(x.view(2, 3, -1))
# print(x.view(3, 2, -1))

# # %%
# class Test():
#     def __init__(self):
#         print(f"U r in __init__")

#     def __call__(self):
#         print(f"U r in __call__")
#         return 'Hello'

# print("Test one")
# test1 = Test()
# print("Test two")
# test2 = Test()()
# # %%
