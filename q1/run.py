# %%
import os
import pandas as pd
from IPython.display import display
import torch
from sentence_transformers import SentenceTransformer

PATH = "/Users/pro/Documents/ml_with_graph_algorithms/q1"
PATH_DATA = f"{PATH}/data"

os.chdir(PATH)
print(f"Current directory: {os.getcwd()}")

movie_path = f'{PATH_DATA}/ml-latest-small/movies.csv'
rating_path = f'{PATH_DATA}/ml-latest-small/ratings.csv'

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

def using_pyg(if_download=True):
    if if_download:
        from torch_geometric.data import download_url, extract_zip
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        extract_zip(download_url(url, f"{PATH_DATA}"), f"{PATH_DATA}")

    # movie_x, movie_mapping = load_node_csv(
    #     movie_path, index_col='movieId', encoders={
    #         'title': SequenceEncoder(),
    #         'genres': GenresEncoder()
    #     })    

    print("Exit in using_pyg")

def using_igraph(if_plot=False):
    from igraph import Graph, plot

    n_vertices = 3264

    # Create graph
    g = Graph()

    # Add vertices
    g.add_vertices(n_vertices)

    edges = []
    weights = []

    with open(f"{PATH}/data/citeseer/citeseer_mini.edges", "r") as edges_file:
        line = edges_file.readline()
        while line != "":
            
            strings = line.rstrip().split(",")
            
            # Add edge to edge list
            edges.append(((int(strings[0])-1), (int(strings[1])-1)))
            
            # Add weight to weight list
            weights.append(float(strings[2]))
               
            line = edges_file.readline()
            # print(line)
    # Add edges to the graph
    g.add_edges(edges)

    # Add weights to edges in the graph
    g.es['weight'] = weights

    print(g.vs())
    print(g.es())
    # print(g.es().__doc__)
    for e in g.es():
        print(e.tuple)

    if if_plot:
        out_fig_name = "graph.eps"

        visual_style = {}

        # Define colors used for outdegree visualization
        colours = ['#fecc5c', '#a31a1c']

        # Set bbox and margin
        visual_style["bbox"] = (3000,3000)
        visual_style["margin"] = 17

        # Set vertex colours
        visual_style["vertex_color"] = 'grey'

        # Set vertex size
        visual_style["vertex_size"] = 20

        # Set vertex lable size
        visual_style["vertex_label_size"] = 8

        # Don't curve the edges
        visual_style["edge_curved"] = False

        # Set the layout
        my_layout = g.layout_fruchterman_reingold()
        visual_style["layout"] = my_layout

        # Plot the graph
        plot(g, out_fig_name, **visual_style)
    print("exit from using_igraph")

# if __name__ == "__main__":
#     print("hello")
#     # using_igraph()
#     using_pyg(if_download=False)
#     using_pyg(if_download=False)

_, movie_mapping = load_node_csv(movie_path, index_col='movieId')       
_, user_mapping = load_node_csv(rating_path, index_col='userId')

# %%
df_movie = pd.read_csv(movie_path)
df_rating = pd.read_csv(rating_path)

display(df_movie.head())
display(df_rating.head())

print(df_movie.shape)
print(df_rating.shape)

# %%
edge_index, edge_label = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

# %%
print(edge_index)
# print(edge_label)
# print(len(edge_label))
# %%

def check_first_few_items(the_dict, n):
    _dict = {k: the_dict[k] for k in list(the_dict)[:n]}
    print(_dict)

check_first_few_items(movie_mapping, 5)
check_first_few_items(user_mapping, 5)

print(f"Number of movie: {len(movie_mapping)}")
print(f"Number of user: {len(user_mapping)}")


# %%

# %%
