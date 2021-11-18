# %%
import os
import pandas as pd
from IPython.display import display
import torch
from pathlib import Path
from torch_geometric.data import download_url, extract_zip
from igraph import Graph, plot


PATH = "/Users/pro/Documents/ml_with_graph_algorithms/q1"
PATH_DATA = f"{PATH}/data"

os.chdir(PATH)
print(f"Current directory: {os.getcwd()}")

dict_data = {
    'citeseer': 'https://nrvis.com/download/data/labeled/citeseer.zip'}

def download_dataset(dict_dataset, dict_key, path_data):
    print(dict_dataset[dict_key])
    url = dict_dataset[dict_key]
    path_data_sub = f"{path_data}/"
    print(path_data_sub)
    
    Path(f"{path_data_sub}").mkdir(parents=True, exist_ok=True)
    extract_zip(download_url(url, path_data_sub), path_data_sub)

def using_igraph(if_plot=False):

    n_vertices = 3264

    # Create graph
    g = Graph()

    # Add vertices
    g.add_vertices(n_vertices)

    edges = []
    weights = []

    with open(f"{PATH_DATA}/citeseer.edges", "r") as edges_file:
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
    # for e in g.es()[:5]:
    #     print(e.tuple)

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

download_dataset = False
if download_dataset == True:
    download_dataset(dict_dataset=dict_data, dict_key='citeseer', path_data=PATH_DATA)
using_igraph(if_plot=False)
print("done")