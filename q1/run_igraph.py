# %%
from igraph import *

n_vertices = 3264

# Create graph
g = Graph()

# Add vertices
g.add_vertices(n_vertices)

edges = []
weights = []

import os

PATH = "/Users/pro/Documents/ml_with_graph_algorithms/q1"

os.chdir(PATH)
print(os.getcwd())

with open(f"{PATH}/data/citeseer/citeseer.edges", "r") as edges_file:
    line = edges_file.readline()
    while line != "":
        
        strings = line.rstrip().split(",")
        
        # Add edge to edge list
        edges.append(((int(strings[0])-1), (int(strings[1])-1)))
        
        # Add weight to weight list
        weights.append(float(strings[2]))
        
        
        line = edges_file.readline()

# Add edges to the graph
g.add_edges(edges)

# Add weights to edges in the graph
g.es['weight'] = weights

print("Done")
# %%
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
print("Exit ")
# %%
