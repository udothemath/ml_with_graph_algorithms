"""
嘗試用state-of-the-art nearest neighbor algorithm 
建立edge list。看看速度回不會比較快~~ A: 會! 只要7分鐘就建立好一個edge lists。

pip install scann==1.2.0 # work with tensorflow 2.4.0
"""
import numpy as np 
import scann
import pandas as pd
import time
avm_table = pd.read_csv('/home/jovyan/if-graph-ml/esb21375/data/avm_node_table_small.csv')
x_y_table = avm_table[['xx:Float', 'yy:Float']]
dataset = x_y_table.to_numpy()
queries = x_y_table.iloc[:100000].to_numpy()
print('size of dataset', dataset.shape)

searcher = scann.scann_ops_pybind.builder(dataset, 10, "squared_l2").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()

start = time.time()
neighbors, distances = searcher.search_batched(queries)
end = time.time()
print('neighbors', neighbors.shape)
print('distances', distances.shape)
print('query time:', end-start)
print('estimated edge building time:', (end-start) * dataset.shape[0] / queries.shape[0]) 
# Need about 4min to create edges if no pallalism considered. 
print('neighbors 0:', neighbors[0, :])
print('distances 0:', distances[0, :])

# Checking if the Nearest neighbors is correct and if the distance is correct

def get_distance(x1, y1, x2, y2):
    return (x1- x2) ** 2 + (y1-y2) ** 2

print('first neighbor')
x1 = dataset[0, 0]
y1 = dataset[0, 1]
x2 = dataset[1582, 0]
y2 = dataset[1582, 1]
dis = get_distance(x1, y1, x2, y2)
print('re-calculated dis:', dis)
print('second neighbor')
x1 = dataset[0, 0]
y1 = dataset[0, 1]
x2 = dataset[2607, 0]
y2 = dataset[2607, 1]
dis = get_distance(x1, y1, x2, y2)
print('re-calculated dis:', dis)