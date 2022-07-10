# CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
# TORCH versions: 1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.8.0
CUDA=cpu     ##cu101
TORCH=1.10.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric