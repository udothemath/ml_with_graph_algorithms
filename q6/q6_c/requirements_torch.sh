pip install torch==1.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.20.2

CUDA=cu102
TORCH=1.9.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

pip install --upgrade pip
pip install ipykernel