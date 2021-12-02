## 目標
建構完整圖神經網絡開發流程(暫時不需要關注成效)

## 套件
[GraphGym from Stanford](https://github.com/snap-stanford/GraphGym)
GraphGym is a platform for designing and evaluating Graph Neural Networks (GNN). GraphGym is proposed in Design Space for Graph Neural Networks, Jiaxuan You, Rex Ying, Jure Leskovec, NeurIPS 2020 Spotlight.

## 安裝
完整套件安裝流程請參考[GraphGym官方說明](https://github.com/snap-stanford/GraphGym)

```bash== 
$ conda create -n graphgym python=3.7
```
```bash== 
$ conda activate graphgym
```
```bash==
$ pip install torch==1.10.0 -f https://download.pytorch.org/whl/torch_stable.html 
``` 
```bash==
$ sh torch==1.10.0 -f https://download.pytorch.org/whl/torch_stable.html 
``` 
```bash==
$ sh requirements_torch.sh
``` 

## Torch版本
``` python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" ```

## 補充資訊
- 印出現在virtual environment的路徑
    ``` python -c 'import sys; print(sys.prefix)' ```
- [Pytorch What's the difference between define layer in __init__() and directly use in forward()?](https://stackoverflow.com/questions/50376463/pytorch-whats-the-difference-between-define-layer-in-init-and-directly-us)