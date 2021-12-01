## 目標
建構完整圖神經網絡開發流程(暫時不需要關注成效)

## 使用套件
[GraphGym from Stanford](https://github.com/snap-stanford/GraphGym)

## 安裝
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

## Torch版本
``` python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" ```

## 印出現在virtual environment的路徑
```
python -c 'import sys; print(sys.prefix)'
```