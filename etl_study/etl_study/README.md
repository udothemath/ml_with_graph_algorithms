# General Profile Framework 

[API Documentation](http://localhost)

## Environment Setup

```
pip install -r requirements.txt
```

## Quick Start 

1. 產生給join使用的資料

以下指令會觸發data generator生成sample數量為`1e7`的合成資料集(table on left-hand side)，同時會產生三張大小不同的資料(table on right-hand side)，用於執行後續的`join`操作：
```
python -m tools.gen_join --n-samples 10000000 --n-int-features 3 --n-float-features 4 --join-key-common-ratio 0.9 --output-path ./data/raw/synthetic/join/
```

2. 產生general使用的資料

以下指令會觸發data generator生成sample數量為`1e7`的合成資料集，供給常用的general APIs (*e.g.*, `groupby`, `rolling`)做profiling：
```
python -m tools.gen_general --n-samples 10000000 --n-str-ids 3 --n-int-ids 3 --n-clusts-per-id 100 --n-int-features 2 --n-float-features 2 --output-path ./data/raw/synthetic/general/
```

3. 執行profiling:


