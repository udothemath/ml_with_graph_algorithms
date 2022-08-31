# General Profile Framework 

[API Documentation](http://localhost)

REFERENCE: https://hackmd.io/WKkHFLv5Tj-ieFXadQTM8A?both


## Environment Setup

```linux
pip install -r requirements.txt
```

## Quick Start 

1. 產生給join使用的資料

以下指令會觸發data generator生成sample數量為`1e7`的合成資料集(table on left-hand side)，同時會產生三張大小不同的資料(table on right-hand side)，用於執行後續的`join`操作：
```linux
python -m tools.gen_join --n-samples 10000000 --n-int-features 3 --n-float-features 4 --join-key-common-ratio 0.9 --output-path ./data/raw/synthetic/join/
```

2. 產生general使用的資料

以下指令會觸發data generator生成sample數量為`1e7`的合成資料集，供給常用的general APIs (*e.g.*, `groupby`, `rolling`)做profiling：
```linux
python -m tools.gen_general --n-samples 10000000 --n-str-ids 3 --n-int-ids 3 --n-clusts-per-id 100 --n-int-features 2 --n-float-features 2 --output-path ./data/raw/synthetic/general/
```

3. 執行profiling:
```linux
python -m tools.profile_main --query '[join] inner on id_l' --input-file ./data/raw/synthetic/join/join_1e07_lhs.parquet --mode pandas --n-profiles 5 --to-benchmark True
```

4. 檢查結果: 

結果會append於`berks.csv`



# Aicloud Fitting TODO: 

## [read_psql] rawdata tmp general_1e07

- [X] 要改connection建立的方法 -> DONE: "python -m tools.profile_main --query '[read_psql] feature wm_aiget dummy' --mode polars --n-profiles 5 --to-benchmark True"
- [ ] 要手動把.parquet檔放進DB

## xxx  mode = modin

- [ ] 要試試看記憶體size更大的情況

