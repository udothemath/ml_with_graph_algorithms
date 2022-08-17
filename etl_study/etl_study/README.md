# Setup 

pip install -r requirements.txt

# Steps: 

1. 產生join的profiling資料

```linux
python -m tools.gen_join --n-samples 10000000 --n-int-features 3 --n-float-features 4 --join-key-common-ratio 0.9 --output-path ./data/raw/synthetic/join/
```

2. 產生general的profiling資料



3. 執行profiling: 

```linux
python -m tools.profile_main --query '[join] inner on id_l' --input-file ./data/raw/synthetic/join/join_1e+07_lhs.parquet --mode pandas --n-profiles 5 --to-benchmark True
```

4. 檢查結果: 

結果會append於`berks.csv`


