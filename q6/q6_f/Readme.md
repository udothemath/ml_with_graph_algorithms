## 測試不同的num_workers對速度的影響
### 說明
總共有兩個檔案，第一個是無graph的dataloader，第二個是我們常用的graphsage for cora

可以直接執行，會印出在不同的num_workers的情況下需要多少時間
- 會發現num_workers變多的情況下時間會變慢
- 是一個沒有人解出來的[問題](https://github.com/pytorch/pytorch/issues/12831)
參照上面那個連結裡面提供的做法，當num_workers > 0的時候，在dataloader的arguments裡加入persistent_workers=True可以稍微加速它
- 但還是沒有設num_workers=0快

> 要注意的是num_workers需要在 if name == 'main':的情況下執行


### 參考資料
- [num_workers issue discussion](https://github.com/pytorch/pytorch/issues/12831)
- [introduction of advanced mini-batching in pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)