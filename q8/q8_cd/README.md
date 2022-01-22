# T-brain

### 問題定義(Problem Formulation)

對於每個不同的客戶，使用過往的消費紀錄(對於不同消費類別的消費金額、消費次數、海內外消費…等)、客戶資訊(婚姻狀態、學歷、性別、年紀…等)來預測下個月份對於不同類別的消費金額高低，預測僅需考慮消費金額Top3的類別，從高至低排序。

![image](https://user-images.githubusercontent.com/66724009/150493789-1dd00063-c48c-4981-a769-409fad57045f.png)

* 共500000個不同客戶，資料筆數高達33M

### 建圖方式
![image](https://user-images.githubusercontent.com/66724009/150494142-faa5e499-19f4-414a-89db-e929e9ca6f4d.png)

### 特徵因子(X)建置方式
![image](https://user-images.githubusercontent.com/66724009/150494321-639c2431-934f-4c9a-aa7f-d69211a668ea.png)

### 特徵因子(Y)建置方式
![image](https://user-images.githubusercontent.com/66724009/150494392-73a431ba-bd6f-4aaa-bac0-40f520e40489.png)

### 流程

1. 建圖找顧客embedding   
2. 把顧客embedding跟原本feature concat  
3. 把2.的feature餵進LightGBM ranker訓練，預測每個顧客在下個月的消費金額前三大的類別

### 好用指令分享

watch –n 秒數 –d nvidia-smi  
Ex: watch –n 1 –d nvidia-smi (每秒刷新)

### 問題討論 & 建議作法

[問題] (特徵因子(X)建置方式) 目前作法無法區分該node為顧客/消費類別 
[建議] 各自產生對應的featrure來描述, 使得總維度為(19+16)  

[問題] (特徵因子(Y)建置方式) 顧客/消費類別的feature的scale會有不小差距 
[建議] 需要做normalization

[問題] 消費類別使用第16類, 會和顧客類別的資訊(0-15)混淆 
[建議] 使用其他易區分的數字(0 or -1等)

### 其他問題
Q: NVIDIA-SMI中顯示Utility為0  
A: (昕靜) 目前想法為實際上有進到GPU, 但時間很短

Q: Dataloader freezes when num_worker > 0  
A: 尚未發現root cause, 但將num_worker設為0即可解決, 且速度差異不大





