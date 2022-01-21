# T-brain

### 問題定義(Problem Formulation)

對於每個不同的客戶，使用過往的消費紀錄(對於不同消費類別的消費金額、消費次數、海內外消費…等)、客戶資訊(婚姻狀態、學歷、性別、年紀…等)來預測下個月份對於不同類別的消費金額高低，預測僅需考慮消費金額Top3的類別，從高至低排序。

![image](https://user-images.githubusercontent.com/66724009/150493789-1dd00063-c48c-4981-a769-409fad57045f.png)

* 共500000個不同客戶，資料筆數高達33M

### 建圖方式
![image](https://user-images.githubusercontent.com/66724009/150494142-faa5e499-19f4-414a-89db-e929e9ca6f4d.png)

* 特徵因子(X)建置方式
![image](https://user-images.githubusercontent.com/66724009/150494321-639c2431-934f-4c9a-aa7f-d69211a668ea.png)

* 特徵因子(Y)建置方式
![image](https://user-images.githubusercontent.com/66724009/150494392-73a431ba-bd6f-4aaa-bac0-40f520e40489.png)

### 好用指令分享

watch –n 秒數 –d nvidia-smi
Ex: watch –n 1 –d nvidia-smi (每秒刷新)

