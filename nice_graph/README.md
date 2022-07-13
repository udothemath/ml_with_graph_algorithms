## 題目發想  
|主題|分享人|說明|資料源|備註|  
|--|--|--|--|--|  
|電話撥打以及應用程式使用情況的相關性關聯分析|甫璋|摘要說明:顧客電話通訊以及APP應用程式的使用行為及關聯分析。預期產出:</br>1. 使用圖篩選語法分類客群</br>2. 使用圖關聯模型定義客群|whoscall提供|討論中的POC合作案| 
|Graph-based recommendation system|崇爾|預期產出:<br> 1. 建置圖資料庫並存放歷史互動資料 <br>2. 建置推薦模型<br>3. 使用Neo4j視覺化人-物關係| |References:https://towardsdatascience.com/exploring-practical-recommendation-engines-in-neo4j-ff09fe767782 <br>![alt text](https://dist.neo4j.com/wp-content/uploads/20220104122129/GDS_1-1024x369.png)
|程式碼架構圖查詢系統|奕勳|摘要說明:程式碼架構圖查詢系統。預期產出:</br>1. 使用pycograph把程式repo放到redisgraph或neo4j以後，用ipycytograph進行視覺化呈現 2. Redisgraph + Neo4j Aicloud Image 3. 提供快速將Redisgraph 資料轉入Neo4j 的方式</br>|Azure Devops上面的repo|方便大家更容易理解程式碼，順便研發可在jupyterlab上使用的redisgraph搜尋接口| 



## 目標(As of 2022/6/30) 
在 python 開發環境(vs code)使用 cypher 語法執行 neo4j，結果呈現於瀏覽器上，並可於瀏覽器頁面上執行 neo4j 的互動式操作。

## 開發看板：
[Link]([https://github.com/udothemath/ml_with_graph_algorithms/projects?type=beta](https://dist.neo4j.com/wp-content/uploads/20220104122129/GDS_1-1024x369.png))

## 開發內容：
- 將資料儲存於圖資料庫中
- python 開發環境(vs code)執行 neo4j
- 視覺化結果呈現在瀏覽器上
- 當圖已經建好時，產生圖結構生成的特徵(graph feature enhancement)
- 研究圖暫存檔格式(效能優化)
- cipher 語法介紹
    - 參考網站:https://ithelp.ithome.com.tw/users/20130217/articles?page=2

## 附註
請同步新增操作說明以及相關參考資料
