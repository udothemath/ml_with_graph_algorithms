# 檔案說明:

## 預審用顧客datamart產製

* prepare_data/ 
    - datamart中各個欄位的產製程式 (含母體顧客id產製程式) 
    
* prepare_datamart_etl.py
    - datamart整合程式，將prepare_data中各個ETL物件整合於此

## 預審邏輯: 

* cust_apply.py 
    - 預審邏輯套用到顧客datamart之ETL程式
    - 產品: 循環型、一次型
    
* cust_logic.py
    - 預審邏輯運算函數 
    - 產品: 循環型、一次型
    
## 各產品計算結果合併: 

* final_etl.py 
    - 整合各產品結果的ETL程式 
    
    
