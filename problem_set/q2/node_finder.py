"""
TODO:
- [X] 找出w103表中的node (1. user. 2. fund_id. 3. transaction_id)
    - [X] 找出各種欄位的unique數量 (要多於500)
    - [X] 找出各欄位資料型態 
"""
# %% Get column names
from sql_tools import ETLBase
etl = ETLBase()
sql = '''
SELECT * 
FROM mlaas_rawdata.witwo103_hist
WHERE txn_dt>='2021-10-01'
'''
table = etl.select_table('rawdata', sql)
columns = table.columns.tolist()
print(columns)


# %% Find columns with # unique >= 500 and type == string
for col in columns: 
    sql = f'''
        SELECT pg_typeof({col}) 
        FROM mlaas_rawdata.witwo103_hist
        LIMIT 1
        '''
    col_type = etl.select_table('rawdata', sql)[
        'pg_typeof'].values[0]
    if col_type != 'numeric' and col_type != 'date': 
        sql = f'''
            SELECT count(DISTINCT {col}) 
            FROM mlaas_rawdata.witwo103_hist
            WHERE txn_dt>='2021-10-01'
            '''
        count = etl.select_table('rawdata', sql)['count'].values[0]
        if count > 500:
            print(col, '\t', count)
"""
可能之node: 

交易序號        data_seq 	    940679
顧客暗碼        cust_id 	    117956
信託編號        purchase_no 	397860
產品編號        wm_prod_code 	4842
轉換產品編號    trf_prod_id 	1239
定期定額次數    deduct_cnt 	    1199
服務理專員編    fc_emp_no 	    2004
交易理專員編    txn_fc_emp_no 	1796

進一步排除掉: 
- 定期定額次數 (可視為信託編號底下之特徵) 
- 交易理專員編 (相對服務理專和顧客接觸較少) 

適合之node:

交易序號        data_seq 	    A
顧客暗碼        cust_id 	    B
信託編號        purchase_no 	C
產品編號        wm_prod_code 	D
轉換產品編號    trf_prod_id 	D
服務理專員編    fc_emp_no 	    E

"""

# %%
