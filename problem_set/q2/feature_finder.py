"""
找出各種節點適合之屬性欄位
"""
# %% get column names 
from sql_tools import ETLBase
etl = ETLBase()
sql = '''
SELECT * 
FROM mlaas_rawdata.witwo103
'''
table = etl.select_table('rawdata', sql)
columns = table.columns.tolist()
print(columns)
# %% get possible feature columns of each node 
node_names = [
    'data_seq',
    'cust_id',
    'purchase_no',
    'wm_prod_code',
    'trf_prod_id',
    'fc_emp_no'
]

def print_possible_features(node_name):
    for col in columns:
        if not (col in node_names):
            sql = f'''
                SELECT *
                FROM (
                    SELECT {node_name}, count(DISTINCT {col})
                    FROM mlaas_rawdata.witwo103
                    GROUP BY {node_name}
                )
                WHERE count > 1
                '''
            table = etl.select_table('rawdata', sql)
            if len(table) == 0:
                print('Possible Feature:\t', col)


#%% Possible Feature of data_seq
print_possible_features('data_seq')
"""
All is possible
""" 
# %% Possible Feature of cust_id
print('check feature of cust_id')
print_possible_features('cust_id')
"""
business_type 業務別 (DBU 或 OBU) 
"""
# %% Possible Feature of purchase_no
print('check feature of purchase_no')
print_possible_features('purchase_no')
"""
invest_type_code 信託性質(投資方式)
business_type  業務別(DBU 或 OBU)
"""
# %% Possible Feature of wm_prod_code
print('check feature of wm_prod_code')
print_possible_features('wm_prod_code')
"""
prod_ccy 計價幣別
"""
# %% Possible Feature of trf_prod_id
print('check feature of trf_prod_id')
print_possible_features('trf_prod_id')
"""
"""
# %% Possible Feature of fc_emp_no
print('check feature of fc_emp_no')
print_possible_features('fc_emp_no')
"""
"""
# %% Check content:
for col in [
    'fc_emp_no', 
    'deduct_nhi_supp_val',
    'refund_nhi_supp_val',
    'refund_nhi_supp_credit_acct_dt',
    'project_auto_txn_ind',
    'split_tax',
    'selling_cost'
    ]: 
    etl = ETLBase()
    sql = f'''
    SELECT DISTINCT {col}
    FROM mlaas_rawdata.witwo103
    '''
    table = etl.select_table('rawdata', sql)
    print(col, len(table))

"""
可排除之特徵欄位:
fm_emp_no (空值)
deduct_nhi_supp_val 
refund_nhi_supp_val 
refund_nhi_supp_credit_acct_dt 
split_tax 
selling_cost 
"""
