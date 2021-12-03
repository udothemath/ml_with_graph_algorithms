"""
TODO:
- [X] 找出w103表中的node (1. user. 2. fund_id. 3. transaction_id)
    - [X] 找出各種欄位的unique數量 (要多於100)
- [X] 找出各種node底下的特徵 
    - [X] 把node對應到其他欄位，其他的欄位對於每個node要只有一個對應的value。
- [X] build mapping of 所有node
- [X] 為各個 node 的特徵 建立 encoder，並跑出結果
    - [X] OneHotEncoding
    - [X] LabelEncoding

NOTE:

Node: 

交易序號        data_seq 	    A -> edge
顧客暗碼        cust_id 	    B
信託編號        purchase_no 	C -> edge feature
產品編號        wm_prod_code 	D
轉換產品編號    trf_prod_id 	D'
服務理專員編    fc_emp_no 	    E -> edge feature

Edges (X):
B - C - A - D(D') 
        | - E 

Edges (X): 
B - C - D(D')
      |_E (edge feature)     

Edges (X): 
B - D(D')
| |_E (edge feature)    
|
|--C (node feature)

Edges (V): 
B - D(D')
  |_E (edge feature)    
  |
  |--C (edge feature)

Node Features: 
- B    - cust_id (business_type & 其他cust表) 



- D'/D - wm_prod_code (prod_ccy & 其他於witwo106表中) 

    參考基金e指選模型使用之欄位: 

    wm_prod_code 產品代碼
    mkt_rbot_ctg_ic 市場分類(IC)代碼
    prod_name 產品名稱
    counterparty_code 上手公司代碼
    counterparty_name 上手公司名稱
    prod_ccy 計價幣別
    prod_risk_code 產品風險收益等級(公會)
    esun_prod_risk_code 內部風險收益等級(玉山)
    prod_detail_type_code 境內外基金還是其他產品，如海外債、海外股票、ETF等等。?
    prod_detail2_type_code 境內外基金還是其他產品，如海外債、海外股票、ETF等等。?
    channel_web_ind 開放交易通路-個人網路銀行
    channel_mobile_ind 開放交易通路-行動銀行
    prod_attr_1_ind ?
    prod_attr_5_ind 精選與否
    fee_type_code 手續費收取類型 (e.g., A: 前收)

Edge Features: 
- A (本表中非B~E的特徵 + E/C/invest_type_code)

"""
# %%
import os
import pandas as pd
from IPython.display import display
import torch
from sql_tools import ETLBase
import numpy as np
print(f"Current directory: {os.getcwd()}")


# BEGIN: [Encoders] 
 
class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

class MultiLabelEncoder(object):
    """
    convert multi-label to a binary vector.
    """
    def __init__(self, sep='|'):
        self.sep = sep
        self.mapping = None

    def __call__(self, df):
        labels = set(g for col in df.values for g in col.split(self.sep))
        mapping = {label: i for i, label in enumerate(labels)}
        self.mapping = mapping
        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for label in col.split(self.sep):
                x[i, mapping[label]] = 1
        return x

class LabelEncoder(object):
    """
    convert label to a binary vector.
    """
    def __init__(self):
        self.mapping = None
        self.dtype = torch.int8

    def __call__(self, df):
        labels = set(col for col in df.values)
        mapping = {label: i for i, label in enumerate(labels)}
        self.mapping = mapping
        result = []
        for i, col in enumerate(df.values):
            result.append(mapping[col])
        return torch.from_numpy(
            np.array(result)).view(-1, 1).to(self.dtype)

class OneHotEncoder(object):
    """
    convert label to a binary vector.
    """
    def __init__(self, sep='|'):
        self.sep = sep
        self.mapping = None

    def __call__(self, df):
        labels = set(col for col in df.values)
        mapping = {label: i for i, label in enumerate(labels)}
        self.mapping = mapping
        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            x[i, mapping[col]] = 1
        return x

# END: [Encoders] 

# BEGIN: [Loaders]
def load_node_from_DB(db, select_SQL, index_col, encoders=None):
    etl = ETLBase()
    df = etl.select_table(db, select_SQL)
    df.set_index(index_col, inplace=True)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping

def load_edge_from_DB(db, select_SQL, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    etl = ETLBase()
    df = etl.select_table(db, select_SQL)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


# END: [Loaders]
# %% w103
sql = '''
SELECT * 
FROM mlaas_rawdata.witwo103
'''
table = etl.select_table('rawdata', sql)
table

# %% w106
sql = '''
SELECT * 
FROM mlaas_rawdata.witwo106
'''
table = etl.select_table('rawdata', sql)
table

# %% check w106 features: 
sql = '''
SELECT DISTINCT prod_attr_5_ind
FROM mlaas_rawdata.witwo106
'''
table = etl.select_table('rawdata', sql)
table
# %% joining w103 and w106 

sql = '''
WITH w103 AS (
    SELECT DISTINCT wm_prod_code
    FROM mlaas_rawdata.witwo103 AS w103
), 
w106 AS(
    SELECT *
    FROM mlaas_rawdata.witwo106 
)
SELECT w106.*
FROM w103
LEFT JOIN w106
ON w103.wm_prod_code = w106.wm_prod_code
ORDER BY wm_prod_code
'''
table = etl.select_table('rawdata', sql)
table.set_index('wm_prod_code')

table.columns.tolist()
# %% Load Cust Nodes: 
sql = '''
SELECT DISTINCT cust_id
FROM mlaas_rawdata.witwo103
'''
index_col = 'cust_id'
_, cust_mapping = load_node_from_DB(
    'rawdata', sql, index_col, encoders=None)

print('Number of cust_id in witwo103:', len(cust_mapping))

# %% Load Prod Nodes:
index_col = 'wm_prod_code'
sql = '''
SELECT DISTINCT wm_prod_code
FROM mlaas_rawdata.witwo103
LEFT JOIN 
SELECT * 
FROM mlaas_rawdata.witwo106
ON wm_prod_code
'''
_, prod_mapping = load_node_from_DB(
    'rawdata', sql, index_col, encoders=None)

print('Number of prod_id in witwo103:', len(prod_mapping))


# %% Load Cust Nodes with Features 
sql = '''
SELECT cust_id
FROM mlaas_rawdata.witwo103
'''
index_col = 'cust_id'
_, cust_mapping = load_node_from_DB(
    'rawdata', sql, index_col, encoders=None)

print('Number of cust_id in witwo103:', len(cust_mapping))

# %% Load Prod Nodes with Features
"""
    [X] mkt_rbot_ctg_ic 市場分類(IC)代碼
    [X] counterparty_code 上手公司代碼
    [X] prod_ccy 計價幣別
    [ ] prod_risk_code 產品風險收益等級(公會)
    [X] esun_prod_risk_code 內部風險收益等級(玉山)
    [X] prod_detail_type_code 產品分類: 境內外基金還是其他產品，如海外債、海外股票、ETF等等。?
    [X] prod_detail2_type_code 股債別代碼: 債、股、貨幣型。
    [ ] prod_attr_5_ind 精選與否
"""
sql = '''
WITH w103 AS (
    SELECT DISTINCT wm_prod_code
    FROM mlaas_rawdata.witwo103 AS w103
), 
w106 AS(
    SELECT *
    FROM mlaas_rawdata.witwo106 
)
SELECT w106.*
FROM w103
LEFT JOIN w106
ON w103.wm_prod_code = w106.wm_prod_code
ORDER BY wm_prod_code
'''
sql = """
SELECT *
FROM mlaas_rawdata.witwo106
"""
index_col = 'wm_prod_code'
prod_ccy_encoder = OneHotEncoder()
mkt_rbot_ctg_ic_encoder = OneHotEncoder()
esun_prod_risk_code_encoder = OneHotEncoder()
counterparty_code_encoder = OneHotEncoder()
product_type_encoder = OneHotEncoder()
stock_bond_type_encoder = OneHotEncoder()
elite_encoder = OneHotEncoder()
prod_x, prod_mapping = load_node_from_DB(
    'rawdata', sql, index_col, encoders={
        'prod_ccy': prod_ccy_encoder, 
        'mkt_rbot_ctg_ic': mkt_rbot_ctg_ic_encoder, 
        'esun_prod_risk_code': esun_prod_risk_code_encoder,
        'counterparty_code': counterparty_code_encoder,
        'prod_detail_type_code': product_type_encoder, 
        'prod_detail2_type_code': stock_bond_type_encoder,
        'prod_attr_5_ind': elite_encoder
    })
print(prod_x.shape)


print('Number of prod_id in witwo103:', len(prod_mapping))
print('prod_ccy dimension:', 
    len(prod_ccy_encoder.mapping))
print('mkt_rbot_ctg_ic dimension:', 
    len(mkt_rbot_ctg_ic_encoder.mapping))
print('esun_prod_risk_code dimension:', 
    len(esun_prod_risk_code_encoder.mapping))
print('counterparty_code dimension:', 
    len(counterparty_code_encoder.mapping))
print('product_type dimension:', 
    len(product_type_encoder.mapping))
print('stock_bond_type dimension:', 
    len(stock_bond_type_encoder.mapping))
print('elite dimension:', 
    len(elite_encoder.mapping))

# %% Load Prod Nodes with Features
"""
    [X] mkt_rbot_ctg_ic 市場分類(IC)代碼
    [X] counterparty_code 上手公司代碼
    [X] prod_ccy 計價幣別
    [ ] prod_risk_code 產品風險收益等級(公會)
    [X] esun_prod_risk_code 內部風險收益等級(玉山)
    [X] prod_detail_type_code 產品分類: 境內外基金還是其他產品，如海外債、海外股票、ETF等等。?
    [X] prod_detail2_type_code 股債別代碼: 債、股、貨幣型。
    [X] prod_attr_5_ind 精選與否
"""
sql = '''
WITH w103 AS (
    SELECT DISTINCT wm_prod_code
    FROM mlaas_rawdata.witwo103 AS w103
), 
w106 AS(
    SELECT *
    FROM mlaas_rawdata.witwo106 
)
SELECT w106.*
FROM w103
LEFT JOIN w106
ON w103.wm_prod_code = w106.wm_prod_code
ORDER BY wm_prod_code
'''
sql = """
SELECT *
FROM mlaas_rawdata.witwo106
"""
index_col = 'wm_prod_code'
prod_ccy_encoder = LabelEncoder()
mkt_rbot_ctg_ic_encoder = LabelEncoder()
esun_prod_risk_code_encoder = OneHotEncoder()
counterparty_code_encoder = LabelEncoder()
product_type_encoder = OneHotEncoder()
stock_bond_type_encoder = OneHotEncoder()
elite_encoder = OneHotEncoder()
prod_x, prod_mapping = load_node_from_DB(
    'rawdata', sql, index_col, encoders={
        'prod_ccy': prod_ccy_encoder, 
        'mkt_rbot_ctg_ic': mkt_rbot_ctg_ic_encoder, 
        'esun_prod_risk_code': esun_prod_risk_code_encoder,
        'counterparty_code': counterparty_code_encoder,
        'prod_detail_type_code': product_type_encoder, 
        'prod_detail2_type_code': stock_bond_type_encoder,
        'prod_attr_5_ind': elite_encoder
    })
print(prod_x.shape)


print('Number of prod_id in witwo103:', len(prod_mapping))
print('prod_ccy dimension:', 
    len(prod_ccy_encoder.mapping))
print('mkt_rbot_ctg_ic dimension:', 
    len(mkt_rbot_ctg_ic_encoder.mapping))
print('esun_prod_risk_code dimension:', 
    len(esun_prod_risk_code_encoder.mapping))
print('counterparty_code dimension:', 
    len(counterparty_code_encoder.mapping))
print('product_type dimension:', 
    len(product_type_encoder.mapping))
print('stock_bond_type dimension:', 
    len(stock_bond_type_encoder.mapping))
print('elite dimension:', 
    len(elite_encoder.mapping))



# %%
check_indicator("Check relation attribute")
edge_index, edge_label = load_edge_csv(
    rating_path,
    src_index_col='userId',
    src_mapping=user_mapping,
    dst_index_col='movieId',
    dst_mapping=movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

print(edge_index)

# %% 

