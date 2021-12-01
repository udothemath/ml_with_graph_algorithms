"""
- [X] 有兩種node，那x要怎麼給:  
    使用HeteroData: 
        https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

- [ ] 建立 pytorch geometric的 Data : 
    - [X] x 
    - [X] y (要定義)
    - [X] edge_index 
    - [ ] train_mask / val_mask / test_mask 
    - data.train_idx = torch.tensor([...], dtype=torch.long)
    - data.test_mask = torch.tensor([...], dtype=torch.bool) 
- [ ] 使用 Hetero Graph 程式 
    - [ ] Using the Heterogenous Convolution Wrapper: Using the Heterogenous Convolution Wrapper
    - [ ] 調整graph架構，使資料可以餵進去


"""

# %%

import os
import pandas as pd
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
def load_node_from_DB(db, select_SQL, index_col, encoders=None, y_encoder=None):
    """
    Args: 
        - db: feature | rawdata 
        - select_SQL 
        - index_col: 目標欄位 
        - encoders (dict): 處理欄位的encoder 
        - y_encoder (tuple): 抓出y欄位的encoder 
    """
    etl = ETLBase()
    df = etl.select_table(db, select_SQL)
    df.set_index(index_col, inplace=True)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    y= None
    if y_encoder is not None:
        y_col, encoder = y_encoder
        y = encoder(df[y_col])
        
    return x, y, mapping

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

# %% Load prod node
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
esun_prod_risk_code_encoder = LabelEncoder()
counterparty_code_encoder = OneHotEncoder()
product_type_encoder = OneHotEncoder()
stock_bond_type_encoder = OneHotEncoder()
elite_encoder = OneHotEncoder()

prod_x, prod_y, prod_mapping = load_node_from_DB(
    'rawdata', sql, index_col, encoders={
        'prod_ccy': prod_ccy_encoder, 
        'mkt_rbot_ctg_ic': mkt_rbot_ctg_ic_encoder, 
        'counterparty_code': counterparty_code_encoder,
        'prod_detail_type_code': product_type_encoder, 
        'prod_detail2_type_code': stock_bond_type_encoder,
        'prod_attr_5_ind': elite_encoder
    },
    y_encoder = ('esun_prod_risk_code', esun_prod_risk_code_encoder)
    )
print(prod_x.shape)
print(prod_y.shape)

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


# %% Load Cust Node 
sql = '''
SELECT cust_id, cust_id AS cust_id2
FROM mlaas_rawdata.witwo103 AS w103
'''
cust_encoder = OneHotEncoder()

cust_x, _, cust_mapping = load_node_from_DB(
    'rawdata', sql, 'cust_id', encoders = {'cust_id2': cust_encoder})

# %% Load Edges 

# 
sql = '''
SELECT 
    cust_id, 
    wm_prod_code
FROM mlaas_rawdata.witwo103 AS w103
WHERE wm_txn_code = 3
'''
dividend_edge_index, _ = load_edge_from_DB(
    'rawdata', sql, 
    'cust_id', cust_mapping, 
    'wm_prod_code', prod_mapping,
    encoders=None)

sql = '''
SELECT 
    cust_id, 
    wm_prod_code
FROM mlaas_rawdata.witwo103 AS w103
WHERE wm_txn_code = 2
'''
redeem_edge_index, _ = load_edge_from_DB(
    'rawdata', sql, 
    'cust_id', cust_mapping, 
    'wm_prod_code', prod_mapping,
    encoders=None)

sql = '''
SELECT 
    cust_id, 
    wm_prod_code
FROM mlaas_rawdata.witwo103 AS w103
WHERE wm_txn_code = 1
'''
purchase_edge_index, _ = load_edge_from_DB(
    'rawdata', sql, 
    'cust_id', cust_mapping, 
    'wm_prod_code', prod_mapping,
    encoders=None)

sql = '''
SELECT 
    cust_id, 
    wm_prod_code
FROM mlaas_rawdata.witwo103 AS w103
WHERE wm_txn_code = 4 AND 
WM_TXN_TRF_TYPE = 1
'''
transfer_to_edge_index, _ = load_edge_from_DB(
    'rawdata', sql, 
    'cust_id', cust_mapping, 
    'wm_prod_code', prod_mapping,
    encoders=None)

sql = '''
SELECT 
    cust_id, 
    wm_prod_code
FROM mlaas_rawdata.witwo103 AS w103
WHERE wm_txn_code = 4 AND 
WM_TXN_TRF_TYPE = 2
'''
transfer_from_edge_index, _ = load_edge_from_DB(
    'rawdata', sql, 
    'cust_id', cust_mapping, 
    'wm_prod_code', prod_mapping,
    encoders=None)


# %% 
# https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html

from torch_geometric.data import HeteroData
data = HeteroData()
data['prod'].x = prod_x # [num_prod, num_feature_prod]
data['prod'].y = prod_y 
data['cust'].x = cust_x
data['cust', 'purchase', 'prod'].edge_index = purchase_edge_index # [2, num_edge]
data['cust', 'redeem', 'prod'].edge_index = redeem_edge_index # [2, num_edge]
data['cust', 'dividend_from', 'prod'].edge_index = dividend_edge_index # [2, num_edge]
data['cust', 'transfer_to', 'prod'].edge_index = transfer_to_edge_index # [2, num_edge]
data['cust', 'transfer_from', 'prod'].edge_index = transfer_from_edge_index # [2, num_edge]

# data = data.to('cuda:0')
# %%

from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('cust', 'purchase', 'prod'): GCNConv(-1, hidden_channels),
                ('cust', 'redeem', 'prod'): GATConv((-1, -1), hidden_channels),
                ('cust', 'transfer_to', 'prod'): SAGEConv((-1, -1), hidden_channels),
                ('cust', 'transfer_from', 'prod'): SAGEConv((-1, -1), hidden_channels),
                ('cust', 'dividend_from', 'prod'): SAGEConv((-1, -1), hidden_channels)
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['prod'])

model = HeteroGNN(
    hidden_channels=64, 
    out_channels=len(esun_prod_risk_code_encoder.mapping),
    num_layers=2)

# %% Lazy Initialization (initialize the model by calling it once)

with torch.no_grad():  # Initialize lazy modules.
     out = model(data.x_dict, data.edge_index_dict)


# %%
