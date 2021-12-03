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
    - [X] Using the Heterogenous Convolution Wrapper: 
            Using the Heterogenous Convolution Wrapper
    - [X] 調整graph架構，使資料可以餵進去


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
    'rawdata', sql, 'cust_id', 
    encoders = {'cust_id2': cust_encoder})

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


# %% Build HeteroData
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


def invert_edge_index(edge_index):
    return torch.stack([edge_index[1], edge_index[0]], dim=0)

data['prod', 'purchase_', 'cust'].edge_index = invert_edge_index(
    data['cust', 'purchase', 'prod'].edge_index)

data['prod', 'redeem_', 'cust'].edge_index = invert_edge_index(
    data['cust', 'redeem', 'prod'].edge_index)

data['prod', 'dividend_from_', 'cust'].edge_index = invert_edge_index(
    data['cust', 'dividend_from', 'prod'].edge_index)

data['prod', 'transfer_to_', 'cust'].edge_index = invert_edge_index(
    data['cust', 'transfer_to', 'prod'].edge_index)

data['prod', 'transfer_from_', 'cust'].edge_index = invert_edge_index(
    data['cust', 'transfer_from', 'prod'].edge_index)


# data = data.to('cuda:0')
# %% Connect Model
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear
USE_PACKAGE = True
if USE_PACKAGE:
    from torch_geometric.nn import HeteroConv
else:
    from typing import Dict, Optional
    from torch_geometric.typing import NodeType, EdgeType, Adj

    from collections import defaultdict

    from torch import Tensor
    from torch.nn import Module, ModuleDict
    from torch_geometric.nn.conv.hgt_conv import group


    class HeteroConv(Module):
        r"""A generic wrapper for computing graph convolution on heterogeneous
        graphs.
        This layer will pass messages from source nodes to target nodes based on
        the bipartite GNN layer given for a specific edge type.
        If multiple relations point to the same destination, their results will be
        aggregated according to :attr:`aggr`.
        In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
        especially useful if you want to apply different message passing modules
        for different edge types.

        .. code-block:: python

            hetero_conv = HeteroConv({
                ('paper', 'cites', 'paper'): GCNConv(-1, 64),
                ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
                ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
            }, aggr='sum')

            out_dict = hetero_conv(x_dict, edge_index_dict)

            print(list(out_dict.keys()))
            >>> ['paper', 'author']

        Args:
            convs (Dict[Tuple[str, str, str], Module]): A dictionary
                holding a bipartite
                :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
                individual edge type.
            aggr (string, optional): The aggregation scheme to use for grouping
                node embeddings generated by different relations.
                (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
                :obj:`None`). (default: :obj:`"sum"`)
        """
        def __init__(self, convs: Dict[EdgeType, Module],
                    aggr: Optional[str] = "sum"):
            super().__init__()
            self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
            self.aggr = aggr

        def reset_parameters(self):
            for conv in self.convs.values():
                conv.reset_parameters()

        def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Adj],
            *args_dict,
            **kwargs_dict,
        ) -> Dict[NodeType, Tensor]:
            r"""
            Args:
                x_dict (Dict[str, Tensor]): A dictionary holding node feature
                    information for each individual node type.
                edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                    holding graph connectivity information for each individual
                    edge type.
                *args_dict (optional): Additional forward arguments of invididual
                    :class:`torch_geometric.nn.conv.MessagePassing` layers.
                **kwargs_dict (optional): Additional forward arguments of
                    individual :class:`torch_geometric.nn.conv.MessagePassing`
                    layers.
                    For example, if a specific GNN layer at edge type
                    :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                    forward argument, then you can pass them to
                    :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                    :obj:`edge_attr_dict = { edge_type: edge_attr }`.
            """
            out_dict = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                print('In HeteroConv edge_type:', edge_type)
                src, rel, dst = edge_type

                str_edge_type = '__'.join(edge_type)
                if str_edge_type not in self.convs:
                    continue

                args = []
                for value_dict in args_dict:
                    if edge_type in value_dict:
                        args.append(value_dict[edge_type])
                    elif src == dst and src in value_dict:
                        args.append(value_dict[src])
                    elif src in value_dict or dst in value_dict:
                        args.append(
                            (value_dict.get(src, None), value_dict.get(dst, None)))

                kwargs = {}
                for arg, value_dict in kwargs_dict.items():
                    arg = arg[:-5]  # `{*}_dict`
                    if edge_type in value_dict:
                        kwargs[arg] = value_dict[edge_type]
                    elif src == dst and src in value_dict:
                        kwargs[arg] = value_dict[src]
                    elif src in value_dict or dst in value_dict:
                        kwargs[arg] = (value_dict.get(src, None),
                                    value_dict.get(dst, None))

                conv = self.convs[str_edge_type]
                print("x_dict in heterconv forward:", x_dict)
                if src == dst:
                    out = conv(x_dict[src], edge_index, *args, **kwargs)
                else:
                    out = conv((x_dict[src], x_dict[dst]), edge_index, *args,
                            **kwargs)

                out_dict[dst].append(out)

            for key, value in out_dict.items():
                out_dict[key] = group(value, self.aggr)

            return out_dict

        def __repr__(self) -> str:
            return f'{self.__class__.__name__}(num_relations={len(self.convs)})'

# %%

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            
            conv = HeteroConv({
                ('cust', 'purchase', 'prod'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('cust', 'redeem', 'prod'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('cust', 'transfer_to', 'prod'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('cust', 'transfer_from', 'prod'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('cust', 'dividend_from', 'prod'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('prod', 'purchase_', 'cust'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('prod', 'redeem_', 'cust'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('prod', 'transfer_to_', 'cust'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('prod', 'transfer_from_', 'cust'): SAGEConv(
                    (-1, -1), 
                    hidden_channels),
                ('prod', 'dividend_from_', 'cust'): SAGEConv(
                    (-1, -1), 
                    hidden_channels)
            }, aggr='sum')
            
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            print('i:', i)
            """
            if i % 2 == 0: 
                edge_dict = dict([
                    (edge_type, tensor) for edge_type, tensor \
                    in edge_index_dict.items() if edge_type[0] == 'cust'])
            else:
                edge_dict = dict([
                    (edge_type, tensor) for edge_type, tensor \
                    in edge_index_dict.items() if edge_type[0] == 'prod'])
            print('edge_types in edge_dict:', edge_dict.keys())
            """
            
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            print('x_dict after relu:', x_dict)
        print('final x_dict: ', x_dict)
        return self.lin(x_dict['prod'])

model = HeteroGNN(
    hidden_channels=8, 
    out_channels=len(esun_prod_risk_code_encoder.mapping),
    num_layers=3)

# %% Lazy Initialization (initialize the model by calling it once)

with torch.no_grad():  # Initialize lazy modules.
     out = model(data.x_dict, data.edge_index_dict)

print('SUCCESSFULLY CONNECT DATA TO MODEL')
print('Output:', out) 
print('Output Shape:', out.shape)

# %%
