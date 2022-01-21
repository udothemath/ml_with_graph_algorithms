#%%
import pandas as pd
import numpy as np
from util import gen_sal_pivot_table, gen_txn_cnt_pivot_table, mask_generation, gen_edge_index, gen_shop_tag_feat

from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import SAGEConv
from torch_cluster import random_walk
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
# %%
data = pd.read_csv('./data/to崇爾/tbrain_cc_training_48tags_hash_final.csv')
# %%
data.sample(5)
# %%
data = data[['dt', 'chid', 'shop_tag', 'txn_cnt', 'txn_amt', 'gender_code', 'age',	'primary_card']]
data['dt'] = data['dt'].astype('int')
data['chid'] = data['chid'].astype('int')
data['gender_code'] = data['gender_code'].apply(lambda x: x if np.isnan(x) else int(x))

# %%
# ignore data with shop_tag = 'other'
shop_tag_list = ['2', '6', '10', '12', '13', '15', '18', '19', '21', '22', '25', '26', '36', '37', '39', '48']
num_tag = len(shop_tag_list)
sub_data = data[data['shop_tag'].isin(shop_tag_list)]
sub_data = sub_data.dropna()
# %%
# data = data.dropna(how='any', axis=0)
print('Data shape: ', sub_data.shape)
print('Number of tag: ', sub_data['shop_tag'].nunique())
sub_data.sample(3)
# %%
class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo() 
        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

def gen_node_embedding(data, chid_shop_tag_pivot_table, shop_tag_list):
    edge_index = gen_edge_index(chid_shop_tag_pivot_table, shop_tag_list)
    # generate shop_tag initial feature(one hot encoding)
    shop_tag_feat = gen_shop_tag_feat()
    chid_shop_tag_pivot_table = chid_shop_tag_pivot_table.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), on='chid', how='inner')
    # normalize numeric features
    cols_to_norm = shop_tag_list + ['age']
    chid_shop_tag_pivot_table[cols_to_norm] = chid_shop_tag_pivot_table[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # Label generation
    y = np.argmax(chid_shop_tag_pivot_table[cols_to_norm[:-1]].values, axis=1) # (num_chid, )
    y = np.concatenate((y, np.array([num_tag] * num_tag)), axis=None) # (num_chid + num_tag, )

    all_chid = chid_shop_tag_pivot_table['chid']
    chid_feat = chid_shop_tag_pivot_table.drop(columns=['chid']).values # (num_chid, 19)
    all_feat = np.concatenate((chid_feat, shop_tag_feat), axis=0) # (num_chid + num_tag, 19)
    all_feat = torch.Tensor(all_feat)

    gdata = Data()
    gdata.x, gdata.edge_index, gdata.y = all_feat, edge_index, y
    gdata.train_mask, gdata.val_mask, gdata.test_mask = mask_generation(gdata.x.shape[0])
    gdata = T.ToUndirected()(gdata)

    train_loader = NeighborSampler(gdata.edge_index, sizes=[10, 10], batch_size=512,
                               shuffle=True, num_nodes=gdata.x.shape[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, edge_index = gdata.x.to(device), gdata.edge_index.to(device)
    model = SAGE(gdata.x.shape[1], hidden_channels=64, num_layers=2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_acc, best_test_acc = float('-inf'), float('inf')
    embed = torch.zeros(x.shape[0], 64)
    train_epoch = 5
    for epoch in range(1, train_epoch):
        model.train()
        total_loss = 0
        for batch_size, n_id, adjs in train_loader: # batch_size: 768
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean() # (-∞, 0]
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * out.size(0)
        loss = total_loss / x.shape[0]
        
        # @torch.no_grad()
        model.eval()
        with torch.no_grad():
            out = model.full_forward(x, edge_index).cpu()
            clf = LogisticRegression(solver='liblinear')
            clf.fit(out[gdata.train_mask], gdata.y[gdata.train_mask])

            val_acc = clf.score(out[gdata.val_mask], gdata.y[gdata.val_mask])
            test_acc = clf.score(out[gdata.test_mask], gdata.y[gdata.test_mask])
            if test_acc > best_test_acc and val_acc > best_test_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                embed = out

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    chid_embed, shop_tag_embed = embed[:-16], embed[-16:]
    embed_df = pd.DataFrame(chid_embed.numpy()).add_suffix('_embed')
    embed_df = embed_df.assign(chid=all_chid.values)
    return embed_df, shop_tag_embed
#%%
# qid is fixed with current txn_cnt
def gen_data_for_ranker(data, start_month, end_month):
    x = gen_sal_pivot_table(data, start_month, end_month, False)
    y = gen_sal_pivot_table(data, end_month, end_month, True)
    assert list(x.columns) == list(y.columns), 'Column name does not match!'

    print('===========Pivot table generation completed===========')
    # Remove chid which contains missing value in 'gender_code', 'age', 'primary_card'
    x = x.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), left_on='chid', right_on='chid', how='inner')
    x.reset_index(inplace=True)
    x.drop(columns=['index', 'gender_code', 'age', 'primary_card'], inplace=True)
    # Graph data and model construction
    embed_df, shop_tag_embed = gen_node_embedding(data, x, shop_tag_list)
    return embed_df, shop_tag_embed
embed_df, shop_tag_embed = gen_data_for_ranker(sub_data, 1, 24)

#%%
def gen_data_for_ranker(data, start_month, end_month):
    x = gen_sal_pivot_table(data, start_month, end_month, False)
    y = gen_sal_pivot_table(data, end_month, end_month, True)
    assert list(x.columns) == list(y.columns), 'Column name does not match!'

    print('===========Pivot table generation completed===========')
    # Remove chid which contains missing value in 'gender_code', 'age', 'primary_card'
    x = x.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), left_on='chid', right_on='chid', how='inner')
    x.reset_index(inplace=True)
    x.drop(columns=['index', 'gender_code', 'age', 'primary_card'], inplace=True)
    # Graph data and model construction
    # embed_df, shop_tag_embed = gen_node_embedding(data, x, shop_tag_list)

    # 以x為重，x有的chid，y如果沒有，y新增該chid並補值0
    keep_same = {'chid'}
    x.columns = ['{}{}'.format(col, '' if col in keep_same else '_sal') for col in x.columns]

    tmp_y = x.merge(y, on='chid', how='left')
    y = tmp_y[['chid', '10','12', '13', '15', '18', '19', '2', '21', '22', '25', \
       '26', '36', '37', '39', '48', '6']]
    y.fillna(value=0, inplace=True)

    last_three_months_txn_cnt = gen_txn_cnt_pivot_table(data, end_month-4, end_month-1)
    last_three_months_txn_cnt.columns = ['{}{}'.format(col, '' if col in keep_same else '_sal') for col in last_three_months_txn_cnt.columns]

    x = x.merge(last_three_months_txn_cnt, on='chid', how='left')
    x.fillna(value=0, inplace=True)
    
    x['index'], y['index'] = range(0, len(x)), range(0, len(y))
    x, y = x.set_index('index'), y.set_index('index')

    assert x[['chid']].equals(y[['chid']]), 'chid does not match!'
    res_list = []
    y_rows = y.values[:, 1:]
    argsort_y_rows = np.argsort(y_rows, axis=1)
    num_tag = y_rows.shape[1]
    num_nonzero = np.count_nonzero(y_rows, axis=1)

    for idx, row in x.iterrows():
        print("\r", end=str(idx))
        max_rank = num_nonzero[idx]
        rank_row = np.zeros(num_tag)
        for max_idx in range(num_tag-1, num_tag-num_nonzero[idx]-1, -1):
            rank_row[argsort_y_rows[idx][max_idx]] = max_rank
            max_rank -= 1

        chid = row[0]
        x_sal = row.values[1:1+num_tag]
        x_cnt = row.values[1+num_tag:]
        row_sum = np.sum(x_sal)
        for tag_idx in range(num_tag):
            tag_sal_list = [0] * (num_tag + 2)
            tag_sal_list[-1] = rank_row[tag_idx]
            if x_sal[tag_idx] == 0:
                tag_sal_list[-2] = row_sum
            else:
                tag_sal_list[-2] = row_sum - x_sal[tag_idx]
                tag_sal_list[tag_idx] = x_sal[tag_idx]
            tag_sal_list += x_cnt.tolist()
            tag_sal_list.insert(0, chid)
            res_list.append(tag_sal_list)
    res_df = pd.DataFrame(res_list)
    res_df.columns = ['chid' , '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', \
                      '11', '12', '13', '14', '15', '16', 'remain', 'rank', '1_cnt', \
                      '2_cnt', '3_cnt', '4_cnt', '5_cnt', '6_cnt', '7_cnt', '8_cnt', \
                      '9_cnt', '10_cnt', '11_cnt', '12_cnt', '13_cnt', '14_cnt', \
                      '15_cnt', '16_cnt']

    res_df = res_df.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), left_on='chid', right_on='chid', how='inner')
    return res_df
#%%
# gen_data_for_ranker(sub_data, 1, 2)
res_df = gen_data_for_ranker(sub_data, 1, 24)
# res_df.shape
# %%
# (training, validation, testing) ratio = (0.7, 0.2, 0.1)

train_ratio, val_ratio = 0.7, 0.2 
train_size = int(res_df.shape[0] * train_ratio)
train_size = int(train_size // 16 * 16)

val_size = int(res_df.shape[0] * val_ratio)
val_size = int(val_size // 16 * 16)

train_df = res_df[:train_size]
val_df = res_df[train_size: train_size+val_size]
test_df = res_df[train_size+val_size:]

qids_train = np.array([num_tag] * (train_df.shape[0] // num_tag))
x_train = train_df.drop(columns=['chid', 'rank'])
y_train = train_df['rank']

qids_val = np.array([num_tag] * (val_df.shape[0] // num_tag))
x_val = val_df.drop(columns=['chid', 'rank'])
y_val = val_df['rank']

qids_test = np.array([num_tag] * (test_df.shape[0] // num_tag))
x_test = test_df.drop(columns=['chid', 'rank'])
y_test = test_df['rank']

print(qids_train.shape)
print(x_train.shape)
print(y_train.shape)
print(qids_val.shape)
print(x_val.shape)
print(y_val.shape)
print(qids_test.shape)
print(x_test.shape)
print(y_test.shape)
# %%
import lightgbm as lgbm
model = lgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=500,
    num_leaves=100,
    max_depth=8,
    learning_rate=0.01,
    random_state=123
)
model.fit(
    X=x_train,
    y=y_train,
    group=qids_train,
    eval_set=[(x_train, y_train), (x_val, y_val), (x_test, y_test)],
    eval_names=['training', 'validation', 'testing'],
    eval_group=[qids_train, qids_val, qids_test],
    eval_at=3,
    verbose=10,
    early_stopping_rounds=50
)
# %%
model.best_iteration_
# %%
x = pd.concat([x_train, x_val, x_test])
y = pd.concat([y_train, y_val, y_test])
qids = np.array([num_tag] * (x.shape[0] // num_tag))
# %%
model.fit(
    X=x,
    y=y,
    group=qids,
    eval_set=[(x, y)],
    eval_names=['training'],
    eval_group=[qids],
    eval_at=3,
    verbose=10
)
# %%
# # save model
# model.booster_.save_model('model_7008.txt')
# # load model
# model = lgbm.Booster(model_file='model_7008.txt')
# %%
def gen_testing_data_for_ranker(data, start_month, end_month, num_tag=16):
    x = gen_sal_pivot_table(data, start_month, end_month, True)
    last_three_months_txn_cnt = gen_txn_cnt_pivot_table(data, end_month-3, end_month)
    last_three_months_txn_cnt.rename({'10':'10_cnt' ,'12':'12_cnt', '13':'13_cnt', '15':'15_cnt', \
                                     '18':'18_cnt', '19':'19_cnt', '2':'2_cnt', '21':'21_cnt', \
                                     '22':'22_cnt', '25':'25_cnt', '26':'26_cnt', '36':'36_cnt', \
                                     '37':'37_cnt', '39':'39_cnt', '48':'48_cnt', '6':'6_cnt'}, inplace=True)
    x = x.merge(last_three_months_txn_cnt, on='chid', how='left')
    x.fillna(value=0, inplace=True)
    res_list = []
    for idx, row in x.iterrows():
        chid = row[0]
        sal = row.values[1:1+num_tag]
        cnt = row.values[1+num_tag:]
        row_sum = np.sum(sal)
        for tag_idx in range(num_tag):
            tag_sal_list = [0] * (num_tag + 1)
            if sal[tag_idx] == 0:
                tag_sal_list[-1] = row_sum
            else:
                tag_sal_list[-1] = row_sum - sal[tag_idx]
                tag_sal_list[tag_idx] = sal[tag_idx]
            tag_sal_list += cnt.tolist()
            tag_sal_list.insert(0, chid)
            res_list.append(tag_sal_list)
    res_df = pd.DataFrame(res_list)
    res_df.columns = ['chid' , '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', \
                      '11', '12', '13', '14', '15', '16', 'remain', '1_cnt', '2_cnt', \
                      '3_cnt', '4_cnt', '5_cnt', '6_cnt', '7_cnt', '8_cnt', '9_cnt', \
                      '10_cnt', '11_cnt', '12_cnt', '13_cnt', '14_cnt', '15_cnt', '16_cnt']
    res_df = res_df.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), left_on='chid', right_on='chid', how='inner')
    return res_df
#%%
testing_data = gen_testing_data_for_ranker(sub_data, 1, 24)
# assert testing_data.shape[0] == np.sum(testing_qids), 'Query num and qid num do not match!!'
#%% rule-based
top3_txn_amt_tag = sub_data.groupby('shop_tag').agg(sal=('txn_amt', 'sum')).sort_values(by='sal', ascending=False).head(3)
top3_txn_amt_tag = list(map(int, list(top3_txn_amt_tag.index[:3])))

chid = testing_data.chid.unique() 
orig_chid = data.chid.unique()
disad_grp_id = pd.DataFrame(set(orig_chid).difference(set(testing_data['chid'])), columns=['chid'])
chid = pd.concat([pd.DataFrame(chid, columns=['chid']), disad_grp_id], axis=0).reset_index(drop=True).astype('int')

testing_data.drop(columns=['chid'], inplace=True)
testing_data.shape
# %%
pred = model.predict(testing_data)
map_dict = {1:'10', 2:'12', 3:'13', 4:'15',
            5:'18', 6:'19', 7:'2', 8:'21', 
            9:'22', 10:'25', 11:'26', 12:'36',
            13:'37', 14:'39', 15:'48', 16:'6'}
final_rank_lst = []
for i in range(pred.shape[0] // 16):
    rank_res = np.argsort(pred[i*16:(i+1)*16])[::-1][:3]
    for idx, res in enumerate(rank_res):
        rank_res[idx] = map_dict[res+1]
    final_rank_lst.append(rank_res.tolist())
# %%
print('before: ', len(final_rank_lst))
final_rank_lst = final_rank_lst + [top3_txn_amt_tag] * disad_grp_id.shape[0]
print('after: ', len(final_rank_lst))
# %%
final_res = pd.concat([chid, pd.DataFrame(final_rank_lst)], axis=1).rename(columns={0: 'top1', 1: 'top2', 2: 'top3'})
final_res.to_csv('./result.csv', index=False)
# %%
final_res
