# %%
import pandas as pd
import numpy as np
import torch
def gen_sal_pivot_table(data, start_month, end_month, y):
    if y: # label
        res = data[(data['dt'] >= start_month) & (data['dt'] <= end_month)]
    else:
        res = data[(data['dt'] >= start_month) & (data['dt'] <= end_month - 1)]
    res = res.groupby(['chid', 'shop_tag']).agg(sal=('txn_amt', 'sum')).reset_index()
    res = pd.pivot_table(res, values='sal', columns=['shop_tag'], index=['chid']).fillna(0)
    res = res.reset_index()
    return res.round(1)

def gen_txn_cnt_pivot_table(data, start_month, end_month):
    res = data[(data['dt'] >= start_month) & (data['dt'] <= end_month)]
    res = res.groupby(['chid', 'shop_tag']).agg(cnt=('txn_cnt', 'sum')).reset_index()
    res = pd.pivot_table(res, values='cnt', columns=['shop_tag'], index=['chid']).fillna(0)
    res = res.reset_index()
    return res

def mask_generation(num_node, train_ratio=0.7, val_ratio=0.2):
    train_size = int(num_node * train_ratio)
    val_size = int(num_node * val_ratio)
    test_size = num_node - train_size - val_size

    train_mask = np.zeros((num_node,), dtype=bool)
    val_mask = np.zeros((num_node,), dtype=bool)
    test_mask = np.zeros((num_node,), dtype=bool)

    train_idx = list(range(train_size))
    res_idx = list(range(train_size, num_node))
    val_idx = res_idx[:val_size]
    test_idx = res_idx[val_size:]
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

def gen_edge_index(chid_shop_tag_pivot_table, shop_tag_list):
    shop_tag_list = [int(shop_tag) for shop_tag in shop_tag_list]
    chid_mapping = {index: i for i, index in enumerate(chid_shop_tag_pivot_table.chid.unique())}
    shop_tag_mapping = {index: i for i, index in enumerate(shop_tag_list, max(chid_mapping.values())+1)}
    
    col_names = chid_shop_tag_pivot_table.columns.values[1:].astype(int)
    masks = chid_shop_tag_pivot_table.gt(0.0).values
    masks = masks[:, 1:]
    non_zero_col_names = [col_names[mask].tolist() for mask in masks]
    src, dst = [], []
    for idx, row in chid_shop_tag_pivot_table.iterrows():
        non_zero_col_num = len(non_zero_col_names[idx])
        src += [chid_mapping[row.chid]] * non_zero_col_num
        dst += [shop_tag_mapping[shop_tag] for shop_tag in non_zero_col_names[idx]]
    edge_index = torch.tensor([src, dst])
    return edge_index

def gen_shop_tag_feat(num_tag=16, num_cust_feat=3):
    idx_arr = np.array(range(0, num_tag))
    shop_tag_feat = np.zeros((idx_arr.size, idx_arr.max()+1))
    shop_tag_feat[np.arange(idx_arr.size),idx_arr] = 1  # (16, 16)
    shop_tag_feat = np.concatenate((shop_tag_feat, np.zeros((num_tag, num_cust_feat))), axis=1) # (16, 19)
    return shop_tag_feat

# qids are not fixed
def gen_data_for_ranker(data, start_month, end_month):
    x = gen_sal_pivot_table(data, start_month, end_month, False)
    y = gen_sal_pivot_table(data, end_month, end_month, True)
    
    assert list(x.columns) == list(y.columns), 'Column name does not match!'
    
    x['index'] = range(0, len(x))
    x = x.set_index('index')

    y['index'] = range(0, len(y))
    y = y.set_index('index')

    assert x[['chid']].equals(y[['chid']]), 'chid does not match!'
    res_list, qids = [], []
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
        x_row = row.values[1:]
        x_row_num_nonzero = np.count_nonzero(x_row)
        qids.append(x_row_num_nonzero)
        # print('x_row: ', x_row)

        row_sum = np.sum(x_row)
        for tag_idx in range(num_tag):
            tag_sal_list = [0] * (num_tag + 2)
            tag_sal_list[-1] = rank_row[tag_idx]
            if x_row[tag_idx] == 0:
                continue
                # tag_sal_list[-2] = row_sum
            else:
                tag_sal_list[-2] = row_sum - x_row[tag_idx]
                tag_sal_list[tag_idx] = x_row[tag_idx]
            tag_sal_list.insert(0, chid)
            res_list.append(tag_sal_list)
    res_df = pd.DataFrame(res_list)
    res_df.columns = ['chid' , '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                  '11', '12', '13', '14', '15', '16', 'remain', 'rank']
    res_df = res_df.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), left_on='chid', right_on='chid', how='inner')
    qids = np.array(qids)
    return qids, res_df

# %% qids are not fixed
def gen_testing_data_for_ranker(data, start_month, end_month):
    x = gen_sal_pivot_table(data, start_month, end_month, False)
    res_list, qids = [], []
    num_tag = 16
    for _, row in x.iterrows():
        chid = row[0]
        row = row.values[1:]
        row_num_nonzero = np.count_nonzero(row)
        qids.append(row_num_nonzero)

        row_sum = np.sum(row)
        for tag_idx in range(num_tag):
            tag_sal_list = [0] * (num_tag + 1)
            if row[tag_idx] == 0:
                continue
            else:
                tag_sal_list[-1] = row_sum - row[tag_idx]
                tag_sal_list[tag_idx] = row[tag_idx]
            tag_sal_list.insert(0, chid)
            res_list.append(tag_sal_list)
    res_df = pd.DataFrame(res_list)
    res_df.columns = ['chid' , '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                  '11', '12', '13', '14', '15', '16', 'remain']
    res_df = res_df.merge(data[['chid', 'gender_code', 'age', 'primary_card']].drop_duplicates(), left_on='chid', right_on='chid', how='inner')
    qids = np.array(qids)
    return qids, res_df
# %%
# testing_qids, testing_data = gen_testing_data_for_ranker(sub_data, 1, 24)

# %%
# qid_sum = np.sum(qids)
# train_ratio = 0.8
# train_size = int(qid_sum * train_ratio)
# qid_tmp_sum = 0
# sep_idx = 0
# min_dist = float('inf')
# for idx, qid in np.ndenumerate(qids):
#     qid_tmp_sum += qid
#     dist = train_size - qid_tmp_sum
#     if dist < min_dist:
#         min_dist = dist
#         if min_dist < 100:
#             sep_idx = idx
#             break

# train_df = res_df[:qid_tmp_sum]
# x_train = train_df.drop(columns=['chid', 'rank'])
# y_train = train_df['rank']

# val_df = res_df[qid_tmp_sum:]
# x_val = val_df.drop(columns=['chid', 'rank'])
# y_val = val_df['rank']

# qids_train = qids[:sep_idx[0]+1]
# qids_val = qids[sep_idx[0]+1:]

# print(qids_train.shape)
# print(x_train.shape)
# print(y_train.shape)
# print(qids_val.shape)
# print(x_val.shape)
# print(y_val.shape)

 # %%
# test_txn_amt = testing_data.values[:, :num_tag]
# shop_tag_idx = np.where(test_txn_amt != 0)[1]
# pred = model.predict(testing_data)
# final_rank_lst = []
# acc_sum = 0
# map_dict = {0:'10', 1:'12', 2:'13', 3:'15',
#             4:'18', 5:'19', 6:'2', 7:'21', 
#             8:'22', 9:'25', 10:'26', 11:'36',
#             12:'37', 13:'39', 14:'48', 15:'6'}
# for test_qid in testing_qids:
#     rank_res = np.argsort(pred[acc_sum:acc_sum+test_qid])[::-1][:3] # rank
#     # print('before: ', rank_res)
#     sub_shop_tag_idx = shop_tag_idx[acc_sum:acc_sum+test_qid]
#     # print('sub_shop_tag_idx: ', sub_shop_tag_idx)
#     print(pred[acc_sum:acc_sum+test_qid])
#     convert_idx = 2
#     if len(rank_res) < 3:
#         diff = sorted(list(set(top3_txn_amt_tag).difference(set(rank_res))), reverse=True)
#         if len(rank_res) == 2:
#             convert_idx = 1
#             rank_res = np.append(rank_res, diff[:1])
#         else:
#             convert_idx = 0
#             rank_res = np.append(rank_res, diff[:2])
#         # print('pad: ', rank_res)
#     # else:
#         # print('no need to pad: ', rank_res)
#     for idx, rank in enumerate(rank_res):
#         if idx <= convert_idx:
#             rank_res[idx] = sub_shop_tag_idx[rank]
#             rank_res[idx] = map_dict[rank_res[idx]] 
#     # print('after: ', rank_res)
#     final_rank_lst.append(rank_res) 
#     acc_sum += test_qi