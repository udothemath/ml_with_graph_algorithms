# %%
import json
import csv
from datetime import datetime
from typing import TypedDict
import os
import pandas as pd
import numpy as np
import pprint
from zipfile import ZipFile
import zipfile
from IPython.display import display

OTTO_PATH = '/Users/pro/Documents/ml_with_graph_algorithms/kaggle2022_otto/'
DATA_PATH = '/Users/pro/Documents/ml_with_graph_algorithms/kaggle2022_otto/data'

data = os.path.join(DATA_PATH, 'demo_test.jsonl')
data_sub = os.path.join(OTTO_PATH, 'udo', 'sample_submission.csv')
data_sub_test = os.path.join(OTTO_PATH, 'udo', 'sample_submission_short2.csv')

print("hello", data)

# %%


def get_df_format():
    train_sessions = pd.DataFrame()
    chunks = pd.read_json(
        data, lines=True, chunksize=1_000_000, nrows=10_000_000)

    for e, chunk in enumerate(chunks):
        print(e, chunk)
        event_dict = {
            'session': [],
            'aid': [],
            'ts': [],
            'type': [],
        }
        if e < 2:
            # train_sessions = pd.concat([train_sessions, chunk])
            for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
                for event in events:
                    event_dict['session'].append(session)
                    event_dict['aid'].append(event['aid'])
                    event_dict['ts'].append(event['ts'])
                    event_dict['type'].append(event['type'])
            chunk_session = pd.DataFrame(event_dict)
            train_sessions = pd.concat([train_sessions, chunk_session])
        else:
            break

    train_sessions = train_sessions.reset_index(drop=True)
    return train_sessions


train_sessions = get_df_format()
print(len(train_sessions))
display(train_sessions[:3])

# %%


class EDA:
    def __init__(self, input_df: pd.DataFrame):
        self.df = input_df

    def quick_check(self):
        display(self.df.shape)

    def show_unique_size(self):
        """ show unique size info of dataframe """
        print(f"--- Info of DataFrame --- ")
        print(f"Size of datafame: {self.df.shape}")
        display(self.df[:3])
        for _col in ['session', 'aid']:
            cnt_uniq_items = len(self.df[_col].unique())
            print(f"Unique size of {_col}: {cnt_uniq_items}")

    def grp_sess_show_cnt_type(self):
        _df = self.df.groupby(['session'])
        display(_df['type'].value_counts()[:3])

    def grp_aid_type_topN(self, freq_topN: int = 3):
        """ Show aid and type count """
        _df = self.df.groupby(['aid', 'type'])
        display(_df['session'].count()
                .nlargest(freq_topN).reset_index(name=f'cnt_aid_type'))

    def get_hot_aid_type(self) -> dict:
        """ Return hot items(aid) """
        _df = self.df.groupby(['aid', 'type'])
        df_hot = (_df['session'].count().reset_index(name='cnt_aid_type'))

        dict_hot_items = {}
        for cur_type in ['clicks', 'carts', 'orders']:
            mask = (df_hot['type'] == cur_type)
            cur_top = df_hot.loc[mask]\
                .sort_values(['cnt_aid_type'], ascending=False)[:20]
            cur_dict = {cur_type: (
                cur_top['aid'].to_list(), cur_top['cnt_aid_type'].to_list())}
            dict_hot_items.update(cur_dict)
        return dict_hot_items


obj_train_sess = EDA(train_sessions)

# %%
obj_train_sess.show_unique_size()
obj_train_sess.grp_aid_type_topN()

# %%
hot_items = obj_train_sess.get_hot_aid_type()
hot_clicks = np.array(hot_items['clicks'][0])
hot_carts = np.array(hot_items['carts'][0])
hot_orders = np.array(hot_items['orders'][0])

# %%
print("clicks:", hot_clicks)
print("carts:", hot_carts)
print("orders:", hot_orders)

# %%
df_data_sub_test = pd.read_csv(data_sub_test)
display(df_data_sub_test)

# %%


def generate_sub_df(type: str,
                    pred: list,
                    pred_range=range(12899779, 14571582)):
    _df = pd.DataFrame()
    list_sess_type = [f"{i}_{type}" for i in pred_range]
    _df['session_type'] = list_sess_type
    _df['labels'] = ' '.join(str(item) for item in pred)
    print(_df.shape)
    display(_df[:5])
    return _df


df_clicks = generate_sub_df("clicks", hot_clicks)
df_carts = generate_sub_df("carts", hot_carts)
df_orders = generate_sub_df("orders", hot_orders)

df_go = pd.concat([df_clicks, df_carts, df_orders])
print(df_go.shape)
display(df_go[:3])

# %%


def create_zip_sub(df: pd.DataFrame):
    """ Create submission with current datetime as filename """
    str_dt = datetime.today().strftime('%Y%m%d%H%M')
    filename = f'sub_{str_dt}'
    df.to_csv(f"{filename}.csv", index=False,
              quoting=csv.QUOTE_NONE, escapechar="")
    with ZipFile(f"{filename}.zip", mode='w', compression=zipfile.ZIP_DEFLATED) as myzip:
        myzip.write(f'{filename}.csv')

    os.remove(f'{filename}.csv')


create_zip_sub(df_go)
# def go_by_api():
#     kaggle competitions submit - c otto-recommender-system - f submission.csv - m "Message"


# %%

# %%
# %%
df_data_sub = pd.read_csv(data_sub)
df_data_sub = df_data_sub.tail(10)

col_ses_typ = ['session', 'type']
display(df_data_sub['session_type'].str)
df_data_sub[col_ses_typ] = df_data_sub['session_type'].str.split(
    '_', 1, expand=True)

display(df_data_sub)

list_session = (df_data_sub["session"].unique())
print(len(list_session))

# %%

# %%

hot_clicks = [1, 2, 3]
hot_carts = [4, 5, 6]
hot_orders = [7, 8, 9]

# df_sub_go = pd.DataFrame(columns=['session', 'type', 'session_type', 'labels'])
# df_sub_go2 = pd.DataFrame()
# for cur_session in list_session:
#     print(cur_session)
#     df_sub_go2['session'] = cur_session
#     df_sub_go2['abc'] = 'abc'
#     df_sub_go2['session_type'] = cur_session
#     display(df_sub_go)

# display(df_sub_go[:3])
# %%


def write_json(json_data: TypedDict, output_data: str):
    # with open(output_data, 'w') as out_file:
    #     json.dumps(json_data, out_file, sort_keys=True, indent=4,
    #                ensure_ascii=False)

    # Serializing json
    json_object = json.dumps(json_data, indent=4)

    # Writing to sample.json
    with open(output_data, "w") as outfile:
        outfile.write(json_object)


def read_json_as_obj(input_data: str = data):
    with open(input_data, 'r') as json_file:
        data = json.load(json_file)


def read_jsonl(input_data: str = data):
    with open(data, 'r') as json_file:

        json_list = list(json_file)

    print(len(json_list))
    print(json_list[:2])
    write_json(json_list[:2], 'test_v1.json')
    # for json_str in json_list[:2]:
    #     result = json.loads(json_str)
    #     print(f"result: {result}")
    #     # print(isinstance(result, dict))


# read_jsonl(data)

# %%
