import pandas as pd


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
