from easy_to_sql.sql_tools import SQLTools
from dataclasses import dataclass
import pandas as pd
import os
import logging
from IPython.display import display
from addr_standardize_v1 import trans_address

MAIN_PATH = os.path.abspath(os.getcwd())
MAIN_PATH = "/home/jovyan/house-automated-valuation-model"
DATA_PATH = os.path.join(MAIN_PATH, 'data')


@dataclass
class TheInfo:
    proj_name: str
    table_name: str
    size_limit: int
    file_path: str
    file_prefix: None
    db_source: str = "feature"
    select_condition: str = "SELECT *"
    sql_condition: str = "None"


class GenCSVfromDB:
    def __init__(self, the_info, logger):
        '''
            Generate csv from db using easy_to_sql
        '''
        self._proj_name = the_info.proj_name
        self._table_name = the_info.table_name
        self._size_limit = the_info.size_limit
        self._file_path = the_info.file_path
        self._file_prefix = the_info.file_prefix
        self._db_source = the_info.db_source
        self._logger = logger
        self._select_condition = the_info.select_condition
        self._sql_condition = the_info.sql_condition
        self._sq = SQLTools(proj_name=self._proj_name,
                            logger=self._logger, run_by_airflow=False)
        self._file_path = self.create_path()

    def create_path(self):
        the_path = self._file_path
        if not os.path.exists(the_path):
            os.makedirs(the_path)
            print(f"U just created {the_path}")
        else:
            print(f"Do nothing. U have {the_path}.")
        return the_path


    def _get_sql(self):

        if self._db_source == 'rawdata':
            from_sql = f"""FROM mlaas_limit.{self._table_name}"""

        else:
            from_sql = f"""FROM {self._proj_name}.{self._table_name}"""

        sql_statement = f'''
            {self._select_condition}
            {from_sql}
        '''

        if self._sql_condition:
            sql_statement = sql_statement + f'''{self._sql_condition}'''

        if self._size_limit:
            sql_statement = sql_statement + f'LIMIT {self._size_limit};'
        print (sql_statement)
        return sql_statement

    @property
    def table_name(self):
        return self._table_name

    def get_db_df(self):
        """
            get dataframe from db
        """  
        select_sql = self._get_sql()
        result = self._sq.query2dataframe(
            select_sql=select_sql,
            db_id=self._db_source,
            output_type='pandas'
        )
        return result

    @property
    def gen_filename(self):
        file_template = f"{self._file_prefix}_{self._table_name}"
        if self._size_limit:
            return os.path.join(self._file_path, f"{file_template}_limit{self._size_limit}")
        else:
            return os.path.join(self._file_path, f"{file_template}")
            

    def create_csv_from_df(self, mod_df=None):
        '''
            create csv file using pandas dataframe
        '''
        cond = isinstance(mod_df, pd.DataFrame)

        df = mod_df if cond else self.get_db_df()
        df.to_csv(f"{self.gen_filename}.csv", sep=',',
                  encoding='utf-8', index=False)
        print(f"save file: {self.gen_filename}")


def create_csv(t: TheInfo):
    a = GenCSVfromDB(t, logging)
    print(a.create_csv_from_df())

def create_csv_with_mod_address(t: TheInfo):
    a = GenCSVfromDB(t, logging)
    df = a.get_db_df()
    df['mod_address'] = df['address'].copy()
    df.loc[df['mod_address'].str.strip() == '', 'mod_address'] = df['cmp_address']
    a.create_csv_from_df(mod_df=df)
    return df


if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)
    print("hello from main")

    the_rawdata = TheInfo(proj_name='social', 
                table_name='bpm_od_moea_cmpinfo_socialnetwork_info', 
                size_limit=5, 
                file_path=f"{cwd}", 
                file_prefix=f'social_addr_v4',
                db_source="rawdata",
                sql_condition=None,
                select_condition="SELECT address, busi_name, busi_project, cmp_address"
                )

    the_df = create_csv_with_mod_address(the_rawdata)

    # print(the_df.describe())
# for series_name, series in df.items():
    print(the_df.columns)
    for series_name, series in the_df.loc[:, ['address', 'cmp_address', 'mod_address']].items():
        print(series_name, series)
        # i_out = trans_address(k)
        # i_out.update({"mod_address": k})
        # print(i_out)

# df[ (df[column_name].notnull()) & (df[column_name]!=u'') ].index



    # df.replace("", "empty_st", inplace=True)

    # display(df)
    # create_csv(t3)
    # 
    #   