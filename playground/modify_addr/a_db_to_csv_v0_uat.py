from easy_to_sql.sql_tools import SQLTools
from dataclasses import dataclass
import os
import logging
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

        selected_sql = f'''
            SELECT *
            {from_sql}
        '''

        if self._sql_condition:
            selected_sql = f'''
                SELECT *
                {from_sql}
                {self._sql_condition}
            '''

        if self._size_limit:
            selected_sql = selected_sql + f'LIMIT {self._size_limit};'
        print (selected_sql)
        return selected_sql

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
            

    def create_csv_from_df(self):
        '''
            create csv file using pandas dataframe
        '''
        df = self.get_db_df()
        df.to_csv(f"{self.gen_filename}.csv", sep=',',
                  encoding='utf-8', index=False)
        print(f"save file: {self.gen_filename}")


def create_csv(t: TheInfo):
    a = GenCSVfromDB(t, logging)
    print(a.create_csv_from_df())

if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)
    print("hello from main")

    # xx = 'B02'
    for xx in ['B05', 
        'B10', 'B11', 
        'H03', 'K02', 'K04', 'K10', 
        'X01', 'X06', 'X09', 'X21',
        "K01",
        "G0700001"
    ]:
        t3 = TheInfo(proj_name='pl_automated_valuation', 
                    table_name='poi_service', 
                    size_limit=3, 
                    file_path=f"{cwd}", 
                    file_prefix=f'avm_uat_{xx}',
                    db_source= "feature",
                    sql_condition= f"""WHERE left(type_code, 3) = '{xx}'
                                        OR left(type_code, 8) = '{xx}'
                    """
                    ) 
        create_csv(t3)    
