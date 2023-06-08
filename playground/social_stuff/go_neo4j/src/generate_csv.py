from easy_to_sql.sql_tools import SQLTools
from src.utils import wrap_log
from src.table_info import FileInfo


class GenCSVfromDB:
    def __init__(self, file_info: FileInfo, logger):
        self._proj_name = 'socialnetwork_info'
        self._file_info = file_info
        self._logger = logger
        self._sq = SQLTools(proj_name=self._proj_name,
                            logger=self._logger, run_by_airflow=False)

    def get_db_df(self):
        select_sql = f'''
        SELECT *
        FROM {self._proj_name}.{self._file_info.table_name}
        '''
        if self._file_info.size_limit:
            size_limit = self._file_info.size_limit
            select_sql = select_sql + f'LIMIT {size_limit}'

        result = self._sq.query2dataframe(
            select_sql=select_sql,
            db_id='feature',
            output_type='pandas'
        )
        return result

    def create_csv_from_df(self):
        df = self.get_db_df()
        df.to_csv(self._file_info.get_path, sep=',',
                  encoding='utf-8', index=True)
        print(f"filename: {self._file_info.get_path}")
