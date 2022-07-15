"""
# TODO:
- [ ] In select_n_insert, check if len(self.tmp_table) is close
        to the row count of tmp_table saved last month in Feature DB.
        (Save log of abnormal source table!)
- [ ] Allow 卡處 to determine an abnormal threshold
- [X] In prepare, do not delete tmp_table saved last month
"""
import abc
import importlib
import os
import logging
import sys
# XXX: [ ] importlib -> plain import
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
ETLBase = getattr(importlib.import_module('src.common.etl_base'), 'ETLBase')
TABLE_SUBFIX = getattr(importlib.import_module('src.config'), 'TABLE_SUBFIX')

ALIAS = '_alias' + TABLE_SUBFIX


class TableETLBase(ETLBase):
    """
    Raw -> Feature 用的 ETL物件
    """

    def __init__(self, alias=ALIAS):
        self.alias = alias

    @property
    @abc.abstractmethod
    def source_table_name(self):
        """
        設定來源表格名稱
        (需覆寫此函數，針對來源資料表客製化)

        Example:

        @property
        def source_table_name(self):
            return "WITWO371"
        """
        return "table_name"

    @property
    def tmp_table_name(self):
        return self.source_table_name + self.alias

    @property
    @abc.abstractmethod
    def tmp_column_types(self):
        """
        設定暫存資料表(tmp)欄位規格
        (需覆寫此函數，針對暫存資料表客製化)

        Example:

        @property
        def tmp_column_types(self):
            return {
                'column1': 'char(1)',
                'column2': 'char(1)'
            }
        """
        pass

    @property
    @abc.abstractmethod
    def tmp_column_defaults(self):
        """
        設定暫存資料表(tmp)欄位彙總後，遇到空值時要補的值(default)。
        (需覆寫此函數，針對暫存資料表客製化)

        Example:

        @property
        def tmp_column_types(self):
            return {
                'column1': "'N'",
                'column2': "'N'"
            }

        此設定將被轉匯成以下SQL:

        ...
        CASE
            WHEN column1 IS NULL THEN 'N'
            ELSE column1
        END
        CASE
            WHEN column2 IS NULL THEN 'N'
            ELSE column2
        END
        ...
        """
        pass

    @property
    @abc.abstractmethod
    def etlSQL(self):
        """
        ETL的SQL於此定義

        Example
        return f'''
        SELECT
            cust_no,
            pd_value,
            pd_grade,
            lgd
        FROM cc_ln_pre_approval.{self.source_table_name}
        '''
        """
        pass

    @property
    @abc.abstractmethod
    def source_db(self):
        """
        來源表的DB: feature or rawdata

        Example:
        return 'feature'
        """
        pass

    def select_n_insert(self):
        self.__create_tmp_in_feature_db()
        self.truncate_tmp_from_feature_db()
        self.select_n_insert_tool(
            self.etlSQL, self.source_db,
            self.schema_name, self.tmp_table_name)
        logging.info('[select_n_insert] SUCCESS')

    @abc.abstractmethod
    def check_source(self):
        """
        此函數會對rawdata_db中的來源表進行檢查。
        (需覆寫此函數，針對來源表客製化)

        範例:
        def check_source(self):
            sqlstring = f'''
                SELECT
                cust_id as cust_no,
                cust_grp_code
            FROM mlaas_rawdata.{self.source_table_name}
            '''
            table = self.select_table('rawdata', sqlstring)

            assert 'K' in table.cust_grp_code.unique()
            assert 'M' in table.cust_grp_code.unique()
            assert 'N' in table.cust_grp_code.unique()
            assert 'O' in table.cust_grp_code.unique()
            """
        pass

    def create_tmp(self):
        self.__create_tmp_in_feature_db()
        logging.info('[create_tmp] SUCCESS')

    def grant_tmp(self):
        for user in self.db_users:
            sqlstring = f"""
                GRANT ALL
                ON TABLE {self.schema_name}.{self.tmp_table_name}
                TO {user};
            """
            print(sqlstring)
            logging.info(sqlstring)
            self.execute_sql('feature', sqlstring)
            logging.info('[grant_tmp] SUCCESS')

    def show_tmp_in_feature_db(self):
        sqlstring = f"""
        SELECT *
        FROM {self.schema_name}.{self.tmp_table_name}
        """
        return self.select_table('feature', sqlstring)

    def check_tmp(self):
        sqlstring = f'''
        select count(cust_no)
        from {self.schema_name}.{self.tmp_table_name}
        '''
        logging.info(f'[check_tmp] sql: \n {sqlstring}')
        row_count = self.select_table('feature', sqlstring)['count'].values[0]
        logging.info(f'[check_tmp] ROW COUNT: {row_count}')
        assert row_count > 0

    @abc.abstractmethod
    def check_target_columns(self, target_table_name):
        """
        檢查一個來源表於顧客資料彙總表所對應的欄位是否正確。
        (需覆寫此函數，針對顧客資料彙總表客製化)

        Example:
        def check_target_columns(self, target_table_name):
            sqlstring = f'''
                SELECT
                cust_no,
                world_card,
                travel_card,
                five_profession_card
            FROM {self.schema_name}.{target_table_name}
            '''
            table = self.select_table('feature', sqlstring)

            assert all(table.world_card.isin(['Y', 'N']))
            assert all(table.travel_card.isin(['Y', 'N']))
            assert all(table.five_profession_card.isin(['Y', 'N']))

        """
        pass

    def truncate_tmp_from_feature_db(self):
        self.__create_tmp_in_feature_db()
        sqlstring = f"""
        TRUNCATE TABLE {self.schema_name}.{self.tmp_table_name};
        """
        logging.info(f'[truncate_tmp_from_feature_db] {sqlstring}')
        self.execute_sql('feature', sqlstring)
        logging.info('[truncate_tmp_from_feature_db] SUCCESS')

    def delete_tmp_from_feature_db(self):
        sqlstring = f"""
        DROP TABLE IF EXISTS {self.schema_name}.{self.tmp_table_name};
        """
        logging.info(f'[delete_tmp_from_feature_db] {sqlstring}')
        self.execute_sql('feature', sqlstring)
        logging.info('[delete_tmp_from_feature_db] SUCCESS')

    def __get_tmp_table_cols(self):
        """
        把中繼表的欄位依順序列出來
        (使要存入的表可以是正確的順序)
        """
        sqlstring = f'''
            SELECT *
            FROM {self.schema_name}.{self.tmp_table_name}
            LIMIT 0
        '''
        cols = self.select_table('feature', sqlstring).columns.tolist()
        logging.info(
            '[__get_tmp_table_cols] Get Column Orders in DB TMP table')
        return cols

    def __create_tmp_in_feature_db(self):
        assert len(self.tmp_column_types) == len(self.tmp_column_defaults)
        if len(self.tmp_column_types) > 0:
            col_assign_sql = ',\n'.join(
                [f'''
                {col_name} {type_str}
                ''' for col_name, type_str in self.tmp_column_types.items()])
            sqlstring = f"""
            CREATE TABLE
            IF NOT EXISTS {self.schema_name}.{self.tmp_table_name}(
                cust_no char(24),
                {col_assign_sql},
                Primary key(cust_no)
            );
            """
        else:
            sqlstring = f"""
            CREATE TABLE
            IF NOT EXISTS {self.schema_name}.{self.tmp_table_name}(
                cust_no char(24),
                Primary key(cust_no)
            );
            """
        logging.info(f'[__create_tmp_in_feature_db] sql: \n{sqlstring}')
        self.execute_sql('feature', sqlstring)
        logging.info('[__create_tmp_in_feature_db] SUCCESS')

    def build_task_group(self, group_id, dag=None):
        """
        Args:
            - group_id: (str) name of the group
        Return:
            - task_group: (TaskGroup) 回傳的tasks group
        """
        from airflow.utils.task_group import TaskGroup
        from airflow.operators.python import PythonOperator
        with TaskGroup(group_id=group_id) as task_group:
            check_op = PythonOperator(
                task_id='check_source',
                python_callable=self.check_source,
                dag=dag
            )
            select_n_insert_op = PythonOperator(
                task_id='select_n_insert',
                python_callable=self.select_n_insert,
                dag=dag
            )
            check_tmp_op = PythonOperator(
                task_id='check_tmp',
                python_callable=self.check_tmp,
                dag=dag
            )
            check_op >> select_n_insert_op >> check_tmp_op
        return task_group

    def run(self):
        # 檢查rawdataDB的來源表
        self.check_source()
        # 以SQL將來源表進行Select並轉換為DM欄位
        self.select_n_insert()
        # 檢查中繼表
        self.check_tmp()
