"""
TODO:
- [X] 增加檢查產生行數是否正常的功能:
    - In run,
        - 先把資料存到暫存表中，
        - 檢查暫存表行數是否和上次的最終表接近，並記錄本次產出行數含上次產出行數
        - 若正常，truncate上次的最終表
        - 把暫存表Insert到最終表
        - truncate暫存表
    - setup要考慮暫存表
    - 加入暫存表的drop和truncate功能

"""
import abc
from src.common.etl_base import ETLBase
import logging


class SQLBase(ETLBase):
    """
    來源與終點表同在feature DB的ETL，可以繼承此物件進行實作
    """

    def __init__(self,
                 target_table_name):
        self.__target_table_name = target_table_name

    @property
    def target_table_name(self):
        return self.__target_table_name

    @abc.abstractmethod
    def get_target_create_SQL(self, schema_name, table_name):
        """
        Args:
            - schema_name: 'cc_ln_pre_approval'
            - table_name:  最終結果表的名稱
        Return:
            - (str) 創建最終結果表的SQL

        Example:

        return f'''
            CREATE TABLE
            IF NOT EXISTS {schema_name}.{table_name}(
            cust_no CHAR(24),
            "group" CHAR(2),
            product CHAR(2),
            apdlv NUMERIC,
            lgd NUMERIC,
            base_int NUMERIC,
            profit_int NUMERIC,
            pre_net_income BIGINT,
            max_limit INT,
            interest_rate_1 NUMERIC,
            period_1 INT,
            interest_rate_2 NUMERIC,
            period_2 INT,
            fee_amount INT,
            all_rate NUMERIC,
            list_name CHAR(20),
            data_dt DATE,
            etl_dt TIMESTAMP,
            Primary key(cust_no, product)
            );'''

        """
        pass

    @abc.abstractmethod
    def get_process_SQL(self, schema_name, target_table_name):
        """
        Args:
            - schema_name: 'cc_ln_pre_approval'

            - target_table_name: 輸出表的名稱 (e.g., 'final_cust_result' or 'esun_cust_loan_preapproval_data')
        Return:
            - (str) Select並且Insert進db的SQL

        Example:

        return f'''

            INSERT INTO {schema_name}.{target_table_name}
            SELECT * FROM {schema_name}.cust_tmp_result

        '''
        """
        pass

    @abc.abstractmethod
    def check_target_table(self, schema_name, table_name):
        """
        檢查db中結果表的正確性

        Args:
            - schema_name: 結果表schema名稱
            - table_name: 結果表名稱
        Return:
            - result (bool): 成功與否
            - row_count (int): 結果表行數
        Example:
            sqlstring = f'''
            SELECT
            *
            FROM {schema_name}.{table_name}
            '''
            table = self.select_table('feature', sqlstring)
            assert all(table['group'].isin(['01', '02', '03']))
            return True
        """
        pass

    def check_target(self):
        """
        執行結果表的正確性檢查
        """
        result, row_count = self.check_target_table(
            self.schema_name, self.__target_table_name)
        print('SUCCESS')
        logging.info('[check_target] SUCCESS')
        logging.info(f'[check_target] TARGET ROW COUNT: {row_count}')
        return result

    def setup(self):
        self.__create_target_table()
        self.__truncate_target_table()
        self.__grant_target_table()

    def drop_target_table(self):
        sqlstring = f'''
            DROP TABLE IF EXISTS
            {self.schema_name}.{self.__target_table_name};
        '''
        self.execute_sql('feature', sqlstring)

    def run_etl(self):
        self.__create_target_table()
        self.__truncate_target_table()
        sql = self.get_process_SQL(
            self.schema_name,
            self.__target_table_name
        )
        self.execute_sql('feature', sql)

    def __grant_target_table(self):
        for user in self.db_users:
            sqlstring = f"""
                GRANT ALL
                ON TABLE {self.schema_name}.{self.__target_table_name}
                TO {user};
            """
            self.execute_sql('feature', sqlstring)

    def __create_target_table(self):
        sqlstring = self.get_target_create_SQL(
            self.schema_name,
            self.__target_table_name
        )
        self.execute_sql('feature', sqlstring)

    def __truncate_target_table(self):
        sqlstring = f'''
            TRUNCATE TABLE
            {self.schema_name}.{self.__target_table_name};
        '''
        self.execute_sql('feature', sqlstring)

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
            run_etl_op = PythonOperator(
                task_id='run_etl',
                python_callable=self.run_etl,
                dag=dag
            )

            check_target_op = PythonOperator(
                task_id='check_target',
                python_callable=self.check_target,
                dag=dag
            )
            run_etl_op >> check_target_op
        return task_group

    def run(self):
        self.run_etl()
        self.check_target()
