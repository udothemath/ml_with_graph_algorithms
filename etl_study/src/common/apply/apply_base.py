"""
This object apply row-by-row function in logic_base.py to a table.
"""
import psutil
import signal
import os
from importlib import import_module
import abc
import pandas as pd
from functools import partial
from airflow.operators.bash import BashOperator
from datetime import timedelta
from gevent import Timeout
import gevent
import gc
import random
import logging
from src.config import SHORT_TIMEOUT
from src.config import MAX_BATCH
from src.config import BATCH_SIZE
from src.config import SHORT_TIMEOUT_MINUTES
from src.config import INSERT_PER_TASK
from src.common.etl_base import ETLBase


class WaitTooLong(Exception):
    """Raise when a method takes too long to complete"""
    pass


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ApplyBase(ETLBase):
    '''
    本類別提供方法進行db->db的ETL，可以把一個資料表batch-by-batch拉出為pd.DataFrame，
    於此pd.DataFrame 進行row-by-row 的 function apply後，batch-by-batch存入db的一個資料表。
    '''

    def __init__(self, input_db, input_table_name, target_table_name,
                 input_schema_name, target_schema_name='cc_ln_pre_approval',
                 set_index=None, verbose=False, max_batch=MAX_BATCH,
                 batch_size=BATCH_SIZE, parallel_cnt=1):
        '''
        Args:
            - input_table_name: input表名稱
            - target_table_name: 產出表名稱
            - input_schema_name: input表的schema名稱
            - target_schema_name: 產出表schema名稱 (預設為專案名稱)
            - set_index: 是否需要把特定欄位設為選取之pd.DataFrame之index，設定後function apply的過程
                即可不考慮此欄位。None代表不需要。 (e.g., set_index = 'cust_no')
            - verbose = True | False
        '''
        self.__func = None
        self.__input_db = input_db
        self.__input_table_name = input_table_name
        self.__input_schema_name = input_schema_name
        self.__target_table_name = target_table_name
        self.__target_schema_name = target_schema_name
        self.__set_index = set_index
        self.__verbose = verbose
        self.__tqdm_verbose = False
        self.__max_batch = max_batch
        self.__batch_size = batch_size
        self.__parallel_cnt = parallel_cnt

    def set_parallel_cnt(self, parallel_cnt):
        self.__parallel_cnt = parallel_cnt

    @property
    def __parallel_func(self):
        if self.__func is None:
            self.__func = self.logic_object().process_df
            return self.__func
        else:
            return self.__func

    @property
    @abc.abstractmethod
    def logic_object(self):
        """
        引入要套用的邏輯物件 (繼承LogicBase)

        Example:

        from src.process.cust_logic import CustLogic
        return CustLogic
        """
        pass

    @property
    def input_table_name(self):
        return self.__input_table_name

    @property
    def target_table_name(self):
        return self.__target_table_name

    def setup(self):
        self.__create_target_table()
        self.__truncate_target_table()
        self.__grant_target_table()
        logging.info('SUCCESS')
        return True

    def __grant_target_table(self):
        for user in self.db_users:
            sqlstring = f"""
                GRANT ALL
                ON TABLE {self.__target_schema_name}.{self.__target_table_name}
                TO {user};
            """
            logging.info(sqlstring)
            self.execute_sql('feature', sqlstring)

    def target_create_SQL(self):
        return self.get_target_create_SQL(
            self.__target_schema_name,
            self.__target_table_name
        )

    @abc.abstractmethod
    def get_target_create_SQL(self, schema_name, table_name):
        """
        產生建立產出表的SQL

        Args:
            - schema_name: 產出表schema名稱
            - table_name: 產出表名稱
        Returns:
            - sqlstring: 產製的SQL

        Example:

        sqlstring = f'''
            CREATE TABLE
            IF NOT EXISTS {schema_name}.{table_name}(
            cust_no CHAR(24),
            pd_value NUMERIC,
            pd_grade INT,
            lgd NUMERIC,
            Primary key(cust_no)
            );
        return sqlstring
        '''
        """
        pass

    def partial_select_SQL(self, offset, size):
        return self.get_partial_select_SQL(
            self.__input_schema_name,
            self.__input_table_name,
            offset,
            size
        )

    @abc.abstractmethod
    def get_partial_select_SQL(self, schema_name, table_name, offset, size):
        """
        產生batch-by-batch Select時，所用的SQL

        Args:
            - schema_name: input表schema名稱
            - table_name: input表名稱
            - offset: Select的起始行數
            - size: batch的大小
        Returns:
            - select_sql: 選表的SQL

        Example:
            select_sql = f'''
                SELECT
                    cust_id AS cust_no,
                    input_data
                FROM {schema_name}.{table_name}
                LIMIT {size}
                OFFSET {offset}
                '''
            return select_sql
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
            print('SUCCESS')
            return True, len(table)
        """
        pass

    def check_target(self):
        """
        執行結果表的正確性檢查
        """
        _, target_size = self.check_target_table(
            self.__target_schema_name, self.__target_table_name)
        logging.info('[check_target] CHECK TARGET SUCCESS')
        logging.info(f'[check_target] TARGET ROW COUNT: {target_size}')
        if self.__max_batch is None:
            input_size = self.check_input_size(
                self.__input_db, self.__input_schema_name, self.input_table_name)
            logging.info(f'[check_target] INPUT ROW COUNT: {input_size}')
            assert target_size == input_size
            logging.info(
                '[check_target] TARGET ROW COUNT IS CORRECT (equals INPUT ROW COUNT)')
        else:
            assert target_size == self.__max_batch * self.__batch_size
            logging.info(
                '[check_target] TARGET ROW COUNT IS CORRECT (equals MAX_BATCH)')
        return True

    def __create_target_table(self):
        """
        建立空的產出表
        """
        sqlstring = self.get_target_create_SQL(
            self.__target_schema_name,
            self.__target_table_name)
        self.execute_sql('feature', sqlstring)
        logging.info('[__create_target_table] SUCCESS')

    def __truncate_target_table(self):
        """
        清空產出表
        """
        sqlstring = f"""
            TRUNCATE TABLE {self.__target_schema_name}.{self.__target_table_name};
        """
        self.execute_sql('feature', sqlstring)
        logging.info('[__truncate_target_table] SUCCESS')

    def drop_target_table(self):
        """
        刪除產出表

        Returns:
            - row_count: 行數
        """
        sqlstring = f"""
            DROP TABLE IF EXISTS {self.__target_schema_name}.{self.__target_table_name};
        """
        self.execute_sql('feature', sqlstring)

    def run(self, parallel_count=16, insert_per_task=INSERT_PER_TASK):
        """
        進行整張表的ETL

        Args:
            - parallel_count: 平行進程的數量
        """
        try:
            if self.__tqdm_verbose:
                tqdm = import_module('tqdm')
            # 因為tqdm沒有在 airflow BaseImage 中裝，為了避免實例化時發生問題，
            # 需要把這個package改成在函數裡面進行import。
            self.__create_target_table()
            self.__truncate_target_table()
            whole_input_size = self.check_input_size(
                self.__input_db, self.__input_schema_name, self.input_table_name)
            offset_list = list(range(0, whole_input_size, self.__batch_size))
            if self.__max_batch is not None:
                assert isinstance(self.__max_batch, int)
                offset_list = offset_list[:self.__max_batch]
            logging.info(f'[run] PID of Main Process: {os.getpid()}')
            logging.info(f'[run] BATCH SIZE: {self.__batch_size}')
            logging.info(f'[run] PARALLEL COUNT: {parallel_count}')
            logging.info(f'[run] OFFSET: {offset_list[0]}~{offset_list[-1]}')
            result_gen = map(
                partial(
                    self.run_partially,
                    size=self.__batch_size),
                offset_list)
            string_io_gen = self.string_io_generator(
                result_gen,
                num_batch_per_insert=len(offset_list) // insert_per_task)
            insert_gen = map(
                partial(
                    self.insert_table_from_string_io,
                    table_name=self.__target_table_name
                ), string_io_gen)
            if self.__tqdm_verbose:
                _ = list(
                    tqdm.tqdm(insert_gen,
                              total=int(
                                  whole_input_size / self.__batch_size),
                              desc='TOP:'
                              ))
            else:
                _ = list(insert_gen)
            logging.info('[run] FINISH PARALLEL RUN')
        except BaseException as e:
            logging.exception('[run] ERROR RAISED IN PARALLEL RUN')
            raise e
        finally:
            logging.info('[run] FINAL FINISH')

    def __on_process_init(self, *args, **kargs):
        pid = os.getpid()
        logging.info(f'[on_process_init ({pid})] start process')
        logging.info(f'[on_process_init ({pid})] args {args}')
        logging.info(f'[on_process_init ({pid})] kargs {kargs}')

    def __on_process_exit(self, pid, exitcode):
        self.close_conn()
        logging.info(f'[on_process_exit ({pid})] exitcode: {exitcode}')
        if exitcode == 1:
            self.__close_children_processes('child')
            raise ValueError(f'[on_process_exit ({pid})] raise error in child')

    def __close_children_processes(self, mode):
        if mode == 'child':
            pid = os.getppid()
        elif mode == 'parent':
            pid = os.getpid()
        children = psutil.Process(pid).children(recursive=True)
        for child in children:
            try:
                os.kill(child.pid, signal.SIGINT)
                logging.info(
                    f'[run ({pid})] child processes {child.pid} interrupt')
            except OSError:
                logging.info(f'[run ({pid})] {child} not interrupt')

    def check_input_size(self, input_db, schema_name, table_name):
        """
        檢查input表的大小

        若要根據分流的情形客製化，請Overide此方法

        Args:
            - input_db: 輸入表db (feature | rawdata)
            - schema_name: 輸入表schema (cc_ln_pre_approval)
            - table_name: 輸入表名稱 (__init__定義的input_table_name)
        Returns:
            - row_count: 行數
        """
        return self.__check_input_size(input_db, schema_name, table_name)

    def __check_input_size(self, input_db, schema_name, table_name):
        """
        檢查input表的大小

        (* 沒有做分流的時候預設用這個)

        Args:
            - input_db: 輸入表db (feature | rawdata)
            - schema_name: 輸入表schema (cc_ln_pre_approval)
            - table_name: 輸入表名稱 (__init__定義的input_table_name)
        Returns:
            - row_count: 行數
        """
        sql = f"""
        SELECT count(*)
        FROM {schema_name}.{table_name}
        """
        logging.info(f'[__check_input_size] count SQL: \n{sql}')
        result_table = self.select_table(input_db, sql)
        row_count = result_table['count'][0]
        logging.info(f'[__check_input_size]: {row_count}')
        return row_count

    def run_partially(self, offset, size):
        """
        處理片段的資料表

        Args:
            - offset: 片段資料表的起始行數
            - size: batch的大小
        """
        try:
            with Timeout(2400, WaitTooLong):
                pid = os.getpid()
                if offset / size < self.__parallel_cnt:
                    gevent.sleep(random.uniform(0, self.__parallel_cnt))
                    logging.info(
                        f'[run_partially ({pid})] START PROCESS OFFSET: {offset} FOR BATCHSIZE: {size}')
                if self.__verbose:
                    logging.info(
                        f'[run_partially ({pid})] 1: START PROCESS TABLE FROM OFFSET: {offset} FOR BATCHSIZE: {size}')
                input_table = self.__read_df_partially(
                    offset, size
                )
                if self.__set_index:
                    assert isinstance(self.__set_index, str)
                    input_table.set_index(self.__set_index, inplace=True)
                if self.__verbose:
                    logging.info(
                        f'[run_partially ({pid})] 2: Row Count after Select: {len(input_table)}')
                with Timeout(1440, WaitTooLong):
                    parallel_func = self.__parallel_func
                    result_table = parallel_func(input_table)
                if self.__verbose:
                    logging.info(
                        f'[run_partially ({pid})] 3: Row Count after Transform: {len(result_table)}')
                if self.__verbose:
                    process = psutil.Process(pid)
                    gb_usage = process.memory_info().rss / (10**9)
                    logging.info(
                        f'[run_partially ({pid})] MEMORY USAGE: {gb_usage}')
                del input_table
                gc.collect()
                if self.__verbose:
                    logging.info(
                        f'[run_partially ({pid})] 4: Delete Input Table')

                if self.__set_index:
                    result_table.reset_index(inplace=True)
                if self.__verbose:
                    logging.info(
                        f'[run_partially ({pid})] 5: DONE PROCESS TABLE WITH OFFSET: {offset}')

                return result_table
        except BaseException as e:
            # Error Handling:
            pid = os.getpid()
            error_message = f'[run_partially ({pid})] X: Error Happend at Offset {offset} of Batch_size {size} : \n{e}'
            logging.exception(error_message)
            # Raise Error
            raise e

    def read_df_partially(self, offset, size):
        return self.__read_df_partially(offset, size)

    def __read_df_partially(self, offset, size):
        """
        Batch-wise Select input表為pd.DataFrame

        Args:
            - offset: 選表的起始行數
            - size: batch的大小
        Returns:
            - input_table: 選出來的片段pd.DataFrame
        """
        with Timeout(15, WaitTooLong):
            select_sql = self.get_partial_select_SQL(
                self.__input_schema_name,
                self.__input_table_name,
                offset,
                size
            )
            input_table = self.__select_table_wo_conn_closed(
                self.__input_db, select_sql)
            return input_table

    def __select_table_wo_conn_closed(self, db, sql):
        return pd.read_sql(sql, self.open_conn(db))

    def __check_target_size(self):
        """
        檢查最終產出的資料表行數

        Returns:
            - row_count: 行數
        """
        sql = f"""
        SELECT count(*)
        FROM {self.__target_schema_name}.{self.__target_table_name}
        """
        conn = self.get_conn('feature', self.__target_schema_name)
        result_table = pd.read_sql(sql, conn)
        row_count = result_table['count'][0]
        return row_count

    @staticmethod
    def build_bash_operator(
            apply_module, apply_class,
            task_id, dag, group_id=-1, custom_timeout_minutes=24 * 60):
        """
        Args:
            - apply_module: e.g., 'src.pd_lgd.model_apply'
            - apply_class:  e.g., 'ModelApply'
            - input_table_name: e.g., 'upl_pd_lgd_datamart_json_airflow'
            - target_table_name: e.g., 'upl_pd_lgd_result_airflow'
            - task_id: dag task id
                e.g., f'run_model_apply_pd_lgd'
            - dag: airflow dag
            - cust_no_group: 分流的編號，若不做分流給-1
        Returns:
            - bash_operator (BashOperator)

        """
        from kubernetes.client import models as k8s
        from src.common.apply import image_config
        CODE_PATH = "/app"
        apply_bash_task = BashOperator(
            task_id=task_id,
            bash_command=f'whoami; cd {CODE_PATH}; ls -al;'
            f'python3 src/common/apply/apply_bash.py'
            f' --apply_module {apply_module}'
            f' --apply_class {apply_class}'
            f' --group_id {group_id}'
            ' || exit 1',
            execution_timeout=timedelta(
                minutes=SHORT_TIMEOUT_MINUTES) if SHORT_TIMEOUT else timedelta(
                minutes=custom_timeout_minutes),
            retries=0 if SHORT_TIMEOUT else 3,
            retry_delay=timedelta(minutes=1),
            executor_config={
                'pod_override': k8s.V1Pod(
                    spec=k8s.V1PodSpec(
                        containers=[
                            k8s.V1Container(
                                name="base",
                                image=image_config.IMAGE,
                                image_pull_policy="Always"
                            )
                        ]
                    )
                )
            },
            dag=dag
        )
        return apply_bash_task
