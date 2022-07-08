"""
- TODO:
- [X] 讓insert可以累積一段的行數以後，再存進DB裡面。
    - [X] 研究StringIO是否可以一直累加上去 -> 可以
    - [X] run裡面的insert map，看怎麼樣能夠在吃了一定筆數時，
            可以去統整出一批的行為StringIO中的CSV，再存進DB中。
        - [X] 有一個map專門產出StringIO (蒐集多個Pandas的CSV於內存)
        - [X] 有一個map專門把StringIO中的CSV從內存送到DB。
    - [X] 放置研究結果於reference。
    - [X] 從DEVELOP checkout出來實作此方法。
"""
import pandas as pd
import psycopg2
import io
from io import StringIO
from src import config
from contextlib import closing
import logging
import os
import gc
# from mlaas_tools.config_build import config_set
# config_set()
# remove the above two lines once config.ini is generated
VERBOSE = False
CLOSE_ON_DEL = False
NUM_BATCH_PER_CONN = 10
PROCESS_CONN = {}
CONN_CNT = {}


class ConnHandler(closing):
    def __init__(self, thing):
        super(ConnHandler, self).__init__(thing)
        pid = os.getpid()
        logging.info(f'[ConnHandler ({pid})] OPEN')

    def __exit__(self, *exc_info):
        self.thing.close()
        pid = os.getpid()
        logging.info(f'[ConnHandler ({pid})] CLOSE')


class ETLBase:
    def __init__(self):
        pass

    @property
    def db_users(self):
        return config.DB_USERS

    @property
    def schema_name(self):
        return config.SCHEMA_NAME

    def execute_sql(self, db, sql):
        with ConnHandler(self.get_conn(db, self.schema_name)) as conn:
            try:
                cur = conn.cursor()
                cur.execute(sql)
                logging.info(f'EXECUTE SQL:\n{sql}')
                conn.commit()
                cur.close()
            except (Exception, psycopg2.DatabaseError) as error:
                conn.rollback()
                cur.close()
                logging.info('EXECUTE SQL FAIL')
                raise error
        logging.info('EXECUTE SQL SUCCESS')

    def select_n_insert_tool(self, etlSQL, source_db,
                             target_schema, target_table):
        with ConnHandler(self.get_conn(source_db, self.schema_name)) as source_conn, ConnHandler(self.get_conn('feature', self.schema_name)) as target_conn:
            in_memory_file = io.BytesIO()
            source_copy_sql = """COPY (%s) TO STDOUT DELIMITER ',' CSV""" % etlSQL
            table_name = target_schema + "." + target_table
            copy_query = """COPY %s FROM STDIN WITH (FORMAT CSV, DELIMITER ',', HEADER FALSE)""" % table_name

            try:
                with source_conn.cursor() as raw_cur:
                    logging.info("Start getting data from source db")
                    raw_cur.copy_expert(source_copy_sql, in_memory_file)
                    in_memory_file.seek(0)
                    logging.info("Done getting data")
                    try:
                        with target_conn.cursor() as ftr_cur:
                            logging.info(
                                "Start insert data into target (feature) db")
                            ftr_cur.copy_expert(copy_query, in_memory_file)
                        target_conn.commit()
                        logging.info("Done inserting data")
                    except Exception:
                        target_conn.rollback()
                        raise
            except Exception:
                source_conn.rollback()
                raise Exception

    def select_table(self, db, sql):
        with ConnHandler(self.get_conn(db, self.schema_name)) as conn:
            df = pd.read_sql(sql, conn)
        return df

    def open_conn(self, db):
        global PROCESS_CONN
        global CONN_CNT
        pid = os.getpid()
        if pid in PROCESS_CONN:
            conn = PROCESS_CONN[pid]
            if VERBOSE:
                logging.info(f'[open_conn ({pid})] GET')
        else:
            conn = self.get_conn(db, self.schema_name)
            logging.info(f'[open_conn ({pid})] OPEN')
            PROCESS_CONN[pid] = conn
        if pid in CONN_CNT:
            CONN_CNT[pid] += 1
        else:
            CONN_CNT[pid] = 1
        return conn

    def close_conn(self):
        global PROCESS_CONN
        pid = os.getpid()
        if pid in PROCESS_CONN:
            PROCESS_CONN[pid].close()
            del PROCESS_CONN[pid]
            logging.info(f'[close_conn ({pid})] DONE')
        else:
            logging.info(
                f'[close_conn ({pid})] NOTHING TO CLOSE: {PROCESS_CONN}')

    def __del__(self):
        global CONN_CNT
        global CLOSE_ON_DEL
        if CLOSE_ON_DEL:
            pid = os.getpid()
            if (pid in CONN_CNT) and (
                CONN_CNT[pid] % NUM_BATCH_PER_CONN == (
                    NUM_BATCH_PER_CONN - 1
                )
            ):
                self.close_conn()

    def select_table_fast(self, db, sql, encoding='utf-8', index=False):
        in_memory_csv = StringIO()
        with ConnHandler(self.get_conn(db, self.schema_name)) as conn:
            cursor = conn.cursor()
            cursor.copy_expert(
                f"COPY ({sql}) TO STDOUT DELIMITER ',' CSV HEADER", in_memory_csv)
            in_memory_csv.seek(0)
            conn.commit()
            cursor.close()
            result = pd.read_csv(
                in_memory_csv,
                sep=',',
                header='infer',
                encoding=encoding)
        return result

    def insert_table(self, df, schema_name, table_name):
        """
        Using cursor.mogrify() to build the bulk insert query
        then cursor.execute() to execute the query
        """
        with ConnHandler(self.get_conn('feature', schema_name)) as conn:
            table = f'{schema_name}.{table_name}'
            # Create a list of tupples from the dataframe values
            tuples = [tuple(x) for x in df.to_numpy()]
            # Comma-separated dataframe columns
            cols = ','.join(list(df.columns))
            # SQL quert to execute
            cursor = conn.cursor()
            mogrify_str = ",".join(["%s"] * len(df.columns))
            values = [
                cursor.mogrify(
                    f"({mogrify_str})", tup
                ).decode('utf8') for tup in tuples]
            query = "INSERT INTO %s(%s) VALUES " % (
                table, cols) + ",".join(values) + """
            """
            logging.info('[insert_table] Before SQL Execute')
            try:
                cursor.execute(query, tuples)
                conn.commit()
                logging.info('[insert_table] SQL Execute SUCCESS')
            except (Exception, psycopg2.DatabaseError) as error:
                conn.rollback()
                error_message = '[insert_table] SQL Execute FAIL \n' +\
                                f' df of size {len(df)}: \n' +\
                                f'{df.head().to_string()} \n'
                logging.exception(error_message)
                raise error
            finally:
                cursor.close()
                logging.info('[insert_table] CURSOR CLOSE')

    def insert_table_fast(self, df, table_name, encoding='utf-8', index=False):
        """
        Args:
            - df: pandas dataframe to be insert to db.
            - table_name: the name of the table to be inserted
            - encoding: utf-8 by default
            - index: False by default
        """
        in_memory_csv = StringIO()
        df.to_csv(
            in_memory_csv,
            sep=',',
            header=False,
            encoding=encoding,
            index=index)
        in_memory_csv.seek(0)
        if VERBOSE:
            logging.info('[insert_table_fast] Before CONN BUILT')
        with ConnHandler(self.get_conn('feature', self.schema_name)) as conn:
            if VERBOSE:
                logging.info('[insert_table_fast] Before SQL Execute')
            try:
                cursor = conn.cursor()
                schema_tablename = '"{}"."{}"'.format(
                    self.schema_name, table_name)
                cursor.copy_expert(
                    "COPY " +
                    schema_tablename +
                    " FROM STDIN WITH (FORMAT CSV)"
                    "",
                    in_memory_csv)
                conn.commit()
                if VERBOSE:
                    logging.info('[insert_table_fast] SQL Execute SUCCESS')
            except Exception as error:
                conn.rollback()
                error_message = '[insert_table_fast] SQL Execute FAIL \n' +\
                                f' df of size {len(df)}: \n' +\
                                f'{df.head().to_string()} \n'
                logging.exception(error_message)
                raise error
            finally:
                cursor.close()
            if VERBOSE:
                logging.info('[insert_table_fast] CONN/CURSOR CLOSE')

    def string_io_generator(
            self, result_gen, num_batch_per_insert, encoding='utf-8', index=False):
        """
        Args:
            - result_gen: generator that generate results processed by run_partially
            - num_batch_per_insert: number of batches aggregated into one StringIO object
                (to avoid too many insert transaction)
        Yeilds:
            - in_memory_csv: StringIO object
        """
        if VERBOSE:
            logging.info('[string_io_generator] Initialize StringIO')
        in_memory_csv = StringIO()
        for i, df in enumerate(result_gen):
            if VERBOSE:
                logging.info(
                    '[string_io_generator] Aggregate CSV files in StringIO')

            self.__df_to_csv(df, in_memory_csv)

            if VERBOSE:
                logging.info('[string_io_generator] Clean up the DF')
            del df
            gc.collect()
            if i % num_batch_per_insert == (num_batch_per_insert - 1):
                if VERBOSE:
                    logging.info('[string_io_generator] Yield StringIO')
                yield in_memory_csv
                if VERBOSE:
                    logging.info(
                        '[string_io_generator] Re-initialize StringIO')
                in_memory_csv = StringIO()
        yield in_memory_csv

    def __df_to_csv(self, df, in_memory_csv, encoding='utf-8', index=False):
        df.to_csv(
            in_memory_csv,
            sep=',',
            header=False,
            encoding=encoding,
            index=index)

    def insert_table_from_string_io(self, in_memory_csv, table_name):
        """
        Args:
            - in_memory_csv: StringIO object which aggregate the csv rows in memory.
            - table_name: the name of the table to be inserted
            - encoding: utf-8 by default
            - index: False by default
        """
        in_memory_csv.seek(0)
        if VERBOSE:
            logging.info('[insert_table_from_string_io] Before CONN BUILT')
        with ConnHandler(self.get_conn('feature', self.schema_name)) as conn:
            if VERBOSE:
                logging.info(
                    '[insert_table_from_string_io] Before SQL Execute')
            try:
                cursor = conn.cursor()
                schema_tablename = '"{}"."{}"'.format(
                    self.schema_name, table_name)
                cursor.copy_expert(
                    "COPY " +
                    schema_tablename +
                    " FROM STDIN WITH (FORMAT CSV)"
                    "",
                    in_memory_csv)
                conn.commit()
                if VERBOSE:
                    logging.info(
                        '[insert_table_from_string_io] SQL Execute SUCCESS')
            except Exception as error:
                conn.rollback()
                error_message = '[insert_table_from_string_io] SQL Execute FAIL'
                logging.exception(error_message)
                raise error
            finally:
                cursor.close()
            if VERBOSE:
                logging.info('[insert_table_from_string_io] CONN/CURSOR CLOSE')

    def get_conn(self, db_name, schema_name, connection=config.CONNECTION):
        """
        get DB connection
        (connection == mlaas_tool or airflow)
        """
        try:
            if connection == 'airflow':
                from airflow.hooks.postgres_hook import PostgresHook
                if db_name == 'rawdata':
                    gw_file_hook = PostgresHook(
                        f"{schema_name}_rawdata_db")
                    postgres_conn = gw_file_hook.get_conn()
                    postgres_conn.autocommit = False
                elif db_name == 'feature':
                    gw_file_hook = PostgresHook(
                        f"{schema_name}_feature_db")
                    postgres_conn = gw_file_hook.get_conn()
                    postgres_conn.autocommit = False
            elif connection == 'mlaas_tools':
                from mlaas_tools2.config_info import ConfigPass
                from mlaas_tools2.db_tool import DatabaseConnections
                config_pass = ConfigPass()
                conns = DatabaseConnections(config_pass, is_pgpool=False)
                if db_name == 'rawdata':
                    postgres_conn = conns.get_rawdata_db_conn()
                    postgres_conn.autocommit = False
                elif db_name == 'feature':
                    postgres_conn = conns.get_feature_db_conn()
                    postgres_conn.autocommit = False
            return postgres_conn

        except (Exception, psycopg2.Error) as error:
            raise error
