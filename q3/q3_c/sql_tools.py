import os
import sys
import pandas as pd
import psycopg2
from io import StringIO

# from mlaas_tools.config_build import config_set
# config_set()
# remove the above two lines once config.ini is generated
VERBOSE = False

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


class ETLBase:
    def __init__(self):
        pass

    @property
    def db_users(self):
        return []

    @property
    def schema_name(self):
        return 'eb_ofsn_wm_fund'

    def execute_sql(self, db, sql):
        with self.get_conn(db, self.schema_name) as conn:
            try:
                cur = conn.cursor()
                cur.execute(sql)
                cur.close()
                conn.commit()
            except (Exception, psycopg2.DatabaseError) as error:
                conn.rollback()
                cur.close()
                raise error
            
    def select_table(self, db, sql):
        with self.get_conn(db, self.schema_name) as conn:
            df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def select_table_fast(self, db, sql, encoding='utf-8', index=False):
        in_memory_csv = StringIO()
        with self.get_conn(db, self.schema_name) as conn:
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
        conn = self.get_conn('feature', schema_name)
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
        try:
            cursor.execute(query, tuples)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            conn.rollback()
            cursor.close()
            raise ValueError(
                f'df of size {len(df)}: {df.head()} causing error: {error}')
        finally:
            cursor.close()
            conn.close()

    def insert_table_fast(self, df, table_name, encoding='utf-8', index=False):
        if VERBOSE: print(f'INSERT ROW: {len(df)}');
        in_memory_csv = StringIO()
        df.to_csv(
            in_memory_csv,
            sep=',',
            header=False,
            encoding=encoding,
            index=index)
        in_memory_csv.seek(0)
        with self.get_conn('feature', self.schema_name) as conn:
            try:
                cursor = conn.cursor()
                schema_tablename = '"{}"."{}"'.format(self.schema_name, table_name)
                cursor.copy_expert(
                    "COPY " +
                    schema_tablename +
                    " FROM STDIN WITH (FORMAT CSV)"
                    "",
                    in_memory_csv)
                conn.commit()
                
            except Exception as error:
                conn.rollback()
                cursor.close()
                raise ValueError(
                    f'df of size {len(df)}: {df.head()} causing error: {error}')
            finally:
                cursor.close()

    # @blockPrinting
    def get_conn(self, db_name, schema_name, connection='mlaas_tools'):
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
                from mlaas_tools.config_info import ConfigPass
                from mlaas_tools.db_tool import DatabaseConnections
                configs = ConfigPass()._configsection
                conns = DatabaseConnections(configs)
                if db_name == 'rawdata':
                    postgres_conn = conns.get_rawdata_db_conn()
                    postgres_conn.autocommit = False
                elif db_name == 'feature':
                    postgres_conn = conns.get_feature_db_conn()
                    postgres_conn.autocommit = False
            return postgres_conn

        except (Exception, psycopg2.Error) as error:
            raise ValueError(error)
