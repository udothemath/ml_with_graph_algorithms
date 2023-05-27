"""
Selecting DB Table from API Operator
"""
import pandas as pd
import psycopg2
from contextlib import closing
SCHEMA_NAME = 'pl_automated_valuation'


class DB_Caller:
    def __init__(self, dbset, logger):
        """
        __init__ Args:
            - dbset: the hook to dbset provided by API Operator
            - logger: the logger provided by API Operator
        """
        self.__dbset = dbset
        self.logger = logger

    @property
    def schema_name(self):
        return SCHEMA_NAME

    def select_table(self, db, sql):
        with closing(self.__get_conn(db, self.schema_name)) as conn:
            df = pd.read_sql(sql, conn)
            return df

    def __get_conn(self, db_name, schema_name, connection='mlaas_tool'):
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
            elif connection == 'mlaas_tool':
                conns = self.__dbset
                if db_name == 'rawdata':
                    postgres_conn = conns.get_rawdata_db_conn()
                    postgres_conn.autocommit = False
                elif db_name == 'feature':
                    postgres_conn = conns.get_feature_db_conn()
                    postgres_conn.autocommit = False
            return postgres_conn

        except (Exception, psycopg2.Error) as error:
            self.logger.exception(error)
            raise
