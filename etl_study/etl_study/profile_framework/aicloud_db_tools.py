import psycopg2
import traceback
import io
import logging
from configparser import ConfigParser
import psycopg2 as _psycopg2
import os
PROJ_NAME = 'wm_aiget'
ENV = 'aicloud'
# from mlaas_tools.config_build import config_set
# config_set()
# remove the above two lines once config.ini is generated


def get_conn(schema_name):
    """
    get DB connection
    (connection == mlaas_tool or airflow)
    """
    if schema_name in ['mlaas_rawdata', 'mlaas_limit']:
        db_name = 'rawdata'
    else:
        db_name = 'feature'
    try:
        if ENV == 'aicloud' and os.path.exists('config.ini'):
            config = ConfigParser()
            config.read('config.ini')
            if db_name == 'rawdata':
                raw_id = f'{PROJ_NAME}_rawdata_db'
                postgres_conn = _psycopg2.connect(
                    database=config.get(raw_id, 'schema'),
                    host=config.get(raw_id, 'host'),
                    port=config.get(raw_id, 'port'),
                    user=config.get(raw_id, 'login'),
                    password=config.get(raw_id, 'passphrase')
                )
                postgres_conn.autocommit = False
            elif db_name == 'feature':
                feature_id = f'{PROJ_NAME}_feature_db'
                postgres_conn = _psycopg2.connect(
                    database=config.get(feature_id, 'schema'),
                    host=config.get(feature_id, 'host'),
                    port=config.get(feature_id, 'port'),
                    user=config.get(feature_id, 'login'),
                    password=config.get(feature_id, 'passphrase')
                )
                postgres_conn.autocommit = False
        else:
            from airflow.hooks.postgres_hook import PostgresHook
            if db_name == 'rawdata':
                gw_file_hook = PostgresHook(
                    f"{PROJ_NAME}_rawdata_db")
                postgres_conn = gw_file_hook.get_conn()
                postgres_conn.autocommit = False
            elif db_name == 'feature':
                gw_file_hook = PostgresHook(
                    f"{PROJ_NAME}_feature_db")
                postgres_conn = gw_file_hook.get_conn()
                postgres_conn.autocommit = False
        return postgres_conn

    except (Exception, psycopg2.Error) as error:        
        raise error