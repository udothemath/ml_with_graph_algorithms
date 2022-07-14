"""ETL basic utilities."""
import gc
import io
import logging
import os
from contextlib import closing
from io import IOBase, StringIO
from typing import Any, Dict, Iterable, List

import pandas as pd
import psycopg2
from psycopg2.extensions import connection
from src import config

# from mlaas_tools.config_build import config_set
# config_set()
# remove the above two lines once config.ini is generated
VERBOSE = False
CLOSE_ON_DEL = False
NUM_BATCH_PER_CONN = 10
PROCESS_CONN: Dict[int, connection] = {}
CONN_CNT: Dict[int, int] = {}


class ConnHandler(closing):
    """DB connection handler.

    `thing` will be any DB connection object needed to be closed
    whether the corresponding DB operation succeeds or not to ensure
    that the resources can be released.

    For information about `closing`, please see:
        https://github.com/python/cpython/blob/main/Lib/contextlib.py

    Parameters:
        thing: database connection object
    """

    thing: connection

    def __init__(self, thing: connection):
        super(ConnHandler, self).__init__(thing)
        pid = os.getpid()
        logging.info(f"[ConnHandler ({pid})] OPEN")

    def __exit__(self, *exc_info: Any) -> None:
        self.thing.close()
        pid = os.getpid()
        logging.info(f"[ConnHandler ({pid})] CLOSE")


class ETLBase:
    """Base class for advanced ETL operations."""

    def __init__(self) -> None:
        pass

    @property
    def db_users(self) -> List[str]:
        return config.DB_USERS

    @property
    def schema_name(self) -> str:
        return config.SCHEMA_NAME

    def execute_sql(
        self,
        db: str,
        sql: str,
    ) -> None:
        """Execute pre-defined SQL logic.

        Parameters:
            db: the database name, the choices are as follows:
                {"rawdata", "feature"}
            sql: SQL logic

        Return:
            None
        """
        with ConnHandler(self.get_conn(db, self.schema_name)) as conn:
            try:
                cur = conn.cursor()
                cur.execute(sql)
                logging.info(f"[execute_sql] EXECUTE SQL:\n{sql}")
                conn.commit()
                cur.close()
            except (Exception, psycopg2.DatabaseError) as error:
                conn.rollback()
                cur.close()
                logging.info("[execute_sql] EXECUTE SQL FAIL")
                raise error
        logging.info("[execute_sql] EXECUTE SQL SUCCESS")

    def select_n_insert_tool(
        self,
        etlSQL: str,
        source_db: str,
        target_schema: str,
        target_table: str,
    ) -> None:
        """Read SQL query or source table into target table via
        in-memory buffer.

        Parameters:
            etlSQL: SQL logic
            source_db: the source database name
            target_schema: the target schema name
            target_table: the target table name

        Return:
            None
        """
        with ConnHandler(self.get_conn(source_db, self.schema_name)) as source_conn, ConnHandler(
            self.get_conn("feature", self.schema_name)
        ) as target_conn:
            in_memory_file = io.BytesIO()
            source_copy_sql = """COPY (%s) TO STDOUT DELIMITER ',' CSV""" % etlSQL
            table_name = target_schema + "." + target_table
            copy_query = """COPY %s FROM STDIN WITH (FORMAT CSV, DELIMITER ',', HEADER FALSE)""" % table_name
            try:
                with source_conn.cursor() as raw_cur:
                    logging.info("[select_n_insert_tool] Start getting data from source db")
                    raw_cur.copy_expert(source_copy_sql, in_memory_file)
                    in_memory_file.seek(0)
                    logging.info("[select_n_insert_tool] Done getting data")
                    try:
                        with target_conn.cursor() as ftr_cur:
                            logging.info("[select_n_insert_tool] Start insert data into target (feature) db")
                            ftr_cur.copy_expert(copy_query, in_memory_file)
                        target_conn.commit()
                        logging.info("[select_n_insert_tool] Done inserting data")
                    except Exception:
                        target_conn.rollback()
                        raise
            except Exception:
                source_conn.rollback()
                raise Exception

    def select_n_insert_tool_binary(
        self,
        etlSQL: str,
        source_db: str,
        target_schema: str,
        target_table: str,
    ) -> None:
        """Read SQL query or source table into target table via
        in-memory buffer using binary data format.

        Parameters:
            etlSQL: SQL logic
            source_db: the source database name
            target_schema: the target schema name
            target_table: the target table name

        Return:
            None
        """
        with ConnHandler(self.get_conn(source_db, self.schema_name)) as source_conn, ConnHandler(
            self.get_conn("feature", self.schema_name)
        ) as target_conn:
            in_memory_file = io.BytesIO()
            source_copy_sql = """COPY (%s) TO STDOUT WITH BINARY""" % etlSQL
            table_name = target_schema + "." + target_table
            copy_query = """COPY %s FROM STDIN (FORMAT BINARY)""" % table_name

            try:
                with source_conn.cursor() as raw_cur:
                    logging.info("[select_n_insert_tool_binary] Start getting data from source db")
                    raw_cur.copy_expert(source_copy_sql, in_memory_file)
                    in_memory_file.seek(0)
                    logging.info("[select_n_insert_tool_binary] Done getting data")
                    try:
                        with target_conn.cursor() as ftr_cur:
                            logging.info("[select_n_insert_tool_binary] Start insert data into target (feature) db")
                            ftr_cur.copy_expert(copy_query, in_memory_file)
                        target_conn.commit()
                        logging.info("[select_n_insert_tool_binary] Done inserting data")
                    except Exception:
                        target_conn.rollback()
                        raise
            except Exception:
                source_conn.rollback()
                logging.info("[select_n_insert_tool_binary] RUN FAIL")
                raise Exception

    def select_table(self, db: str, sql: str) -> pd.DataFrame:
        """Read SQL query or table into a DataFrame.

        Parameters:
            db: the database name, the choices are as follows:
                {"rawdata", "feature"}
            sql: SQL logic

        Return:
            df: retrieved DataFrame
        """
        with ConnHandler(self.get_conn(db, self.schema_name)) as conn:
            df = pd.read_sql(sql, conn)
        return df

    def open_conn(self, db: str) -> connection:
        """Return DB connection object.

        Note that a new DB connection is created only if the current
        process doesn't have one.

        Parameters:
            db: the database name, the choices are as follows:
                {"rawdata", "feature"}

        Return:
            conn: database connection object
        """
        global PROCESS_CONN
        global CONN_CNT
        pid = os.getpid()
        if pid in PROCESS_CONN:
            conn = PROCESS_CONN[pid]
            if VERBOSE:
                logging.info(f"[open_conn ({pid})] GET")
        else:
            conn = self.get_conn(db, self.schema_name)
            logging.info(f"[open_conn ({pid})] OPEN")
            PROCESS_CONN[pid] = conn
        if pid in CONN_CNT:
            CONN_CNT[pid] += 1
        else:
            CONN_CNT[pid] = 1
        return conn

    def close_conn(self) -> None:
        """Close DB connection in the current process.

        Return:
            None
        """
        global PROCESS_CONN
        pid = os.getpid()
        if pid in PROCESS_CONN:
            PROCESS_CONN[pid].close()
            del PROCESS_CONN[pid]
            logging.info(f"[close_conn ({pid})] DONE")
        else:
            logging.info(f"[close_conn ({pid})] NOTHING TO CLOSE: {PROCESS_CONN}")

    def __del__(self) -> None:
        global CONN_CNT
        global CLOSE_ON_DEL
        if CLOSE_ON_DEL:
            pid = os.getpid()
            if (pid in CONN_CNT) and (CONN_CNT[pid] % NUM_BATCH_PER_CONN == (NUM_BATCH_PER_CONN - 1)):
                self.close_conn()

    def select_table_fast(
        self,
        db: str,
        sql: str,
        encoding: str = "utf-8",
        index: bool = False,
    ) -> pd.DataFrame:
        """Read SQL query or table into a DataFrame via in-memory
        buffer.

        Parameters:
            db: the database name, the choices are as follows:
                {"rawdata", "feature"}
            sql: SQL logic
            encoding: encoding
            index: always ignored

        Return:
            df: retrieved DataFrame
        """
        logging.info(f"[select_table_fast] Start (index: {index})")
        in_memory_csv = StringIO()
        with ConnHandler(self.get_conn(db, self.schema_name)) as conn:
            cursor = conn.cursor()
            cursor.copy_expert(f"COPY ({sql}) TO STDOUT DELIMITER ',' CSV HEADER", in_memory_csv)
            in_memory_csv.seek(0)
            conn.commit()
            cursor.close()
            result = pd.read_csv(in_memory_csv, sep=",", header="infer", encoding=encoding)
        return result

    def insert_table(
        self,
        df: pd.DataFrame,
        schema_name: str,
        table_name: str,
    ) -> None:
        """Insert DataFrame into target table.

        Parameters:
            df: input DataFrame
            schema_name: the target schema name
            table_name: the target table name

        Return:
            None
        """
        with ConnHandler(self.get_conn("feature", schema_name)) as conn:
            table = f"{schema_name}.{table_name}"
            # Create a list of tupples from the dataframe values
            tuples = [tuple(x) for x in df.to_numpy()]
            # Comma-separated dataframe columns
            cols = ",".join(list(df.columns))
            # SQL quert to execute
            cursor = conn.cursor()
            mogrify_str = ",".join(["%s"] * len(df.columns))
            values = [cursor.mogrify(f"({mogrify_str})", tup).decode("utf8") for tup in tuples]
            query = (
                "INSERT INTO %s(%s) VALUES " % (table, cols)
                + ",".join(values)
                + """
            """
            )
            logging.info("[insert_table] Before SQL Execute")
            try:
                cursor.execute(query, tuples)
                conn.commit()
                logging.info("[insert_table] SQL Execute SUCCESS")
            except (Exception, psycopg2.DatabaseError) as error:
                conn.rollback()
                error_message = (
                    "[insert_table] SQL Execute FAIL \n" + f" df of size {len(df)}: \n" + f"{df.head().to_string()} \n"
                )
                logging.exception(error_message)
                raise error
            finally:
                cursor.close()
                logging.info("[insert_table] CURSOR CLOSE")

    def insert_table_fast(
        self,
        df: pd.DataFrame,
        table_name: str,
        encoding: str = "utf-8",
        index: bool = False,
    ) -> None:
        """Insert DataFrame into target table via in-memory buffer.

        Parameters:
            df: input DataFrame
            table_name: the target table name
            encoding: encoding
            index: write row names

        Return:
            None
        """
        in_memory_csv = StringIO()
        df.to_csv(in_memory_csv, sep=",", header=False, encoding=encoding, index=index)
        in_memory_csv.seek(0)
        if VERBOSE:
            logging.info("[insert_table_fast] Before CONN BUILT")
        with ConnHandler(self.get_conn("feature", self.schema_name)) as conn:
            if VERBOSE:
                logging.info("[insert_table_fast] Before SQL Execute")
            try:
                cursor = conn.cursor()
                schema_tablename = '"{}"."{}"'.format(self.schema_name, table_name)
                cursor.copy_expert(
                    "COPY " + schema_tablename + " FROM STDIN WITH (FORMAT CSV)" "",
                    in_memory_csv,
                )
                conn.commit()
                if VERBOSE:
                    logging.info("[insert_table_fast] SQL Execute SUCCESS")
            except Exception as error:
                conn.rollback()
                error_message = (
                    "[insert_table_fast] SQL Execute FAIL \n"
                    + f" df of size {len(df)}: \n"
                    + f"{df.head().to_string()} \n"
                )
                logging.exception(error_message)
                raise error
            finally:
                cursor.close()
            if VERBOSE:
                logging.info("[insert_table_fast] CONN/CURSOR CLOSE")

    def string_io_generator(
        self,
        result_gen: Iterable[pd.DataFrame],
        num_batch_per_insert: int,
        encoding: str = "utf-8",
        index: bool = False,
    ) -> Iterable[IOBase]:
        """Return full in-memory buffer.

        Parameters:
            result_gen: generator that generates results processed by
                `run_partially`
            num_batch_per_insert: number of batches aggregated into one
                StringIO object
                *Note: Avoid processing too many insert transactions
            encoding: encoding
            index: always ignored

        Yeilds:
            in_memory_csv: in-memory buffer
        """
        if VERBOSE:
            logging.info(f"[string_io_generator] Initialize StringIO (index: {index})")
        in_memory_csv = StringIO()
        for i, df in enumerate(result_gen):
            if VERBOSE:
                logging.info("[string_io_generator] Aggregate CSV files in StringIO")

            self.__df_to_csv(df, in_memory_csv)

            if VERBOSE:
                logging.info("[string_io_generator] Clean up the DF")
            del df
            gc.collect()
            if i % num_batch_per_insert == (num_batch_per_insert - 1):
                if VERBOSE:
                    logging.info("[string_io_generator] Yield StringIO")
                yield in_memory_csv
                if VERBOSE:
                    logging.info("[string_io_generator] Re-initialize StringIO")
                in_memory_csv = StringIO()
        yield in_memory_csv

    def __df_to_csv(
        self,
        df: pd.DataFrame,
        in_memory_csv: IOBase,
        encoding: str = "utf-8",
        index: bool = False,
    ) -> None:
        """Write DataFrame to in-memory buffer.

        Parameters:
            df: input DataFrame
            in_memory_csv: in-memory buffer
            encoding: encoding
            index: write row names

        Return:
            None
        """
        df.to_csv(in_memory_csv, sep=",", header=False, encoding=encoding, index=index)

    def insert_table_from_string_io(
        self,
        in_memory_csv: IOBase,
        table_name: str,
    ) -> None:
        """Insert DataFrame into target table via in-memory buffer.

        Parameters:
            in_memory_csv: in-memory buffer
            table_name: the target table name

        Return:
            None
        """
        in_memory_csv.seek(0)
        if VERBOSE:
            logging.info("[insert_table_from_string_io] Before CONN BUILT")
        with ConnHandler(self.get_conn("feature", self.schema_name)) as conn:
            if VERBOSE:
                logging.info("[insert_table_from_string_io] Before SQL Execute")
            try:
                cursor = conn.cursor()
                schema_tablename = '"{}"."{}"'.format(self.schema_name, table_name)
                cursor.copy_expert(
                    "COPY " + schema_tablename + " FROM STDIN WITH (FORMAT CSV)" "",
                    in_memory_csv,
                )
                conn.commit()
                if VERBOSE:
                    logging.info("[insert_table_from_string_io] SQL Execute SUCCESS")
            except Exception as error:
                conn.rollback()
                error_message = "[insert_table_from_string_io] SQL Execute FAIL"
                logging.exception(error_message)
                raise error
            finally:
                cursor.close()
            if VERBOSE:
                logging.info("[insert_table_from_string_io] CONN/CURSOR CLOSE")

    def get_conn(
        self,
        db_name: str,
        schema_name: str,
        connection: str = config.CONNECTION,
    ) -> connection:
        """Create and return DB connection object.

        Parameters:
            db_name: the database name, the choices are as follows:
                {"rawdata", "feature"}
            schema_name: the schema name
            connection: connection channel, the choices are as follows:
                {"airflow", "mlaas_tool"}

        Return:
            postgres_conn: database connection object
        """
        try:
            if connection == "airflow":
                from airflow.hooks.postgres_hook import PostgresHook

                if db_name == "rawdata":
                    gw_file_hook = PostgresHook(f"{schema_name}_rawdata_db")
                    postgres_conn = gw_file_hook.get_conn()
                    postgres_conn.autocommit = False
                elif db_name == "feature":
                    gw_file_hook = PostgresHook(f"{schema_name}_feature_db")
                    postgres_conn = gw_file_hook.get_conn()
                    postgres_conn.autocommit = False
            elif connection == "mlaas_tools":
                from mlaas_tools2.config_info import ConfigPass
                from mlaas_tools2.db_tool import DatabaseConnections

                config_pass = ConfigPass()
                conns = DatabaseConnections(config_pass, is_pgpool=False)
                if db_name == "rawdata":
                    postgres_conn = conns.get_rawdata_db_conn()
                    postgres_conn.autocommit = False
                elif db_name == "feature":
                    postgres_conn = conns.get_feature_db_conn()
                    postgres_conn.autocommit = False
            return postgres_conn

        except (Exception, psycopg2.Error) as error:
            raise error
