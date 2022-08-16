"""ETL query parser."""
from io import IOBase, StringIO
from typing import Any, Callable, Dict, List, Tuple, Union

import psycopg2

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


class QueryParser:
    """ETL query parser.

    **Parameters**:
    - `etl_op_zoo`: mode-specific ETL operation zoo
    """

    def __init__(self, etl_op_zoo: BaseETLOpZoo):
        self._etl_op_zoo = etl_op_zoo

    def parse(self, query: str) -> Tuple[Callable, Dict[str, Any]]:
        """Parse ETL operation query.

        **Parameters**:
        - `query`: ETL operation query

        **Return**:
        - `etl_func`: ETL operation function
        - `etl_func_kwargs`: arguments passed to ETL operation function
        """
        etl_op_name, *etl_op_body = query.split(" ")

        # Retrieve ETL operation function corresponding to op head
        etl_op_name = etl_op_name[1:-1]
        etl_func = eval(f"self._etl_op_zoo.{etl_op_name}")

        # Retrieve ETL operation arguments corresponding to op body
        etl_func_kwargs = self._parse_etl_op_body(etl_op_name, etl_op_body)

        return etl_func, etl_func_kwargs

    def _parse_etl_op_body(self, etl_op_name: str, etl_op_body: List[str]) -> Dict[str, Any]:
        """Parse ETL operation body.

        Parameters:
            etl_op_name: ETL operation name
            etl_op_body: ETL operation body

        Return:
            etl_func_kwargs: arguments passed to ETL operation function
        """
        parse_etl_op_body: Callable = None
        if etl_op_name == "apply":
            parse_etl_op_body = self._parse_apply
        elif etl_op_name == "groupby":
            parse_etl_op_body = self._parse_groupby
        elif etl_op_name == "rolling":
            parse_etl_op_body = self._parse_rolling
        elif etl_op_name == "join":
            parse_etl_op_body = self._parse_join
        elif etl_op_name == "read_parquet":
            parse_etl_op_body = self._parse_read_parquet
        elif etl_op_name == "read_psql":
            parse_etl_op_body = self._parse_read_psql

        etl_func_kwargs = parse_etl_op_body(etl_op_body)

        return etl_func_kwargs

    def _parse_apply(self) -> None:
        pass

    def _parse_groupby(
        self,
        op_body: List[str],
    ) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
        """Parse `groupby` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            groupby_keys: keys to determine groups
            stats_dict: stats to derive for each column
        """
        kwargs: Dict[str, Any] = {}

        groupby_keys: List[str] = []
        stats_dict: Dict[str, List[str]] = {}
        for i, ele in enumerate(op_body):
            if ele.startswith("int") or ele.startswith("float"):
                col_name = ele
                stats_dict[col_name] = []
            elif ele == "by":
                break
            else:
                stats_dict[col_name].append(ele.replace(",", ""))
        groupby_keys = op_body[i + 1 :]

        kwargs = {
            "groupby_keys": groupby_keys,
            "stats_dict": stats_dict,
        }

        return kwargs

    def _parse_rolling(
        self,
        op_body: List[str],
    ) -> Dict[str, Any]:
        """Parse `rolling` operation body.

        Parameters:
            op_body: ETL operation body

        Return:

        """
        kwargs: Dict[str, Any] = {}

        kwargs = {}

        return kwargs

    def _parse_join(
        self,
        op_body: List[str],
    ) -> Dict[str, str]:
        """Parse `join` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            how: type of `join` to perform
            on: column to join on
        """
        kwargs: Dict[str, str] = {
            "how": op_body[0],
            "on": op_body[2],
        }

        return kwargs

    def _parse_read_parquet(
        self,
        op_body: List[str],
    ) -> Dict[str, str]:
        """Parse `read_parquet` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            input_file: input file
        """
        kwargs: Dict[str, str] = {
            "input_file": op_body[0],
        }

        return kwargs

    def _parse_read_psql(
        self,
        op_body: List[str],
    ) -> Dict[str, IOBase]:
        """Parse `read_psql` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            in_memory_csv: in-memory buffer
        """
        kwargs: Dict[str, Any] = {}

        db_name, schema_name, table_name = op_body

        # Build DB connection
        conn = psycopg2.connect(dbname=db_name, user="abaowei")
        conn.autocommit = False

        # Construct SQL logic
        sql = f"COPY (SELECT * FROM {schema_name}.{table_name}) TO STDOUT " "WITH (FORMAT CSV, DELIMITER ',', HEADER)"

        in_memory_csv = StringIO()
        try:
            cur = conn.cursor()
            cur.copy_expert(sql, in_memory_csv)
            conn.commit()
            in_memory_csv.seek(0)
        except (Exception, psycopg2.DatabaseError) as error:
            conn.rollback()
            raise error
        finally:
            cur.close()
        conn.close()

        kwargs = {
            "in_memory_csv": in_memory_csv,
        }

        return kwargs
