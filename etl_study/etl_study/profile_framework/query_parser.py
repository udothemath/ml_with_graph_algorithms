"""ETL query parser."""
from typing import Any, Callable, Dict, List, Tuple, Union

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
        - `kwargs`: arguments passed to ETL operation function
        """
        etl_op_name, *etl_op_body = query.split(" ")
        etl_op_name = etl_op_name[1:-1]
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

        etl_func, kwargs = parse_etl_op_body(etl_op_body)

        return etl_func, kwargs

    def _parse_apply(self) -> None:
        pass

    def _parse_groupby(
        self,
        op_body: List[str],
    ) -> Tuple[Callable, Dict[str, Union[List[str], Dict[str, List[str]]]]]:
        """Parse `groupby` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: groupby operation function
            groupby_keys: keys to determine groups
            stats_dict: stats to derive for each column
        """
        etl_func = self._etl_op_zoo.groupby
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

        return etl_func, kwargs

    def _parse_rolling(self) -> Tuple[Callable, Dict[str, Any]]:
        """Parse `rolling` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: rolling operation function
        """
        etl_func = self._etl_op_zoo.rolling
        kwargs: Dict[str, Any] = {}

        kwargs = {}

        return etl_func, kwargs

    def _parse_join(
        self,
        op_body: List[str],
    ) -> Tuple[Callable, Dict[str, str]]:
        """Parse `join` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: join operation function
            how: type of `join` to perform
            on: column to join on
        """
        etl_func = self._etl_op_zoo.join
        kwargs: Dict[str, Any] = {}

        kwargs = {
            "how": op_body[0],
            "on": op_body[2],
        }

        return etl_func, kwargs

    def _parse_read_parquet(
        self,
        op_body: List[str],
    ) -> Tuple[Callable, Dict[str, str]]:
        """Parse `read_parquet` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: read `.parquet` operation function
            input_file: input file
        """
        etl_func = self._etl_op_zoo.read_parquet
        kwargs: Dict[str, str] = {}

        kwargs = {
            "input_file": op_body[0],
        }

        return etl_func, kwargs
