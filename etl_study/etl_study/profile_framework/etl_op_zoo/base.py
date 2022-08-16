"""Base class of ETL operation zoo."""
from io import IOBase
from typing import Any, Dict, List, Optional

import pandas as pd


class BaseETLOpZoo:
    """Base class of ETL operation zoo.

    See also www.learncodewithmike.com/2020/01/python-method.html.
    """

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Read and return input file with .parquet extension."""
        df = pd.read_parquet(input_file)

        return df

    @staticmethod
    def read_psql(
        in_memory_csv: IOBase,
        df: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Read and return table in in-memory buffer."""
        in_memory_csv.seek(0)
        df = pd.read_csv(in_memory_csv)

        return df

    @staticmethod
    def groupby(
        df: Any,
        groupby_keys: List[str],
        stats_dict: Dict[str, List[str]],
    ) -> Any:
        """Group samples and derive stats for each group."""
        etl_result = df.groupby(groupby_keys).agg(stats_dict)

        return etl_result

    @staticmethod
    def rolling(
        df: Any,
    ) -> Any:
        """Derive rolling stats."""
        etl_result = None
        return etl_result

    @staticmethod
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.merge(df_rhs, how=how, on=on)

        return etl_result
