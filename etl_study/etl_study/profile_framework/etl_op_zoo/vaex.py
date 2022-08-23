"""Vaex ETL operation zoo."""
from io import IOBase
from typing import Any, Optional

import vaex
from pyarrow import csv as pa_csv

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


class ETLOpZoo(BaseETLOpZoo):
    """Vaex ETL operation zoo."""

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> vaex.dataframe.DataFrameLocal:
        """Read and return input file with .parquet extension."""
        df = vaex.open(input_file)

        return df

    @staticmethod
    def read_psql(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> vaex.dataframe.DataFrameLocal:
        """Read and return table in in-memory buffer."""
        in_memory_buff.seek(0)
        df = vaex.from_csv(in_memory_buff)

        return df

    @staticmethod
    def read_psql_advanced(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> vaex.dataframe.DataFrameLocal:
        """Read and return table converted from in-memory arrow table."""
        in_memory_buff.seek(0)
        df_arrow = pa_csv.read_csv(in_memory_buff)
        df = vaex.from_arrow_table(df_arrow)

        return df

    @staticmethod
    def to_parquet(df: vaex.dataframe.DataFrameLocal) -> None:
        """Directly write input file to the output file with .parquet
        extension.
        """
        df.export("tmp.parquet")

        return None

    @staticmethod
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> vaex.dataframe.DataFrameLocal:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.join(df_rhs, how=how, on=on, rsuffix="_rhs")

        return etl_result
