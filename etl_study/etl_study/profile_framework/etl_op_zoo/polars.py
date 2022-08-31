"""Polars ETL operation zoo."""
from io import IOBase
from typing import Any, Optional

import polars as pl
from pyarrow import csv as pa_csv

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


class ETLOpZoo(BaseETLOpZoo):
    """Polars ETL operation zoo."""

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> pl.DataFrame:
        """Read and return input file with .parquet extension."""
        df = pl.read_parquet(input_file)

        return df

    @staticmethod
    def read_psql(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> pl.DataFrame:
        """Read and return table in in-memory buffer."""
        in_memory_buff.seek(0)
        df = pl.read_csv(in_memory_buff)

        return df

    @staticmethod
    def read_psql_advanced(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> pl.DataFrame:
        """Read and return table converted from in-memory arrow table."""
        in_memory_buff.seek(0)
        df_arrow = pa_csv.read_csv(in_memory_buff)
        df = pl.from_arrow(df_arrow)

        return df

    @staticmethod
    def to_parquet(df: pl.DataFrame) -> None:
        """Directly write input file to the output file with .parquet
        extension.
        """
        df.write_parquet("tmp.parquet")

        return None

    @staticmethod
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> pl.DataFrame:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.join(df_rhs, how=how, on=on)

        return etl_result
