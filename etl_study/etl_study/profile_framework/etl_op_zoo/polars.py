"""Polars ETL operation zoo."""
from io import IOBase
from typing import Any, Optional

import polars as pl

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
        in_memory_csv: IOBase,
        df: Optional[Any] = None,
    ) -> pl.DataFrame:
        """Read and return table in in-memory buffer."""
        in_memory_csv.seek(0)
        df = pl.read_csv(in_memory_csv)

        return df

    @staticmethod
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.join(df_rhs, how=how, on=on)

        return etl_result
