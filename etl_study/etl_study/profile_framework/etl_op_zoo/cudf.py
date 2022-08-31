"""CUDF ETL operation zoo."""
from io import IOBase
from typing import Any, Optional

import cudf
from pyarrow import csv as pa_csv

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


class ETLOpZoo(BaseETLOpZoo):
    """CUDF ETL operation zoo."""

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> cudf.DataFrame:
        """Read and return input file with .parquet extension."""
        df = cudf.read_parquet(input_file)

        return df

    @staticmethod
    def read_psql(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> cudf.DataFrame:
        """Read and return table in in-memory buffer."""
        in_memory_buff.seek(0)
        df = cudf.read_csv(in_memory_buff)

        return df

    @staticmethod
    def read_psql_advanced(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> cudf.DataFrame:
        """Read and return table converted from in-memory arrow table."""
        in_memory_buff.seek(0)
        df_arrow = pa_csv.read_csv(in_memory_buff)
        df = cudf.DataFrame.from_arrow(df_arrow)

        return df
