"""Modin ETL operation zoo."""
from io import IOBase
from typing import Any, Optional

import modin.pandas as mpd
from pyarrow import csv as pa_csv

from profile_framework.etl_op_zoo.base import BaseETLOpZoo

import ray
ray.init(object_store_memory=100*10**9)

class ETLOpZoo(BaseETLOpZoo):
    """Modin ETL operation zoo."""

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> mpd.DataFrame:
        """Read and return input file with .parquet extension."""
        df = mpd.read_parquet(input_file)

        return df

    @staticmethod
    def read_psql(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> mpd.DataFrame:
        """Read and return table in in-memory buffer."""
        in_memory_buff.seek(0)
        df = mpd.read_csv(in_memory_buff)

        return df

    @staticmethod
    def read_psql_advanced(
        in_memory_buff: IOBase,
        df: Optional[Any] = None,
    ) -> mpd.DataFrame:
        """Read and return table converted from in-memory arrow table."""
        in_memory_buff.seek(0)
        df_arrow = pa_csv.read_csv(in_memory_buff)
        df = mpd.utils.from_arrow(df_arrow)

        return df
