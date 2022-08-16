"""Modin ETL operation zoo."""
from io import IOBase
from typing import Any, Optional

import modin.pandas as mpd

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


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
        in_memory_csv: IOBase,
        df: Optional[Any] = None,
    ) -> mpd.DataFrame:
        """Read and return table in in-memory buffer."""
        in_memory_csv.seek(0)
        df = mpd.read_csv(in_memory_csv)

        return df
