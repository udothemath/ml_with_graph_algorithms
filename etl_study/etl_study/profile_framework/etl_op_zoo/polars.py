"""Polars ETL operation zoo."""
from typing import Any, Optional

import polars as pd

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


class ETLOpZoo(BaseETLOpZoo):
    """Polars ETL operation zoo."""

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Read and return input file with .parquet extension."""
        df = pd.read_parquet(input_file)

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
