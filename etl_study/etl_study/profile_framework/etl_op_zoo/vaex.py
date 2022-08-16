"""Vaex ETL operation zoo."""
from typing import Any, Optional

import vaex

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
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.join(df_rhs, how=how, on=on, rsuffix="_rhs")

        return etl_result
