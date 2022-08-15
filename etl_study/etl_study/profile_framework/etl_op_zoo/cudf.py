"""CUDF ETL operation zoo."""
from typing import Any, Optional

import cudf as pd

from profile_framework.etl_op_zoo.base import BaseETLOpZoo


class ETLOpZoo(BaseETLOpZoo):
    """CUDF ETL operation zoo."""

    @staticmethod
    def read_parquet(
        input_file: str,
        df: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Read and return input file with .parquet extension."""
        df = pd.read_parquet(input_file)

        return df
