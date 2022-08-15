"""Vaex ETL operation zoo."""
from typing import Any

from profile_framework.etl_op_zoo.base import BaseETLOpZoo
from profile_framework.utils.profile import Profiler


class ETLOpZoo(BaseETLOpZoo):
    """Vaex ETL operation zoo."""

    @staticmethod
    @Profiler.profile_factory(return_prf=True)
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.join(df_rhs, how=how, on=on, rsuffix="_rhs")

        return etl_result
