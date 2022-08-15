"""Base class of ETL operation zoo."""
from typing import Any, Dict, List


class BaseETLOpZoo:
    """Base class of ETL operation zoo.

    See also www.learncodewithmike.com/2020/01/python-method.html.
    """

    @staticmethod
    def groupby(
        df: Any,
        groupby_keys: List[str],
        stats_dict: Dict[str, List[str]],
    ) -> Any:
        """Group samples and derive stats for each group."""
        etl_result = df.groupby(groupby_keys).agg(stats_dict)

        return etl_result

    @staticmethod
    def rolling(
        df: Any,
    ) -> Any:
        """Derive rolling stats."""
        etl_result = None
        return etl_result

    @staticmethod
    def join(
        df: Any,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        etl_result = df.merge(df_rhs, how=how, on=on)

        return etl_result
