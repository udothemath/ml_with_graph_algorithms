"""PostgreSQL ETL operation zoo."""
import inspect
from typing import Any, Dict, List, Optional

from common.etl_base import ETLBase
from profile_framework.sql_constructor import SQLConstructor

sql_ctor = SQLConstructor()
etl_interface = ETLBase()


class ETLOpZoo:
    """PostgreSQL ETL operation zoo.

    Because all the operations are done in PostgreSQL DB, where no file
    I/O participates; hence, PostgreSQL ETL operation zoo isn't
    inherited from `BaseETLOpZoo`.
    """

    @staticmethod
    def groupby(
        table_name: str,
        groupby_keys: List[str],
        stats_dict: Dict[str, List[str]],
        df: Optional[Any] = None,
    ) -> None:
        """Group samples and derive stats for each group."""
        groupby_sql = sql_ctor.construct(
            etl_op_name=inspect.currentframe().f_code.co_name,
            table_name=table_name,
            groupby_keys=groupby_keys,
            stats_dict=stats_dict,
        )  # Constructed overhead is ignored (~ 5us)
        etl_interface.execute_sql("rawdata", groupby_sql)

        return None

    @staticmethod
    def rolling(
        df: Any,
    ) -> Any:
        """Derive rolling stats."""
        etl_result = None
        return etl_result

    @staticmethod
    def join(
        table_name: str,
        how: str,
        on: str,
        df: Optional[Any] = None,
        df_rhs: Optional[Any] = None,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        join_sql = sql_ctor.construct(
            etl_op_name=inspect.currentframe().f_code.co_name,
            table_name=table_name,
            how=how,
            on=on,
        )
        etl_interface.execute_sql("rawdata", join_sql)

        return None
