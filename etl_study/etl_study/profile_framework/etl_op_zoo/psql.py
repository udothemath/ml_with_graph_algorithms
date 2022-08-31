"""PostgreSQL ETL operation zoo."""
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import config_local as config
from common.etl_base import ETLBase

etl_interface = ETLBase()


class SQLConstructor:
    """SQL logic constructor.

    Given arguments passed to ETL operation function, the corresponding
    SQL logic is constructed.
    """

    def __init__(self) -> None:
        pass

    def construct(self, etl_op_name: str, table_name: str, **kwargs: Any) -> str:
        """Construct SQL logic.

        **Parameters**:
        - `etl_op_name`: ETL operation name
        - `table_name`: table name
        - `kwargs`: arguments passed to ETL operation function

        **Return**:
        - `sql`: constructed SQL logic
        """
        construct_sql: Callable = None
        if etl_op_name == "groupby":
            construct_sql = self._construct_groupby
        elif etl_op_name == "join":
            construct_sql = self._construct_join
        elif etl_op_name == "rolling":
            pass

        sql = construct_sql(table_name, **kwargs)

        return sql

    def _construct_groupby(
        self,
        table_name: str,
        groupby_keys: List[str],
        stats_dict: Dict[str, List[str]],
    ) -> str:
        """Construct SQL logic for deriving grouped stats.

        Parameters:
            table_name: table name
            groupby_keys: keys to determine groups
            stats_dict: stats to derive for each column

        Return:
            groupby_sql: groupby SQL logic
        """
        table_name = f"{config.SCHEMA_NAME}.{table_name}"

        # Construct selection SQL logic
        agg_feats, as_cols = self._get_agg_feats_with_as_cols(stats_dict)
        as_clause = self._get_as_clause(agg_feats, as_cols)
        slc_sql = f"""
            SELECT {', '.join(groupby_keys)}, {as_clause} FROM {table_name}
            GROUP BY {', '.join(groupby_keys)};
        """

        # Define output columns
        output_cols = groupby_keys + as_cols

        groupby_sql = self._construct_sql(output_cols, slc_sql)

        return groupby_sql

    def _construct_join(
        self,
        table_name: str,
        how: str,
        on: str,
    ) -> str:
        """Construct SQL logic for joining datasets on left-hand and
        right-hand sides.

        Parameters:
            table_name: table name
            how: type of `join` to perform
            on: column to join on

        Return:
            join_sql: join SQL logic
        """
        table_name_lhs = f"{config.SCHEMA_NAME}.{table_name}"
        table_name_rhs = table_name_lhs.replace("lhs", f"rhs_{on.split('_')[-1]}")

        # Construct selection SQL logic
        slc_sql = f"""
            SELECT * FROM {table_name_rhs}
            {how.upper()} JOIN {table_name_lhs} USING({on});
        """

        # Define output columns
        cols_lhs = self._get_cols(table_name_lhs)
        cols_rhs = self._get_cols(table_name_rhs)
        cols_lhs.remove(on)  # Remove duplicated join key
        output_cols = cols_rhs + cols_lhs

        join_sql = self._construct_sql(output_cols, slc_sql)

        return join_sql

    def _construct_sql(self, output_cols: List[str], slc_sql: str) -> str:
        """Construct ETL operation-specific SQL logic.

        Parameters:
            output_cols: output colunmn names
            slc_sql: selection SQL logic

        Return:
            sql: constructed SQL logic
        """
        output_col_dtypes = self._get_col_dtypes(output_cols)
        create_table_sql = self._get_create_table_sql(output_col_dtypes)
        insert_table_sql = self._get_insert_table_sql(slc_sql)

        sql = create_table_sql + insert_table_sql

        return sql

    def _get_agg_feats_with_as_cols(
        self,
        stats_dict: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Return feature names with aggregate functions and their
        corresponding column names used in `AS` clause.

        Example:
            >>> stats_dict = {"int_f0": ["min", "max"]}
            ...
            >>> agg_feats
            ['MIN(int_f0)', 'MAX(int_f0)']
            >>> as_cols
            ['int_f0_min', 'int_f0_max']

        Parameters:
            stats_dict: stats to derive for each column

        Return:
            agg_feats: feature name with aggregate function name
            as_cols: col name used in `AS` clause
        """
        agg_feats = []
        as_cols = []
        for feat, stats_list in stats_dict.items():
            for stats in stats_list:
                as_cols.append(f"{feat}_{stats}")
                agg_fn = "AVG" if stats == "mean" else stats.upper()
                agg_feats.append(f"{agg_fn}({feat})")

        return agg_feats, as_cols

    def _get_as_clause(self, slc_cols: List[str], as_cols: List[str]) -> str:
        """Return `AS` clause for selection SQL logic.

        Parameters:
            slc_cols: original column name
            as_cols: column alias

        Return:
            as_clause: `AS` clause
        """
        slc_col_as_col = []
        for slc_col, as_col in zip(slc_cols, as_cols):
            slc_col_as_col.append(f"{slc_col} AS {as_col}")
        as_clause = ", ".join(slc_col_as_col)

        return as_clause

    def _get_cols(self, table_name: str) -> List[str]:
        """Return column names of the given table.

        Parameters:
            table_name: table name

        Return:
            cols: column name
        """
        if "." in table_name:
            # If table schema exists
            table_name = table_name.split(".")[-1]

        cols = etl_interface.select_table(
            "rawdata",
            sql=f"""
                SELECT * FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{config.SCHEMA_NAME}' AND TABLE_NAME = '{table_name}';
            """,
        ).column_name.tolist()

        return cols

    def _get_col_dtypes(self, cols: List[str]) -> Dict[str, str]:
        """Return column data types.

        Parameters:
            cols: column name

        Return:
            col_dtypes: column name with dtype
        """
        col_dtypes = {}
        for col in cols:
            if ("mean" in col) or ("float" in col):
                col_dtypes[col] = "DOUBLE PRECISION"
            elif ("int" in col) or (col in ["id_l", "id_m", "id_s"]):
                col_dtypes[col] = "BIGINT"
            elif "str" in col:
                col_dtypes[col] = "TEXT"

        return col_dtypes

    def _get_create_table_sql(self, col_dtypes: Dict[str, str]) -> str:
        """Return create table SQL logic.

        Parameters:
            col_dtypes: column name with dtype

        Return:
            create_table_sql: create table SQL logic
        """
        create_table_sql = f"""
            CREATE TEMP TABLE output (
                {', '.join([f'{col} {dtype}' for col, dtype in col_dtypes.items()])}
            );
        """

        return create_table_sql

    def _get_insert_table_sql(self, slc_sql: str) -> str:
        """Return insert SQL logic.

        After insersion, temporary output table is dropped.

        Parameters:
            slc_sql: selection SQL logic

        Return:
            insert_table_sql: insert table SQL logic
        """
        insert_table_sql = f"""
            INSERT INTO output {slc_sql};
            DROP TABLE output;
        """

        return insert_table_sql


sql_ctor = SQLConstructor()


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
    ) -> None:
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
    ) -> None:
        """Join datasets on left-hand and right-hand sides."""
        join_sql = sql_ctor.construct(
            etl_op_name=inspect.currentframe().f_code.co_name,
            table_name=table_name,
            how=how,
            on=on,
        )
        etl_interface.execute_sql("rawdata", join_sql)

        return None
