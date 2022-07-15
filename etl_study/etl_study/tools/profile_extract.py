"""
Main script for profiling extraction operations.

Todo:
1. Support multiple raw tables
2. Enable logging or plotting for profiling result (e.g., avg +- std)
3. dtype study (cf. Extension)
    -> Use files containing different dtypes to test if dtype has an
        impact on extraction efficiency (e.g., btnk at df conversion)
4. Whether to merge profiling for extraction & loading
    -> Arg to control "E" | "L" with corresponding ETL func and sql
        building
"""
from argparse import Namespace
from typing import Any, Callable, Dict

import pandas as pd

from common.etl_base import ETLBase
from engine.defaults import ExtrProfileArgParser
from metadata import ECG
from utils.profile import Profiler

# Variable definitions
DB_NAME = "rawdata"
SCHEMA_NAME = "tmp"


class ExtrProfiler(ETLBase, Profiler):
    """Profiler for extraction operations."""

    def __init__(self) -> None:
        super(ExtrProfiler, self).__init__()

    @Profiler.profile
    def select_table(self, db_name: str, sql: str) -> pd.DataFrame:
        return super().select_table(
            db_name=db_name,
            sql=sql,
        )

    @Profiler.profile
    def select_table_stream(
        self,
        db_name: str,
        sql: str,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        return super().select_table_fast(
            db_name=db_name,
            sql=sql,
            encoding=encoding,
        )


def _build_extr_func(extr_method: str) -> Callable:
    """Return extraction function to profile.

    Parameters:
        extr_method: extraction method

    Retunr:
        extr_func: extraction function
    """
    extr_profiler = ExtrProfiler()

    if extr_method == "naive":
        extr_func = extr_profiler.select_table
    elif extr_method == "stream":
        extr_func = extr_profiler.select_table_stream

    return extr_func


def _build_sql(table_name: str, n_samples: int, n_cols: int) -> str:
    """Return SQL logic.

    Parameters:
        table_name: the table name
        n_samples: number of samples to extract
        n_cols: number of columns to extract

    Return:
        sql: SQL logic
    """
    table = f"{SCHEMA_NAME}.{table_name}"

    table_meta: Dict[str, Any] = ECG
    assert n_samples <= table_meta["N_SAMPLES"], f"n_samples should be lower than {table_meta['N_SAMPLES']}"
    assert n_cols <= table_meta["N_COLS"], f"n_cols should be lower than {table_meta['N_COLS']}"

    limit = f"LIMIT {n_samples}"
    cols = ", ".join(table_meta["COLUMNS"][:n_cols])
    sql = f"SELECT {cols} FROM {table} " + limit

    return sql


def main(args: Namespace) -> None:
    """Profile the specified ETL extraction operation.

    Parameters:
        args: arguments driving profiling process

    Return:
        None
    """
    # Configure profiling process
    table_name = args.table_name
    n_samples = args.n_samples
    n_cols = args.n_cols
    extr_method = args.extraction_method
    n_profiles = args.n_profiles

    # Build extraction operation to profile
    extr_func = _build_extr_func(extr_method)

    # Construct SQL logic
    sql = _build_sql(table_name, n_samples, n_cols)

    # Start profiling
    for i in range(n_profiles):
        extr_func(db_name=DB_NAME, sql=sql)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = ExtrProfileArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
