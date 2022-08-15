"""
Main script for profiling DataFrame operations in either pandas or cuDF
mode.
"""
import logging
from argparse import Namespace
from typing import List, Union

import cudf
import numpy as np
import pandas as pd

from common.etl_base import ETLBase
from engine.defaults import BaseArgParser
from rapids_study.cudf_logic import CUDFLogic
from rapids_study.pd_logic import PDLogic
from utils.profile import Profiler

logging.getLogger("numba").setLevel(logging.WARNING)
DB_NAME = "rawdata"


class DFProfileArgParser(BaseArgParser):
    """Argument parser for profiling DataFrame operations."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--table-name",
            type=str,
            default=None,
            help="table name",
        )
        self.argparser.add_argument(
            "--input-cols",
            type=str,
            default=None,
            nargs="+",
            help="input column names",
        )
        self.argparser.add_argument(
            "--null-col",
            type=str,
            default=None,
            help="column having null values",
        )
        self.argparser.add_argument(
            "--n-samples",
            type=int,
            default=None,
            help="number of samples (tuples) to use",
        )
        self.argparser.add_argument(
            "--n-profiles",
            type=int,
            default=None,
            help="number of profilings for the monitored operation",
        )
        self.argparser.add_argument(
            "--logic-mode", type=str, choices=["pd", "cudf"], default="cudf", help="logic mode"
        )


def _load_data(
    table_name: str,
    input_cols: List[str],
    null_col: str,
    n_samples: int,
    logic_mode: str,
) -> Union[pd.DataFrame, cudf.DataFrame]:
    """Load and return DataFrame for profiling.

    Parameters:
        table_name: table name
        input_cols: input column names
        null_col: column having null values
        n_samples: number of samples (tuples) to use
        logic_mode: logic mode

    Return:
        df: DataFrame for profiling
    """
    # Construct SQL logic
    sql = f"SELECT {', '.join(input_cols)} FROM tmp.{table_name} "
    if null_col is not None:
        sql += f"WHERE {null_col} is NULL "
    sql += f"LIMIT {n_samples}"

    etl = ETLBase()
    df = etl.select_table_fast(db_name=DB_NAME, sql=sql)
    assert len(df) == n_samples, "#Sample of SQL query result doesn't match requested one."
    if logic_mode == "cudf":
        df = cudf.DataFrame(df)

    return df


def _log_final_report(
    logic_mode: str,
    t_elapsed: List[float],
    peak_mem_usage: List[float],
) -> None:
    """Log final performance report.

    Parameters:
        logic_mode: logic mode
        t_elapsed: time consumption
        peak_mem_usage: peak memory usage

    Return:
        None
    """
    # Derive performance stats
    if logic_mode == "cudf":
        t_compile, t_elapsed = t_elapsed[0], t_elapsed[1:]
        m_compile, peak_mem_usage = peak_mem_usage[0], peak_mem_usage[1:]
    t_avg, t_std = np.mean(t_elapsed), np.std(t_elapsed)
    m_avg, m_std = np.mean(peak_mem_usage), np.std(peak_mem_usage)

    # Log final report
    logging.info("\n=====Final Report=====")
    logging.info(f"Logic mode: {logic_mode}")
    logging.info(f"Average time: {t_avg:.4f} ± {t_std:.4f} sec")
    logging.info(f"Peak memory usage: {m_avg:.4f} ± {m_std:.4f} MiB")
    if logic_mode == "cudf":
        logging.info(f"Time in cuDF compilation round: {t_compile:.4f} sec")
        logging.info(f"Peak memory usage in cuDF compilation round: {m_compile:.4f} MiB")


def main(args: Namespace) -> None:
    # Configure profiling process
    table_name = args.table_name
    input_cols = args.input_cols
    null_col = args.null_col
    n_samples = args.n_samples
    n_profiles = args.n_profiles
    logic_mode = args.logic_mode

    # Load data
    df = _load_data(table_name, input_cols, null_col, n_samples, logic_mode)

    # Build operation
    op = PDLogic().process_df if logic_mode == "pd" else CUDFLogic().process_df

    # Start profiling
    profiler = Profiler()
    t_elapsed: List[float] = []
    peak_mem_usage: List[float] = []

    for i in range(n_profiles):
        _, (t, m) = profiler.run(op, df)
        t_elapsed.append(t)
        peak_mem_usage.append(m)

    # Log final report
    _log_final_report(logic_mode, t_elapsed, peak_mem_usage)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = DFProfileArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
