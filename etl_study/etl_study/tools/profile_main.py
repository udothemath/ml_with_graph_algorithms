"""Main script for profiling ETL opration."""
import json
import os
from argparse import Namespace
from datetime import datetime

import pandas as pd

from engine.defaults import BaseArgParser
from profile_framework.etl_profiler import ETLProfiler, ETLProfileResult
from utils.logger import Logger


class ETLProfileArgParser(BaseArgParser):
    """Argument parser for ETL operation profiling."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--query",
            type=str,
            default=None,
            help="ETL operation query",
        )
        self.argparser.add_argument(
            "--input-file",
            type=str,
            default=None,
            help="input file",
        )
        self.argparser.add_argument(
            "--mode",
            type=str,
            choices=["pandas", "cudf", "modin", "polars", "vaex"],
            default=None,
            help="execution mode",
        )
        self.argparser.add_argument(
            "--n-profiles",
            type=int,
            default=None,
            help="number of profilings to run",
        )
        self.argparser.add_argument(
            "--to-benchmark",
            type=self._str2bool,
            default=None,
            help="whether to append profiling result for further benchmarking",
        )


def _setup_logging_path(query: str) -> str:
    """Return unique logging path for current profiling process.

    Parameters:
        query: ETL operation query

    Return:
        logging_path: logging path
    """

    def check_logging_root() -> None:
        """Check if root directory for loggin exists.

        Return:
            None
        """
        assert os.path.exists("./profile_result"), "Please make directory `profile_result` in current dir."

    etl_op_name = query.split(" ")[0][1:-1]
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    profile_id = f"{etl_op_name}-{cur_time}"

    check_logging_root()
    logging_path = os.path.join("./profile_result", profile_id)
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)

    return logging_path


def _dump_profile_result(etl_profile_result: ETLProfileResult, logging_path: str) -> None:
    """Dump ETL operation profiling result.

    Parameters:
        etl_profile_result: ETL operation profiling result
        logging_path: logging path

    Return:
        None
    """
    with open(os.path.join(logging_path, "prf.json"), "w") as f:
        json.dump(etl_profile_result._asdict(), f)


def _add_profile_result_to_berk(args: Namespace, etl_profile_result: ETLProfileResult) -> None:
    """Append profiling result for further benchmarking.

    Parameters:
        args: arguments driving ETL operation profiling process
        etl_profile_result: ETL operation profiling result

    Return:
        None
    """
    flatten = lambda x: {
        f"{prf_ind_name}-{stats_name}": prf_val
        for prf_ind_name, prf_summary in x._asdict().items()
        for stats_name, prf_val in prf_summary.items()
    }

    etl_profile_result = flatten(etl_profile_result)
    etl_profile_result = {
        "query": args.query,
        "input_file": args.input_file,
        "mode": args.mode,
        "n_profiles": args.n_profiles,
        **etl_profile_result,
    }
    berk_new_row = pd.DataFrame(etl_profile_result, index=[0])
    if os.path.exists("./berk.csv"):  # Tmp fixed
        berk = pd.read_csv("./berk.csv")
        berk = pd.concat([berk, berk_new_row], ignore_index=True)
    else:
        berk = berk_new_row

    berk.to_csv("./berk.csv", index=False)


def main(args: Namespace) -> None:
    """Run ETL operation profiling process.

    Parameters:
        args: arguments driving ETL operation profiling process
    """
    # Configure profiling process
    query = args.query
    input_file = args.input_file
    mode = args.mode
    n_profiles = args.n_profiles

    # Setup logging path and logger
    logging_path = _setup_logging_path(query)
    logger = Logger("INFO", os.path.join(logging_path, "profile.log")).get_logger()

    # Start profiling
    etl_profiler = ETLProfiler(
        query=args.query,
        input_file=args.input_file,
        mode=args.mode,
        n_profiles=args.n_profiles,
    )
    etl_profiler.run()

    # Dump profiling result
    _dump_profile_result(etl_profiler.etl_profile_result_, logging_path)

    # Append profiling result for further benchmarking
    if args.to_benchmark:
        _add_profile_result_to_berk(args, etl_profiler.etl_profile_result_)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = ETLProfileArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
