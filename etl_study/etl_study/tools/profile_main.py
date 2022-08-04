"""Main script for profiling ETL opration."""
from argparse import Namespace

from engine.defaults import BaseArgParser
from profile_framework.etl_profiler import ETLProfiler


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

    # Start profiling
    etl_profiler = ETLProfiler(
        query=query,
        input_file=input_file,
        mode=mode,
        n_profiles=n_profiles,
    )
    etl_profiler.run()


if __name__ == "__main__":
    # Parse arguments
    arg_parser = ETLProfileArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
