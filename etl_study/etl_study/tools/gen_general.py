"""Main script for generating synthetic dataset for general operations.

General operations include some of the commonly used APIs (e.g., join,
groupby, rolling).
"""
import os
from argparse import Namespace

from engine.defaults import BaseArgParser
from profile_framework.data_generator import DataGenerator


class DGArgParser(BaseArgParser):
    """Argument parser for synthetic data generation."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--n-samples",
            type=int,
            default=None,
            help="number of samples",
        )
        self.argparser.add_argument(
            "--n-str-ids",
            type=int,
            default=None,
            help="number of string identifiers",
        )
        self.argparser.add_argument(
            "--n-int-ids",
            type=int,
            default=None,
            help="number of integer identifiers",
        )
        self.argparser.add_argument(
            "--n-clusts-per-id",
            type=int,
            default=None,
            help="number of clusters in each identifier",
        )
        self.argparser.add_argument(
            "--n-int-features",
            type=int,
            default=None,
            help="number of integer features",
        )
        self.argparser.add_argument(
            "--n-float-features",
            type=int,
            default=None,
            help="number of floating-point features",
        )
        self.argparser.add_argument(
            "--random-state",
            type=int,
            default=None,
            help="control randomness of data generation process",
        )
        self.argparser.add_argument(
            "--output-path",
            type=str,
            default=None,
            help="output path to dump synthetic data",
        )


def main(args: Namespace) -> None:
    """Generate synthetic dataset.

    Parameters:
        args: arguments driving synthetic data generation process.
    """
    # Configure data generation process
    n_samples = args.n_samples
    n_str_ids = args.n_str_ids
    n_int_ids = args.n_int_ids
    n_clusts_per_id = args.n_clusts_per_id
    n_int_features = args.n_int_features
    n_float_features = args.n_float_features
    random_state = args.random_state
    output_path = args.output_path

    # Start generating synthetic data
    print("Generating the table for general operations...")
    df = DataGenerator(
        n_samples=n_samples,
        n_str_ids=n_str_ids,
        n_int_ids=n_int_ids,
        n_clusts_per_id=n_clusts_per_id,
        n_int_features=n_int_features,
        n_float_features=n_float_features,
        random_state=random_state,
    ).run()
    print("Done.")

    # Dump synthetic data
    dump_path = os.path.join(output_path, f"general_{n_samples:.0e}.parquet".replace("+", ""))
    df.to_parquet(dump_path, index=False)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = DGArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
