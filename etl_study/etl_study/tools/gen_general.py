"""Main script for generating synthetic dataset for general operations.

General operations include some of the commonly used APIs (e.g., apply,
groupby, rolling).

Note:
In the profile framework, .parquet file name is the same as the table
name in PostgreSQL database. Because "+" character can't exist in the
table name in PostgreSQL, "+" in the scientific notation is removed
before the generated dataset is dumped.

Example:
python -m tools.gen_general
       --n-samples 10000000
       --n-str-ids 2
       --n-int-ids 2
       --n-clusts-per-id 100
       --n-int-features 1
       --n-float-features 1
       --random-state 42
       --output-path ./data/raw/synthetic/general/

After running the command above, the synthetic dataset is generated
with the following appearance:

`general_1e7.parquet`
| str_id0 | str_id_hc | int_id0 | int_id_hc | int_f0 | float_f0 |
| ------- | --------- | ------- | --------- | ------ | -------- |
| c62     | c7663     | 16      | 44890     | 448    | 0.696174 |
| c33     | c98918    | 39      | 79923     | 215    | 0.675639 |
                                .
                                .
                                .
                      10000000 rows in total
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
