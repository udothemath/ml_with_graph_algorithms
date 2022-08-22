"""Main script for generating synthetic dataset for `join` only.

Annotation:
- LHS: the table on left-hand side
- RHS: the table on right-hand side

Note:
In the profile framework, .parquet file name is the same as the table
name in PostgreSQL database. Because "+" character can't exist in the
table name in PostgreSQL, "+" in the scientific notation is removed
before the generated dataset is dumped.

Example:
python -m tools.gen_join
       --n-samples 10000000
       --n-int-features 1
       --n-float-features 1
       --join-key-common-ratio 0.9
       --random-state 42
       --output-path ./data/raw/synthetic/join/

After running the command above, the synthetic datasets are generated
with the following appearance:

`join_1e07_lhs.parquet`
| id_l    | id_m | id_s | int_f0 | float_f0 |
| ------- | ---- | ---- | ------ | -------- |
| 5500567 | 4585 | 4    | 984    | 0.392675 |
| 2913422 | 331  | 5    | 430    | 0.372806 |
                    .
                    .
                    .
          10000000 rows in total

`join_1e07_rhs_l.parquet`
| id_l    | int_f0_rhs |
| ------- | ---------- |
| 5500567 | 102        |
| 2913422 | 435        |
            .
            .
            .
  10000000 rows in total

`join_1e07_rhs_m.parquet`
| id_m  | int_f0_rhs |
| ----- | ---------- |
| 10747 | 102        |
| 5985  | 435        |
         .
         .
         .
 10000 rows in total

`join_1e07_rhs_s.parquet`
| id_s  | int_f0_rhs |
| ----- | ---------- |
| 2     | 102        |
| 4     | 435        |
         .
         .
         .
  10 rows in total
"""
import gc
import os
from argparse import Namespace
from typing import Dict

import numpy as np
import pandas as pd

from engine.defaults import BaseArgParser
from profile_framework.data_generator import DataGenerator

CARD_REGULATORS = {
    "l": 1,
    "m": 1e3,
    "s": 1e6,
}  # Regulate cardinality of each `id` in LHS
CARD_LEVEL = {"l": "highest", "m": "medium", "s": "lowest"}


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
            help="number of samples in the table on left-hand side",
        )
        self.argparser.add_argument(
            "--n-int-features",
            type=int,
            default=None,
            help="number of integer features in the table on left-hand side",
        )
        self.argparser.add_argument(
            "--n-float-features",
            type=int,
            default=None,
            help="number of floating-point features in the table on left-hand side",
        )
        self.argparser.add_argument(
            "--join-key-common-ratio",
            type=float,
            default=None,
            help="percentage of samples with common keys to join on",
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


def _gen_lhs_id(n_samples: int, join_key_common_ratio: float, output_path: str) -> pd.DataFrame:
    """Generate identifiers of LHS.

    To dispatch identifiers to LHS and corresponding RHSs, RHSs are
    created and dumped during the generation process.

    Parameters:
        n_samples: number of samples in LHS
        join_key_common_ratio: percentage of samples with common keys
            to join on
        output_path: output path to dump synthetic data

    Return:
        df_lhs_id: identifiers of LHS
    """
    df_lhs_id = pd.DataFrame()

    for rhs_size, card_reg in CARD_REGULATORS.items():
        print(f"Generating identifier base with {CARD_LEVEL[rhs_size]} cardinality...")
        id_n_unique = int(n_samples / card_reg)
        ids_base = _get_ids_base(id_n_unique, join_key_common_ratio)

        ids_lhs = np.concatenate((ids_base["common"], ids_base["lhs_only"]))
        ids_rhs = np.concatenate((ids_base["common"], ids_base["rhs_only"]))

        if len(ids_lhs) < n_samples:
            df_lhs_id[f"id_{rhs_size}"] = np.random.choice(ids_lhs, n_samples)
        else:
            df_lhs_id[f"id_{rhs_size}"] = ids_lhs

        # Generate RHS
        print(f"Generating the table on right-hand side with size '{rhs_size.upper()}'...")
        df_rhs_id = pd.DataFrame(ids_rhs, columns=[f"id_{rhs_size}"])
        df_rhs_val = DataGenerator(n_samples=id_n_unique, n_int_features=1).run()
        df_rhs_val.columns = [f"{col}_rhs" for col in df_rhs_val.columns]
        df_rhs = pd.concat([df_rhs_id, df_rhs_val], axis=1)
        df_rhs.to_parquet(
            os.path.join(output_path, f"join_{n_samples:.0e}_rhs_{rhs_size}.parquet".replace("+", "")), index=False
        )
        print("Done.\n")

        del df_rhs
        _ = gc.collect()

    return df_lhs_id


def _get_ids_base(id_n_unique: int, join_key_common_ratio: float) -> Dict[str, np.ndarray]:
    """Return identifier candidate pool for both LHS and RHS.

    To generate table-dependent (either LHS or RHS) identifiers, number
    of unique identifiers will be increased in the beginning.

    Parameters:
        id_n_unique: number of unique identifiers
        join_key_common_ratio: percentage of samples with common keys

    Return:
        ids_base: identifier candidate pool
    """
    id_n_unique_amp = int(id_n_unique * (1 + (1 - join_key_common_ratio)))
    ids_perm = np.random.permutation(id_n_unique_amp)

    ids_base = {
        "common": ids_perm[0 : int(id_n_unique * join_key_common_ratio)],
        "lhs_only": ids_perm[int(id_n_unique * join_key_common_ratio) : id_n_unique],
        "rhs_only": ids_perm[id_n_unique:],
    }

    return ids_base


def main(args: Namespace) -> None:
    """Generate synthetic dataset.

    Parameters:
        args: arguments driving synthetic data generation process.
    """
    # Configure data generation process
    n_samples = args.n_samples
    n_int_features = args.n_int_features
    n_float_features = args.n_float_features
    join_key_common_ratio = args.join_key_common_ratio
    random_state = args.random_state
    output_path = args.output_path

    assert n_samples > CARD_REGULATORS["s"], f"Number of the samples must be larger than {int(CARD_REGULATORS['s'])}."

    # Start generating synthetic data
    df_lhs_id = _gen_lhs_id(n_samples, join_key_common_ratio, output_path)
    print("Generating value columns for the table on left-hand side...")
    df_lhs_val = DataGenerator(
        n_samples=n_samples,
        n_int_features=n_int_features,
        n_float_features=n_float_features,
        random_state=random_state,
    ).run()
    print("Done.")
    df_lhs = pd.concat([df_lhs_id, df_lhs_val], axis=1)
    df_lhs.to_parquet(os.path.join(output_path, f"join_{n_samples:.0e}_lhs.parquet".replace("+", "")), index=False)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = DGArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
