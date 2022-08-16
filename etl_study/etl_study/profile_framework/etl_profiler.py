"""ETL operation profiler."""
import gc
import json
import logging
import os
import sys
from collections import namedtuple
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from profile_framework.query_parser import QueryParser
from profile_framework.utils.profile import Profiler

# Define ETL operation profiling result schema
ETLProfileResult = namedtuple(
    "ETLProfileResult", ["t_elapsed", "peak_mem_usage", "cpu_power", "gpu_power", "ram_power"]
)


class ETLProfiler:
    """ETL operation profiler.

    **Parameters**:
    - `query`: ETL operation query
    - `input_file`: input file
    - `mode`: execution mode, the choices are as follows:
        - `pandas`: Pandas without acceleration
        - `cudf`: RAPIDS cuDF with GPU acceleration
        - `modin`: Replacement for Pandas using Ray or Dask as engine
        - `polars`: A lightning fast DataFrame library written in rust
        - `vaex`: Python library for lazy out-of-core DataFrame
    - `n_profiles`: number of profilings to run

    **Attributes**:
    - `etl_profile_result_`: ETL operation profiling result
    """

    etl_profile_result_: ETLProfileResult = None

    def __init__(
        self,
        query: str,
        input_file: str,
        mode: str = "pandas",
        n_profiles: int = 5,
    ):
        self.query = query
        self.input_file = input_file
        self.mode = mode
        self.n_profiles = n_profiles

        self._log_meta()
        self._setup()

    def _log_meta(self) -> None:
        """Log profiling metadata.

        Return:
            None
        """
        logging.info("=====Welcome to Profiling World=====")
        logging.info(f"Query: {self.query}")
        logging.info(f"Input file: {self.input_file}")
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Profiling runs for {self.n_profiles} rounds...\n")

    def _setup(self) -> None:
        """Setup ETL profiler.

        The setup process includes configuring execution mode with the
        corresponding library, and initialize query parser and general
        profiler.

        See also https://stackoverflow.com/questions/20096499.

        Return:
            None
        """
        mode = "modin.pandas" if self.mode == "modin" else self.mode
        try:
            self._pd = import_module(mode)
        except ImportError as e:
            print(f"Mode {mode} isn't supported in the current environment.")
            sys.exit(1)

        if self.mode == "pandas":
            self._etl_op_zoo = import_module("profile_framework.etl_op_zoo.base").BaseETLOpZoo
        else:
            self._etl_op_zoo = import_module(f"profile_framework.etl_op_zoo.{self.mode}").ETLOpZoo
        self._query_parser = QueryParser(self._etl_op_zoo)

    def run(self) -> None:
        """Run ETL operation profiling process.

        **Return**: None
        """
        etl_func, kwargs = self._query_parser.parse(self.query)
        df_lhs, df_rhs = self._prepare_data(etl_func, **kwargs)
        if df_rhs is not None:
            kwargs = {"df_rhs": df_rhs, **kwargs}

        profile_results = self._profile(etl_func, df_lhs, **kwargs)
        self._summarize(profile_results)
        self._log_summary()

    def dump_profile_result(self, dump_path: str) -> None:
        """Dump ETL operation profiling result.

        **Parameters**:
        - `dump_path`: path to dump profiling result

        **Return**:
            None
        """
        with open(os.path.join(dump_path, "prf.json"), "w") as f:
            json.dump(self.etl_profile_result_._asdict(), f)

    def add_profile_result_to_berk(self, berk_path: str) -> None:
        """Record profiling result for further benchmarking.

        `berk_path` records profiling results in different scenarios,
        facilitating further benchmarking (e.g., performance ranking,
        visualization).

        **Parameters**:
        - `berk_path`: path of the file recording profiling result

        **Return**:
            None
        """
        flatten = lambda x: {
            f"{prf_ind_name}-{stats_name}": stats_val
            for prf_ind_name, prf_stats in x._asdict().items()
            for stats_name, stats_val in prf_stats.items()
        }

        etl_profile_result = flatten(self.etl_profile_result_)
        etl_profile_result = {
            "query": self.query,
            "input_file": self.input_file,
            "mode": self.mode,
            "n_profiles": self.n_profiles,
            **etl_profile_result,
        }
        berk_new_row = pd.DataFrame(etl_profile_result, index=[0])
        if os.path.exists(berk_path):
            berk = pd.read_csv(berk_path)
            berk = pd.concat([berk, berk_new_row], ignore_index=True)
        else:
            berk = berk_new_row

        berk.to_csv(berk_path, index=False)

    def _prepare_data(
        self,
        etl_func: Callable,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Prepare all the datasets needed for profiling.

        Parameters:
            etl_func: ETL operation function
            kwargs: arguments passed to ETL operation function

        Return:
            df_lhs: dataset on left-hand side
            df_rhs: dataset on right-hand side (only for `join`)
        """
        etl_func_name = etl_func.__name__
        df_lhs, df_rhs = None, None

        if not etl_func_name.startswith("read"):
            df_lhs = self._load_data()

            if etl_func_name == "join":
                input_file_rhs = self.input_file.replace("lhs", f"rhs_{kwargs['on'].split('_')[-1]}")
                df_rhs = self._load_data(input_file_rhs)
        else:
            logging.warning("Argument --input-file is ignored when profiled ETL operation is [read_<channel>].")

        return df_lhs, df_rhs

    def _load_data(self, input_file: Optional[str] = None) -> Any:  # Tmp workaround for type annotation
        """Load and return a single dataset for profiling.

        Parameters:
            input_file: input file
                *Note: `self.input_file` is used if this `input_file`
                    is not specified

        Return:
            df: dataset for profiling
        """
        input_file = input_file if input_file is not None else self.input_file
        df = self._etl_op_zoo.read_parquet(input_file)

        return df

    def _profile(
        self,
        etl_func: Callable,
        df_lhs: Optional[Any] = None,  # Tmp workaround for type annotation
        **kwargs: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        """Profile ETL operation for `n_profiles` rounds.

        Parameters:
            etl_func: ETL operation function
            df_lhs: dataset on left-hand side
            kwargs: arguments passed to ETL operation function

        Return:
            profile_results: performance recorded round-by-round
        """
        profile_results: Dict[str, List[float]] = {
            "t_elapsed": [],
            "peak_mem_usage": [],
            "cpu_power": [],
            "gpu_power": [],
            "ram_power": [],
        }

        for i in range(self.n_profiles):
            etl_result, profile_result = Profiler.profile_factory(return_prf=True)(etl_func)(df=df_lhs, **kwargs)
            assert profile_result is not None, "Please enable `return_prf` for decorated operation in `ETLOpZoo`."

            # Record profiling result in the current round
            profile_results["t_elapsed"].append(profile_result.t_elapsed)
            profile_results["peak_mem_usage"].append(profile_result.peak_mem_usage)
            profile_results["cpu_power"].append(profile_result.emission_summary.cpu_power)
            profile_results["gpu_power"].append(profile_result.emission_summary.gpu_power)
            profile_results["ram_power"].append(profile_result.emission_summary.ram_power)

            del etl_result, profile_result
            gc.collect()

        return profile_results

    def _summarize(self, profile_results: Dict[str, List[float]]) -> None:
        """Summarize ETL operation profiling results.

        Parameters:
            profile_results: performance recorded round-by-round

        Return:
            None
        """

        def summarize(prf_vals: List[float]) -> Dict[str, float]:
            """Derive average and standard deviation of performance
            values from different profiling rounds.

            Parameters:
                prf_vals: performance values of a single indicator

            Return:
                prf_stats: average and standard deviation of
                    performance values
            """
            prf_stats = {
                "avg": np.mean(prf_vals),
                "std": np.std(prf_vals),
            }

            return prf_stats

        prf_summary = {}
        for prf_ind_name, prf_vals in profile_results.items():
            prf_summary[prf_ind_name] = summarize(prf_vals)

        self.etl_profile_result_ = ETLProfileResult(**prf_summary)

    def _log_summary(self) -> None:
        """Log profiling summary.

        Return:
            None
        """
        logging.info("=====Summary=====")
        for prf_ind, unit in zip(self.etl_profile_result_._fields, ["sec", "MiB", "W", "W", "W"]):
            prf = getattr(self.etl_profile_result_, prf_ind)
            logging.info(f"{prf_ind}: {prf['avg']:.4f} ± {prf['std']:.4f} {unit}")
        logging.info("=====Profiling Success=====")
