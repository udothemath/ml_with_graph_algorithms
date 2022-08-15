"""ETL operation profiler."""
import json
import logging
import os
import sys
from collections import namedtuple
from importlib import import_module
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

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

        self._query_parser = QueryParser()

    def run(self) -> None:
        """Run ETL operation profiling process.

        **Return**: None
        """
        etl_func, kwargs = self._query_parser.parse(self.query)
        df = self._load_data()
        if etl_func.__name__ == "join":
            df_rhs = self._load_data_rhs(kwargs["on"])
            kwargs = {"df_rhs": df_rhs, **kwargs}

        profile_results = self._profile(etl_func, df, **kwargs)
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

    def _load_data(self) -> Any:  # Tmp workaround for type annotation
        """Load and return dataset for profiling.

        Return:
            df: dataset for profiling
        """
        if self.mode == "vaex":
            df = self._pd.open(self.input_file)
        else:
            df = self._pd.read_parquet(self.input_file)

        return df

    def _load_data_rhs(self, on: str) -> Any:
        """Load dataset on right-hand side for `join` operation.

        Parameters:
            on: column to join on

        Return:
            df_rhs: dataset on right-hand side
        """
        input_file_rhs = self.input_file.replace("lhs", f"rhs_{on.split('_')[-1]}")
        if self.mode == "vaex":
            df_rhs = self._pd.open(input_file_rhs)
        else:
            df_rhs = self._pd.read_parquet(input_file_rhs)

        return df_rhs

    def _profile(
        self,
        etl_func: Callable,
        df: Any,  # Tmp workaround for type annotation
        **kwargs: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        """Profile ETL operation for `n_profiles` rounds.

        Parameters:
            etl_func: ETL operation function
            df: dataset for profiling
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
            etl_result, profile_result = etl_func(df=df, mode=self.mode, **kwargs)
            assert profile_result is not None, "Please enable `return_prf` for decorated operation in `ETLOpZoo`."

            # Record profiling result in the current round
            profile_results["t_elapsed"].append(profile_result.t_elapsed)
            profile_results["peak_mem_usage"].append(profile_result.peak_mem_usage)
            profile_results["cpu_power"].append(profile_result.emission_summary.cpu_power)
            profile_results["gpu_power"].append(profile_result.emission_summary.gpu_power)
            profile_results["ram_power"].append(profile_result.emission_summary.ram_power)

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


class QueryParser:
    """ETL query parser."""

    def __init__(self) -> None:
        pass

    def parse(self, query: str) -> Tuple[Callable, Dict[str, Any]]:
        """Parse ETL operation query.

        **Parameters**:
        - `query`: ETL operation query

        **Return**:
        - `etl_func`: ETL operation function
        - `kwargs`: arguments passed to ETL operation function
        """
        etl_op_name, *etl_op_body = query.split(" ")
        etl_op_name = etl_op_name[1:-1]
        parse_etl_op_body: Callable = None

        if etl_op_name == "apply":
            parse_etl_op_body = self._parse_apply
        elif etl_op_name == "groupby":
            parse_etl_op_body = self._parse_groupby
        elif etl_op_name == "rolling":
            parse_etl_op_body = self._parse_rolling
        elif etl_op_name == "join":
            parse_etl_op_body = self._parse_join

        etl_func, kwargs = parse_etl_op_body(etl_op_body)

        return etl_func, kwargs

    def _parse_apply(self) -> None:
        pass

    def _parse_groupby(
        self,
        op_body: List[str],
    ) -> Tuple[Callable, Dict[str, Union[List[str], Dict[str, List[str]]]]]:
        """Parse `groupby` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: groupby operation function
            groupby_keys: keys to determine groups
            stats_dict: stats to derive for each column
        """
        etl_func = ETLOpZoo.groupby
        kwargs: Dict[str, Any] = {}

        groupby_keys: List[str] = []
        stats_dict: Dict[str, List[str]] = {}
        for i, ele in enumerate(op_body):
            if ele.startswith("int") or ele.startswith("float"):
                col_name = ele
                stats_dict[col_name] = []
            elif ele == "by":
                break
            else:
                stats_dict[col_name].append(ele.replace(",", ""))
        groupby_keys = op_body[i + 1 :]

        kwargs = {
            "groupby_keys": groupby_keys,
            "stats_dict": stats_dict,
        }

        return etl_func, kwargs

    def _parse_rolling(self) -> Tuple[Callable, Dict[str, Any]]:
        """Parse `rolling` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: rolling operation function

        """
        etl_func = ETLOpZoo.rolling
        kwargs: Dict[str, Any] = {}

        kwargs = {}

        return etl_func, kwargs

    def _parse_join(
        self,
        op_body: List[str],
    ) -> Tuple[Callable, Dict[str, str]]:
        """Parse `join` operation body.

        Parameters:
            op_body: ETL operation body

        Return:
            etl_func: join operation function
            how: type of `join` to perform
            on: column to join on
        """
        etl_func = ETLOpZoo.join
        kwargs: Dict[str, Any] = {}

        kwargs = {
            "how": op_body[0],
            "on": op_body[2],
        }

        return etl_func, kwargs


class ETLOpZoo:
    """ETL operation zoo.

    See also www.learncodewithmike.com/2020/01/python-method.html.
    """

    @staticmethod
    @Profiler.profile_factory(return_prf=True)
    def groupby(
        df: Any,
        mode: str,
        groupby_keys: List[str],
        stats_dict: Dict[str, List[str]],
    ) -> Any:
        """Group samples and derive stats for each group."""
        etl_result = df.groupby(groupby_keys).agg(stats_dict)

        return etl_result

    @staticmethod
    @Profiler.profile_factory(return_prf=True)
    def rolling(
        df: Any,
        mode: str,
    ) -> Any:
        """Derive rolling stats."""
        etl_result = None
        return etl_result

    @staticmethod
    @Profiler.profile_factory(return_prf=True)
    def join(
        df: Any,
        mode: str,
        df_rhs: Any,
        how: str,
        on: str,
    ) -> Any:
        """Join datasets on left-hand and right-hand sides."""
        if mode in ["pandas", "cudf", "modin"]:
            etl_result = df.merge(df_rhs, how=how, on=on)
        elif mode == "polars":
            etl_result = df.join(df_rhs, how=how, on=on)
        else:
            etl_result = df.join(df_rhs, how=how, on=on, rsuffix="_rhs")

        return etl_result
