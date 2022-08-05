"""ETL operation profiler."""
import logging
import sys
from importlib import import_module
from typing import Any, Callable, Dict, List, Tuple, Union

from profile_framework.utils.profile import Profiler


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
    - `t_elapsed_`: time consumption
    - `peak_mem_usage_`: peak memory usage
    """

    t_elapsed_: List[float] = []
    peak_mem_usage_: List[float] = []

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

    def run(self) -> None:
        """Run ETL operation profiling process.

        **Return**: None
        """
        etl_func, kwargs = self._query_parser.parse(self.query)
        df = self._load_data()

        self._profile(etl_func, df, **kwargs)

    #         self._summarize()

    def _log_meta(self) -> None:
        """Log profiling metadata.

        Return:
            None
        """
        logging.info("=====Welcome to Profiling World=====")
        logging.info(f"Query: {self.query}")
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
        self._profiler = Profiler()

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

    def _profile(
        self,
        etl_func: Callable,
        df: Any,  # Tmp workaround for type annotation
        **kwargs: Dict[str, Any],
    ) -> None:
        """Profile ETL operation for `n_profiles` rounds.

        Parameters:
            etl_func: ETL operation function
            df: dataset for profiling
            kwargs: arguments passed to ETL operation function

        Return:
            None
        """
        for i in range(self.n_profiles):
            etl_result, (t_elapsed, peak_mem_usage) = etl_func(df=df, **kwargs)
            self.t_elapsed_.append(t_elapsed)
            self.peak_mem_usage_.append(peak_mem_usage)

    def _summarize(self) -> None:
        """Summarize profiling performance."""
        pass


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

    def _parse_rolling(self) -> None:
        pass

    def _parse_join(self) -> None:
        pass


class ETLOpZoo:
    """ETL operation zoo.

    See also www.learncodewithmike.com/2020/01/python-method.html.
    """

    @staticmethod
    @Profiler.profile_factory(return_prf=True)
    def groupby(
        df: Any,
        groupby_keys: List[str],
        stats_dict: Dict[str, List[str]],
    ) -> Any:
        """Group samples and derive stats for each group."""
        etl_result = df.groupby(groupby_keys).agg(stats_dict)

        return etl_result
