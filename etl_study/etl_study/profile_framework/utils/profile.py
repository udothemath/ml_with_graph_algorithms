"""Profiler."""
import inspect
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

from memory_profiler import memory_usage

# Configure standard output logger (DeprecationWarning)
logging.basicConfig(level=logging.INFO)


class Profiler:
    """Profiler providing function decorator for performance report."""

    def __init__(self) -> None:
        pass

    def profile_factory(return_prf: bool) -> Any:
        def profile(func: Callable) -> Any:
            @wraps(func)
            def inner(*args: Any, **kwargs: Dict[Any, Any]) -> Union[Tuple[Optional[Any], Tuple[float, float]]]:
                # Configure monitored function prototype
                func_kwargs = []
                for k, v in kwargs.items():
                    if isinstance(v, str):
                        func_kwargs.append(f'{k}="{v}"')
                    else:
                        func_kwargs.append(f"{k}={v}")
                func_kwargs = ", ".join(func_kwargs)
                func_prototype = f"{func.__name__}({func_kwargs})"

                # Monitor time and memory usage of the function under test
                t_start = time.perf_counter()
                mem, result = memory_usage((func, args, kwargs), retval=True, timeout=200, interval=1e-7)
                t_elapsed = time.perf_counter() - t_start
                peak_mem_usage = max(mem) - min(mem)

                # Log performace report
                logging.info("=====Profiling=====")
                if inspect.ismethod(func):
                    logging.info(f"Class: {args[0].__class__.__name__}")
                logging.info(f"Function: {func_prototype}")
                logging.info(f"Time: {t_elapsed:.4f} sec")
                logging.info(f"Peak memory: {peak_mem_usage:.4f} MiB")
                logging.info("======End of Profiling=====\n")

                if return_prf:
                    return result, (t_elapsed, peak_mem_usage)
                else:
                    return result

            return inner

        return profile

    @profile_factory(return_prf=True)
    def run(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Optional[Any], Tuple[float, float]]:
        """Run profiling and return performance report."""
        assert callable(func), "Function to profile must be callable."

        return func(*args, **kwargs)
