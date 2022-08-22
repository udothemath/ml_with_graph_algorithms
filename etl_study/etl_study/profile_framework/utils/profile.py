"""Profiler."""
import inspect
import logging
import time
from collections import namedtuple
from functools import wraps
from typing import Any, Callable, Optional, Tuple

from codecarbon import OfflineEmissionsTracker
from memory_profiler import memory_usage

__all__ = [
    "Profiler",
]

# Define profiling result schema
ProfileResult = namedtuple("ProfileResult", ["t_elapsed", "peak_mem_usage", "emission_summary"])

emission_tracker = OfflineEmissionsTracker(
    country_iso_code="TWN",
    measure_power_secs=30,
    tracking_mode="process",
)


class Profiler:
    """Profiler providing function decorator for performance report."""

    def __init__(self) -> None:
        pass

    def profile_factory(return_prf: bool) -> Any:
        def profile(func: Callable) -> Any:
            @wraps(func)
            def inner(*args: Any, **kwargs: Any) -> Tuple[Any, Optional[ProfileResult]]:
                # Configure monitored function prototype
                func_kwargs = []
                for k, v in kwargs.items():
                    if isinstance(v, str):
                        func_kwargs.append(f'{k}="{v}"')
                    else:
                        func_kwargs.append(f"{k}={v}")
                func_kwargs = ", ".join(func_kwargs)
                func_prototype = f"{func.__name__}({func_kwargs})"

                # Monitor time, memory usage and energy consumption of
                # the function under test
                emission_tracker.start()
                t_start = time.perf_counter()
                mem, result = memory_usage((func, args, kwargs), retval=True, timeout=200, interval=1e-7)
                emission_tracker.stop()
                t_elapsed = time.perf_counter() - t_start
                peak_mem_usage = max(mem) - min(mem)
                emission_summary = emission_tracker.final_emissions_data

                # Log performace report
                logging.info("=====Profiling=====")
                if inspect.ismethod(func):
                    logging.info(f"Class: {args[0].__class__.__name__}")
                logging.info(f"Function: {func_prototype}")
                logging.info(f"Time: {t_elapsed:.4f} sec")
                logging.info(f"Peak memory: {peak_mem_usage:.4f} MiB")
                logging.info(f"CPU power: {emission_summary.cpu_power:.4f} W")
                logging.info(f"GPU power: {emission_summary.gpu_power:.4f} W")
                logging.info(f"RAM power: {emission_summary.ram_power:.4f} W")
                logging.info("======End of Profiling=====\n")

                if return_prf:
                    return result, ProfileResult(t_elapsed, peak_mem_usage, emission_summary)
                else:
                    return result, None

            return inner

        return profile

    @profile_factory(return_prf=True)
    def run(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Optional[Any], Tuple[float, float]]:
        """Run profiling and return performance report."""
        assert callable(func), "Function to profile must be callable."

        return func(*args, **kwargs)
