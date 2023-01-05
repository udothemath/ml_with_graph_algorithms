# %%
import time
from functools import wraps


def elapsed_time(func) -> None:
    ''' print elapsed time '''
    @wraps(func)
    def out(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        print(f"Elapsed time of {func.__name__}: {elapsed_time:.4f}")
        return result
    return out
