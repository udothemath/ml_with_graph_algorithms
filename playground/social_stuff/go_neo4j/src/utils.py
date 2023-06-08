import pandas as pd
from functools import wraps
import inspect
import time
import traceback


def read_csv_as_chunk(fname, sample_size, chunk_size=1000):
    reader = pd.read_csv(fname, header=0, nrows=sample_size,
                         iterator=True, low_memory=False)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Finish reading csv. Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac


def save_df_to_csv(input_df: pd.DataFrame(), to_filename: str,
                   to_path: str) -> None:
    file_with_path = f"{to_path}/{to_filename}"
    try:
        input_df.to_csv(f"{file_with_path}", index=False)
        print(f"U have successfully save file {file_with_path}")
    except Exception as e:
        print("Fail to save csv file")
        raise e


def wrap_log(orig_func):
    """decorator for saving input, output & elapased time of a function"""

    @wraps(orig_func)
    def wrapper(self, *args, **kwargs):
        """
        function warped by warp_log
        """
        filename_with_path = inspect.getfile(orig_func)
        time_start = time.time()
        try:
            out = orig_func(self, *args, **kwargs)
            success = True
        except BaseException as e:
            exception = e
            error_traceback = str(traceback.format_exc())
            success = False
        time_elapsed = time.time() - time_start
        if success:
            logs = {
                "success": success,
                "in": {"args": str(args), "kwargs": str(kwargs)},
                "out": str(out),
                "time": time_elapsed,
                "func": orig_func.__name__,
                "module": filename_with_path,
            }
        else:
            logs = {
                "success": success,
                "in": {"args": str(args), "kwargs": str(kwargs)},
                "exception": str(exception),
                "traceback": error_traceback,
                "time": time_elapsed,
                "func": orig_func.__name__,
                "module": filename_with_path,
            }
        self.logger.info({"saved_log": logs})
        if success:
            return out
        else:
            raise exception

    return wrapper
