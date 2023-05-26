"""
utils
"""
import inspect
import time
import traceback
from functools import wraps

VERBOSE = False


def wrap_log(orig_func):
    '''decorator for saving input, output & elapased time of a function'''
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
                'success': success,
                'in': {
                    'args': str(args),
                    'kwargs': str(kwargs)
                },
                'out': str(out),
                'time': time_elapsed,
                'func': orig_func.__name__,
                'module': filename_with_path
            }
            self.logger.info({'saved_log': logs})
        else:
            logs = {
                'success': success,
                'in': {
                    'args': str(args),
                    'kwargs': str(kwargs)
                },
                'exception': str(exception),
                'traceback': error_traceback,
                'time': time_elapsed,
                'func': orig_func.__name__,
                'module': filename_with_path
            }
            self.logger.error({'saved_log': logs})
        if success:
            return out
        else:
            raise exception
    return wrapper
