"""Logic base class with cuDF acceleration."""
import abc
from typing import Any, Callable, Dict, List, Tuple, Union

import astunparse
import cudf

from common.apply.logic_base import LogicBase
from common.apply.udf_parser import UDFParser


class CUDFLogicBase(LogicBase):
    """Logic base class with cuDF acceleration.

    Parameters:
        udf_logic_cls_path: file path of user-defined logic class

    Examples:
        When users write a user-defined logic class inherited from this
        base class, the initializer should be defined as follows:

            class UDFLogic(CUDFLogicBase):
                def __init__(self):
                    super(UDFLogic, self).__init__(__file__)

        where <UDF> can be replaced with project-specific name.
    """

    def __init__(self, udf_logic_cls_path: str):
        self.udf_logic_cls_path = udf_logic_cls_path

    @property
    @abc.abstractmethod
    def input_column_names(self) -> List[str]:
        """A list of input columns.

        Input columns must match those of input table in database.

        Examples:
            return ["input_column1", "input_column2"]
        """
        return []

    @property
    @abc.abstractmethod
    def output_column_names(self) -> List[str]:
        """A list of output columns.

        Output columns must match those of output table in database.

        Examples:
            return ["output_column1", "output_column2"]
        """
        return []

    @property
    @abc.abstractmethod
    def output_column_dtypes(self) -> Dict[str, Any]:
        """A dictionary of output column names and their dtype.

        Examples:
            return {
                "output_column1": np.int16,
                "output_column2": np.float32,
            }
        """
        return {}

    @abc.abstractmethod
    def run_all(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Union[List[Any], Tuple[Any]]:
        """User-defined function with the specified input and output
        columns."""
        pass

    def process_df(self, input_table: cudf.DataFrame) -> cudf.DataFrame:
        """Apply user-defined logic to input table.

        Parameters:
            input_table: input table to process

        Return:
            result_table: processed table
        """
        result_table = input_table.apply_rows(
            self.run_all_boost,
            incols=self.input_column_names,
            outcols=self.output_column_dtypes,
            kwargs={},
        )[self.output_column_names]

        return result_table

    @property
    def run_all_boost(self) -> Callable:
        """Numba kernel corresponding to user-defined `run_all`."""
        udf_parser = UDFParser(self.udf_logic_cls_path)
        run_all_numba_ast = udf_parser.parse_run_all()
        run_all_numba_str = astunparse.unparse(run_all_numba_ast)
        exec(run_all_numba_str, globals())

        return run_all_numba
