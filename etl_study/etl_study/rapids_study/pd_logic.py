"""UDF logic in pandas mode."""
import math
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from common.apply.logic_base import LogicBase

warnings.simplefilter("ignore")


class PDLogic(LogicBase):
    """UDF logic in pandas mode."""

    @property
    def input_column_names(self) -> List[str]:
        return []

    @property
    def output_column_names(self) -> List[str]:
        return []

    def cluster_with_str_col(self, s_2: str) -> int:
        """Cluster samples based on determinant with str dtype.

        Parameters:
            s_2: date

        Return:
            before_19_04_20: date before 2019-04-20 or not
        """
        if s_2 < "2019-04-20":
            before_19_04_20 = 1
        else:
            before_19_04_20 = 0

        return before_19_04_20

    def cluster_with_collection(self, b_30: np.int64) -> Tuple[int, int]:
        """Cluster samples with python built-in collections acting as
        determinant.

        Parameters:
            b_30: categorical feature

        Return:
            clust_list: clustering result based on list determinant
            clust_tuple: clustering result based on tuple determinant
        """
        clust_list, clust_tuple = None, None

        if b_30 in [-1, 0]:
            clust_list = 0
        elif b_30 in [1, 2]:
            clust_list = 1

        tuple1 = (-1, 1)
        if b_30 in tuple1:
            clust_tuple = 0
        elif b_30 in (0, 2):
            clust_tuple = 1

        return clust_list, clust_tuple

    def round_and_max(self, d_45: np.float64, d_52: np.float64) -> np.float64:
        """Perform continuous value rounding and return the greatest of
        the two.

        Parameters:
            d_45: numeric feature
            d_52: numeric feature

        Return:
            max_val: the greatest value among input features
        """
        d_45 = round(d_45, 4)
        d_52 = round(d_52, 4)
        max_val = max(d_45, d_52)

        return max_val

    def cast_dtypes(self, b_30: np.int64) -> Tuple[float, int]:
        """Cast data types.

        Parameters:
            b_30: categorical feature

        Return:
            int2float: casted float value
            float2int: casted int value
        """
        int2float = float(b_30)
        float2int = int(int2float)

        return int2float, float2int

    def apply_operators(self, b_30: np.int64, d_52: np.float64) -> Tuple[bool, float]:
        """Apply common operators (e.g., mod, shifting).

        Parameters:
            b_30: categorical feature
            d_52: numeric feature

        Return:
            output1: first output
            output2: second output
        """
        output1 = False
        output2 = 0

        if d_52 < 1.5:
            output1 = False
        else:
            output1 = True

        if b_30 == 0:
            output2 = int(b_30) << 2
        elif b_30 == 1:
            output2 = int(b_30) ^ 100
        elif b_30 == 2:
            output2 = d_52 % 1.25 + int(b_30) ** 2
        else:
            output2 = int(b_30) & 100 | 30

        return output1, output2

    def apply_complex_fp(self, d_45: np.float64, d_52: np.float64) -> np.float64:
        """Apply complex floating-point arithmetic.

        Parameters:
            d_45: numeric feature
            d_52: numeric feature

        Return:
            output: computation result
        """
        output = 0.0

        output = d_45 * 3.141598 / (d_52 / 2.71828) + (d_45 + 13.14) - (d_52 - 8.88)

        return output

    def handle_missing(self, r_9: np.int64, d_142: np.float64) -> Tuple[int, bool]:
        """Handle missing value.

        Parameters:
            r_9_imp: imputed value
            d_142_isnull: whether d_142 is missing
        """
        r_9_imp = 0
        d_142_isnull = False

        if r_9 is None:
            r_9_imp = 0
        elif r_9 is pd.NA:
            r_9_imp = 1
        elif r_9 is pd.NaT:
            r_9_imp = 2
        elif math.isnan(r_9):
            r_9_imp = 3

        if d_142 is None:
            d_142_isnull = True
        elif d_142 is pd.NA:
            d_142_isnull = True
        elif d_142 is pd.NaT:
            d_142_isnull = True
        elif math.isnan(d_142):
            d_142_isnull = True

        return r_9_imp, d_142_isnull

    def run_all(self) -> List[Any]:
        #         before_19_04_20 = self.cluster_with_str_col(s_2)
        #         clust_list, clust_tuple = self.cluster_with_collection(b_30)
        #         max_val = self.round_and_max(d_45, d_52)
        #         float2int, int2float = self.cast_dtypes(b_30)
        #         output1, output2 = self.apply_operators(b_30, d_52)
        #         output = self.apply_complex_fp(d_45, d_52)
        #         r_9_imp, d_142_isnull = self.handle_missing(r_9, d_142)

        return None
