"""UDF logic in cudf mode."""
import numpy as np

from common.apply.cudf_logic_base import CUDFLogicBase

class CUDFLogic(CUDFLogicBase):
    """UDF logic in cuDF mode."""
    
    def __init__(self):
        super(CUDFLogic, self).__init__(__file__)

    @property
    def input_column_names(self):
        return [
            "leadi",
            "leadii",
            "leadiii",
            "leadavr",
        ]

    @property
    def output_column_names(self):
        return ["i_ii", "i_iii", "i_avr"]

    @property
    def output_column_dtypes(self):
        return {"i_ii": np.int16, "i_iii": np.int16, "i_avr": np.int16}

    def sub(self, x, y):
        diff = x - y

        return diff

    def run_all(self, leadi, leadii, leadiii, leadavr):
        i_ii = self.sub(leadi, leadii)
        i_iii = self.sub(leadi, leadiii)
        i_avr = self.sub(leadi, leadavr)

        return i_ii, i_iii, i_avr
