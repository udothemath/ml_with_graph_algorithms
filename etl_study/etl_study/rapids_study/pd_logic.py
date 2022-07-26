"""UDF logic in pandas mode."""
from common.apply.logic_base import LogicBase

class PDLogic(LogicBase):
    """UDF logic in pandas mode."""
    
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

    def sub(self, x, y):
        """Subtract y from x.

        Parameters:
            x: minuend
            y: subtrahend

        Return:
            diff: difference
        """
        diff = x - y

        return diff

    def run_all(self, leadi, leadii, leadiii, leadavr):
        """Apply user-defined function (logic) to a single row."""
        i_ii = self.sub(leadi, leadii)
        i_iii = self.sub(leadi, leadiii)
        i_avr = self.sub(leadi, leadavr)

        return [i_ii, i_iii, i_avr]