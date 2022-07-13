import copy
import abc
from abc import ABC


class LogicBase(ABC):
    """
    要套用的邏輯運算寫在此
    """
    @property
    @abc.abstractmethod
    def input_column_names(self):
        """
        定義函數的輸入欄位

        需與輸入表之db欄位有相同名稱

        Returns:
            - col_names: 一個欄位名稱string組成的list
        """
        return []

    @property
    @abc.abstractmethod
    def output_column_names(self):
        """
        定義函數的輸出欄位

        需與輸出之表的db欄位有相同的名稱

        Returns:
            - col_names: 一個欄位名稱string組成的list
        """
        return []

    @abc.abstractmethod
    def run_all(self, *args, **kwargs):
        """
        要被套用的函數

        可以有任意的輸入和任意的輸出

        輸出以list或tuple的方式輸出

        """
        pass

    def post_processing(self, result_table):
        """
        對資料表進行最終的處理 (避免存入db時發生型態問題)

        Args:
            - result_table: 經過邏輯函數轉換過後的結果pd.DataFrame
        Returns:
            - result_table: 經過型態轉換過後的pd.DataFrame
        Example:
            result_table.col1 = result_table.col1.map(int)
            return result_table

        Note:
            若沒有override此function，預設不對result_table進行處理
        """
        return result_table

    # @log_sparse
    def process_df(self, input_table):
        """
        轉換選出來的片段pd.DataFrame

        Args:
            - input_table: 選出來的片段pd.DataFrame
        Returns:
            - result_table: 轉換後的pd.DataFrame
        """
        result_table = input_table.apply(
            self.__apply_func, axis=1)
        result_table = self.post_processing(result_table)
        return result_table

    def __apply_func(self, input_row):
        """
        針對pd.DataFrame的一行進行轉換

        Args:
            - row: 要被轉換的行
        Returns:
            - row: 轉換後的行
        """
        # 計算結果
        try:
            row = copy.deepcopy(input_row)
            results = list(
                self.run_all(
                    *[row[col_name] for col_name in self.input_column_names]
                )
            )
            # 將結果整理進row中。
            for rm_col in self.input_column_names:
                del row[rm_col]
            for col, result in zip(self.output_column_names, results):
                row[col] = result
            return row
        except BaseException as e:
            raise ValueError(
                f'error caused by run_all on row {input_row}\n traceback: {e}'
            )
