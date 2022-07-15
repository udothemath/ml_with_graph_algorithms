"""
分流平行化模組的其中一塊
"""
import abc
import logging
from src.common.apply.apply_base import ApplyBase
from src.config import SPLIT_DEGREE

VERBOSE = True


class ForkingBase(ApplyBase):
    """
    平行處理模組
    """

    def __init__(self, input_db, input_table_name, target_table_name,
                 input_schema_name, target_schema_name='cc_ln_pre_approval',
                 set_index=None, verbose=False, group_id=-1,
                 split_degree=SPLIT_DEGREE):
        assert isinstance(group_id, int)
        assert isinstance(split_degree, int) and split_degree >= 0
        assert set_index is not None
        super(ForkingBase, self).__init__(
            input_db=input_db,
            input_table_name=input_table_name,
            target_table_name=target_table_name if (
                group_id == -1
            ) else target_table_name + f'_split_{group_id}',
            input_schema_name=input_schema_name,
            target_schema_name=target_schema_name,
            set_index=set_index,
            verbose=verbose,
            parallel_cnt=1
        )
        self.__group_id = group_id
        self.__split_degree = split_degree
        self.__set_index = set_index
        self.target_schema_name = target_schema_name

    @property
    def split_condition(self):
        """
        分流機制中，篩選流向用的條件statement
        透過WHERE {self.split_condition} 來鑲嵌於select SQL之中。
        """
        if self.__split_degree > 0:
            hex_str = "{0:0{1}x}".format(self.__group_id, self.__split_degree)
            if self.split_condition_extension == "":
                condition_str = f"""
                SUBSTR(MD5({self.__set_index}), 1, {self.__split_degree}) = '{hex_str}'
                """
            else:
                condition_str = f"""
                SUBSTR(MD5({self.__set_index}), 1, {self.__split_degree}) = '{hex_str}' AND ({self.split_condition_extension})
                """
        else:
            condition_str = self.split_condition_extension
        return condition_str

    @property
    def split_condition_extension(self):
        """
        若在SELECT的時候要加入額外條件請覆寫此函式
        """
        return ""

    def get_partial_select_SQL(self, schema_name, table_name, offset, size):
        """
        產生batch-by-batch Select時，所用的SQL

        Args:
            - schema_name: input表schema名稱
            - table_name: input表名稱
            - offset: Select的起始行數
            - size: batch的大小
        Returns:
            - select_sql: 選表的SQL

        Example:
            select_sql = f'''
                SELECT
                    cust_id AS cust_no,
                    input_data
                FROM {schema_name}.{table_name}
                LIMIT {size}
                OFFSET {offset}
                '''
            return select_sql
        """
        if VERBOSE:
            logging.info(
                f'[get_partial_select_SQL] schema_name: {schema_name}')
            logging.info(f'[get_partial_select_SQL] table_name: {table_name}')
            logging.info(f'[get_partial_select_SQL] offset: {offset}')
            logging.info(f'[get_partial_select_SQL] size: {size}')
            logging.info(
                f'[get_partial_select_SQL] index_col: {self.__set_index}')
            logging.info(
                f'[get_partial_select_SQL] condition_SQL: {self.split_condition}')
        select_sql = self.get_forking_partial_select_SQL(
            schema_name, table_name, offset, size, index_col=self.__set_index,
            condition_SQL=self.split_condition)
        if VERBOSE:
            logging.info(
                f'[get_partial_select_SQL] Resulting select_sql: {select_sql}')
        return select_sql

    @abc.abstractmethod
    def get_forking_partial_select_SQL(
            self, schema_name, table_name, offset, size, index_col, condition_SQL):
        """
        產生batch-by-batch Select時，所用的SQL

        Args:
            - schema_name: input表schema名稱
            - table_name: input表名稱
            - offset: Select 的起始行數
            - size: batch 的大小
            - index_col: index_col的名稱 (e.g. 'cust_no')
            - condition_SQL: 一個cust_no=XXX的SQL string，是分流用的篩選條件
        Returns:
            - select_sql: 選表的SQL

        Example:
            select_sql = f'''
                SELECT
                    {index_col},
                    /*index欄位，如cust_no*/
                    input_data
                    /*非index的欄位*/
                FROM {schema_name}.{table_name}
                WHERE {condition_SQL}
                /*分流用的篩選條件*/
                ORDER BY pg_column_size(input_data) DESC, {index_col}
                /*
                    NOTE:
                    1. 將json檔由大排到小以避免記憶體使用越來越多
                    2. 一定要用index_col來排序，確保每次撈取的行不重複
                */
                LIMIT {size}
                OFFSET {offset}
                '''
            return select_sql
        """
        pass

    def check_input_size(self, input_db, schema_name, table_name):
        """
        檢查input表的大小

        Args:
            - input_db: 輸入表db (feature | rawdata)
            - schema_name: 輸入表schema (cc_ln_pre_approval)
            - table_name: 輸入表名稱 (__init__定義的input_table_name)
        Returns:
            - row_count: 行數
        """
        sql = f"""
        SELECT
            count(*)
        FROM {schema_name}.{table_name}
        WHERE {self.split_condition}
        """
        logging.info(f'count SQL: {sql}')
        result_table = self.select_table(input_db, sql)
        row_count = result_table['count'][0]
        logging.info(f'CHECK INPUT TABLE SIZE: {row_count}')
        return row_count
