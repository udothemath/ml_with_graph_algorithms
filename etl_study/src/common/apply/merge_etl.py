"""
合併apply產出的所有中繼表
"""
import logging
from src.common.sql_base import SQLBase
from src.config import SPLIT_DEGREE, MAX_BATCH, BATCH_SIZE


class MergeETL(SQLBase):
    """
    最終表的產製SQL在此撰寫
    """

    def __init__(self, source_table_name,
                 target_table_name, target_create_SQL, split_condition_extension=""):
        super(MergeETL, self).__init__(
            target_table_name
        )
        self.__target_create_SQL = target_create_SQL
        self.__source_table_name = source_table_name
        self.__split_condition_extension = split_condition_extension

    def get_target_create_SQL(self, schema_name, table_name):
        """
        Args:
            - schema_name: 'cc_ln_pre_approval'
            - table_name:  最終結果表的名稱
        Return:
            - (str) 創建最終結果表的SQL
        """
        logging.info('[get_target_create_SQL] schema_name: {schema_name}')
        logging.info('[get_target_create_SQL] table_name: {table_name}')
        return self.__target_create_SQL

    def get_process_SQL(self, schema_name, target_table_name):
        """
        Args:
            - schema_name: 'cc_ln_pre_approval'
            - target_table_name: 輸出表的名稱 (e.g., 'esun_cust_loan_preapproval_data')
        Return:
            - (str) Select並且Insert進db的SQL
        """
        union_sql = f"""
            (SELECT
            *
            FROM {schema_name}.{target_table_name}_split_0
            )
            """
        for i in range(1, 16**SPLIT_DEGREE):
            union_sql = union_sql + "UNION ALL" + f"""
            (SELECT
            *
            FROM {schema_name}.{target_table_name}_split_{i}
            )
            """
        return f'''
        WITH final_result AS ({union_sql})
        INSERT INTO {schema_name}.{target_table_name}
        SELECT * FROM final_result
        '''

    def check_target_table(self, schema_name, table_name):
        """
        檢查db中結果表的正確性

        Args:
            - schema_name: 結果表schema名稱
            - table_name: 結果表名稱
        Return:
            - result (bool): 成功與否
            - row_count (int): 結果表行數
        """

        result_cnt_sqlstring = f'''
        select count(cust_no)
        from {schema_name}.{table_name}
        '''
        logging.info('CHECK RESULT:' + result_cnt_sqlstring)
        result_count = self.select_table(
            'feature', result_cnt_sqlstring)['count'].values[0]
        logging.info(f'RESULT ROW COUNT: {result_count}')
        if self.__split_condition_extension == "":
            source_cnt_sqlstring = f'''
            select count(cust_no)
            from {schema_name}.{self.__source_table_name}
            '''
        else:
            source_cnt_sqlstring = f'''
            select count(cust_no)
            from {schema_name}.{self.__source_table_name}
            WHERE {self.__split_condition_extension}
            '''
        logging.info('CHECK SOURCE:' + source_cnt_sqlstring)
        source_count = self.select_table(
            'feature', source_cnt_sqlstring)['count'].values[0]
        logging.info(f'SOURCE ROW COUNT: {source_count}')
        if MAX_BATCH is None:
            assert result_count == source_count
        else:
            assert result_count == MAX_BATCH * BATCH_SIZE * (16**SPLIT_DEGREE)
        return True, result_count
