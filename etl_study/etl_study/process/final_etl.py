"""
NOTE:

- FinalBase不定義source_table_name，由卡處於__init__中自行定義

- 由卡處在FinalETL __init__中自行定義各個產品的產出表名稱為變數，讓這些產出表名稱可以在get_process_SQL中整併進SQL。

"""
from src.common.sql_base import SQLBase
from src.config import TABLE_SUBFIX


class FinalETL(SQLBase):
    """
    最終表的產製SQL在此撰寫
    """

    def __init__(self, source_table_name=None,
                 target_table_name='esun_cust_loan_preapproval_data'
                 ):
        super(FinalETL, self).__init__(
            target_table_name
        )
        self.source_table_name = source_table_name

    def get_target_create_SQL(self, schema_name, table_name):
        """
        Args:
            - schema_name: 'cc_ln_pre_approval'
            - table_name:  最終結果表的名稱
        Return:
            - (str) 創建最終結果表的SQL
        """
        return f'''
            CREATE TABLE
            IF NOT EXISTS {schema_name}.{table_name}(
            cust_no CHAR(24),
            "group" CHAR(2),
            product CHAR(2),
            apdlv NUMERIC(10,6),
            lgd NUMERIC(10,3),
            base_int NUMERIC(10,6),
            profit_int NUMERIC(10,6),
            pre_net_income BIGINT,
            max_limit INT,
            interest_rate_1 NUMERIC(10,6),
            period_1 INT,
            interest_rate_2 NUMERIC(10,6),
            period_2 INT,
            fee_amount INT,
            all_rate NUMERIC(10,6),
            list_name NCHAR(20),
            data_dt DATE,
            etl_ts TIMESTAMP,
            Primary key(cust_no, product)
            );'''

    def get_process_SQL(self, schema_name, target_table_name):
        """
        Args:
            - schema_name: 'cc_ln_pre_approval'
            - target_table_name: 輸出表的名稱 (e.g., 'esun_cust_loan_preapproval_data')
        Return:
            - (str) Select並且Insert進db的SQL
        """
        return f'''
        WITH
            base_cust_result_one AS (
                SELECT
                    *,
                    DATE_TRUNC(\'MONTH\', NOW() + INTERVAL \'1 month\')::DATE AS data_dt,
                    NOW()::timestamp AS etl_ts
                FROM {schema_name}.cust_tmp_result_one{TABLE_SUBFIX}
            ),
            base_cust_result_rl AS (
                SELECT
                    *,
                    DATE_TRUNC(\'MONTH\', NOW() + INTERVAL \'1 month\')::DATE AS data_dt,
                    NOW()::timestamp AS etl_ts
                FROM {schema_name}.cust_tmp_result_revolving{TABLE_SUBFIX}

            ),
            onetime_loan_result AS (
                SELECT
                    cust_no,
                    "group",
                    \'01\' AS product,
                    apdlv,
                    lgd,
                    base_int,
                    profit_int,
                    pre_net_income,
                    max_limit,
                    interest_rate_1,
                    period_1,
                    interest_rate_2,
                    period_2,
                    fee_amount,
                    all_rate,
                    CONCAT(TO_CHAR(data_dt, \'YYYYMM\'), \'一次撥付預審\') AS list_name,
                    data_dt,
                    etl_ts
                FROM base_cust_result_one

            ),
            cycle_loan_result AS (
                SELECT
                    cust_no,
                    "group",
                    \'02\' AS product,
                    apdlv,
                    lgd,
                    base_int,
                    profit_int,
                    pre_net_income,
                    max_limit,
                    interest_rate_1,
                    period_1,
                    interest_rate_2,
                    period_2,
                    fee_amount,
                    all_rate,
                    CONCAT(TO_CHAR(data_dt, \'YYYYMM\'), \'循環動用預審\') AS list_name,
                    data_dt,
                    etl_ts
                FROM base_cust_result_rl
                WHERE product IN (\'02\')
                AND pre_net_income > 24000
                AND max_limit >= 30000
                AND apdlv <> -1
                AND lgd <> -1
            ),
            final_result AS (
                SELECT *
                FROM onetime_loan_result
                UNION
                SELECT *
                FROM cycle_loan_result
            )
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

        sqlstring = f'''
            SELECT product, COUNT(DISTINCT cust_no) AS cust_count
            FROM {schema_name}.{table_name}
            GROUP BY product
            '''
        table = self.select_table('feature', sqlstring)
        assert all(table['cust_count'] >= 0)
        row_count = table['cust_count'].sum()
        assert self._check_product(schema_name, table_name)
        return True, row_count

    def _check_product(self, schema_name, table_name):
        sqlstring = f'''
          SELECT *
            FROM (
                    SELECT cust_no AS cust_no2,
                           "group" AS group2,
                           product AS product2,
                           apdlv AS apdlv2,
                           lgd AS lgd2,
                           base_int AS base_int2,
                           pre_net_income AS pre_net_income2
                    FROM {schema_name}.{table_name}
                    WHERE product = '02'
                 ) A
            LEFT JOIN (
                        SELECT *
                        FROM {schema_name}.{table_name}
                        WHERE product = '01'
                      ) B ON A.cust_no2 = B.cust_no
            '''
        table = self.select_table('feature', sqlstring)
        assert all(table['group2'] == table['group'])
        assert all(table['apdlv2'] == table['apdlv'])
        assert all(table['lgd2'] == table['lgd'])
        assert all(table['base_int2'] == table['base_int'])
        assert all(table['pre_net_income2'] == table['pre_net_income'])
        return True
