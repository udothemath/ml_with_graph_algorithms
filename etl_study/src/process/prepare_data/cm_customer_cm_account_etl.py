"""
抓出全行個人戶 (biz_line_code='P')，且沒有銷戶 (acct_status_code='A')
的顧客cust_no 作為母體。

SQL見: select_n_transform
"""
from src.common.prepare_data.table_etl_base import TableETLBase


class CM_CUSTOMER_CM_ACCOUNTETL(TableETLBase):
    @property
    def source_table_name(self):
        return "CM_CUSTOMER_CM_ACCOUNT"

    @property
    def tmp_column_types(self):
        # 母體表只有cust_no欄位，沒有其他，因此給空的{}就好
        return {}

    @property
    def tmp_column_defaults(self):
        # 母體表只有cust_no欄位，沒有其他，因此給空的{}就好
        return {}

    def check_source(self):
        sqlstring = '''
            SELECT
            count(*)
        FROM mlaas_rawdata.CM_CUSTOMER
        '''
        row_count_cm_customer = self.select_table(
            'rawdata', sqlstring)['count'].values[0]
        assert row_count_cm_customer > 0

        sqlstring = '''
            SELECT
            count(*)
        FROM mlaas_rawdata.CM_ACCOUNT
        '''
        row_count_cm_account = self.select_table(
            'rawdata', sqlstring)['count'].values[0]
        assert row_count_cm_account > 0

    @property
    def etlSQL(self):

        sqlstring = '''
            /*母體邏輯*/
            SELECT
                t1.cust_no AS cust_no
            FROM (
                SELECT DISTINCT cust_no
                FROM "mlaas_rawdata"."cm_account"
                WHERE acct_status_code = 'A'
                /*未銷戶條件*/
            ) AS t1
            INNER JOIN (
                SELECT
                    cust_no,
                    duplicate_count
                FROM (
                    SELECT
                        cust_no,
                        count(*) AS duplicate_count
                    FROM mlaas_rawdata.cm_customer
                    WHERE biz_line_code = 'P'
                    /*個人戶條件*/
                    GROUP BY cust_no
                )
                WHERE duplicate_count = 1
                /*排除重複出現顧客*/
            ) AS t2
            ON t1.cust_no = t2.cust_no
        '''
        return sqlstring

    @property
    def source_db(self):
        """
        來源表的DB: feature or rawdata
        """
        return 'rawdata'

    def extract_pandas_df(self):
        """
        開發時，可執行此函數，以使用`etlSQL`從`source_db`所指定的DB，
        將資料表選出為pandas.DataFrame。
        """
        return self.select_table(self.source_db, self.etlSQL)

    def check_target_columns(self, target_table_name):
        sqlstring = f'''
            SELECT
            count(*)
        FROM {self.schema_name}.{target_table_name}
        '''
        row_count = self.select_table(
            'feature', sqlstring)['count'].values[0]
        assert row_count > 0
