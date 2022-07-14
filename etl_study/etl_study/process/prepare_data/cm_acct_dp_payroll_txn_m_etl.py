from src.common.prepare_data.table_etl_base import TableETLBase


class CM_ACCT_DP_PAYROLL_TXN_METL(TableETLBase):
    @property
    def source_table_name(self):
        return "CM_ACCT_DP_PAYROLL_TXN_M"

    @property
    def tmp_column_types(self):
        return {
            'salary_monin': 'bigint',
            'salary_ind': 'char(1)'
        }

    @property
    def tmp_column_defaults(self):
        return {
            'salary_monin': "-1",
            'salary_ind': "'N'"
        }

    def check_source(self):
        sqlstring = f'''
            SELECT
            count(*)
        FROM mlaas_rawdata.{self.source_table_name}
        '''
        row_count = self.select_table('rawdata', sqlstring)['count'].values[0]
        assert row_count > 0

    @property
    def etlSQL(self):
        """
        【salary_monin】
        1.篩選【資料年月 data_ym】為近3個月(含3個月)
        2.計算顧客平均【撥薪金額 PAYROLL_AMT】

        【salary_ind】
        1.篩選【資料年月_data_ym】為近3個月(含3個月)
        2.依上述篩選後給予註記"Y"，其餘為"N"

        """

        sqlstring = f'''
            SELECT cust_no, CAST(salary_monin AS BIGINT) AS salary_monin,
                   'Y' AS salary_ind
            FROM (
                      SELECT cust_no, AVG(payroll_amt) AS salary_monin
                      FROM mlaas_rawdata.{self.source_table_name}
                      WHERE data_ym >= DATE_TRUNC('month',ADD_MONTHS(NOW(),-3))
                      GROUP BY cust_no
                  ) SALARY
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
            cust_no,
            salary_monin,
            salary_ind
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(table.salary_monin >= -1)
        assert all(table.salary_ind.isin(['Y', 'N']))
