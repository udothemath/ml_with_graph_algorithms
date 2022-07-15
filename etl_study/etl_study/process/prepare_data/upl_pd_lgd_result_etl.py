from src.common.prepare_data.table_etl_base import TableETLBase
from src.config import TABLE_SUBFIX


class UPL_PD_LGD_RESULTETL(TableETLBase):
    @property
    def source_table_name(self):
        return 'upl_pd_lgd_result' + TABLE_SUBFIX

    @property
    def tmp_column_types(self):
        return {
            'pd_value': 'numeric',
            'pd_grade': 'int',
            'lgd': 'numeric'
        }

    @property
    def tmp_column_defaults(self):
        return {
            'pd_value': '-1',
            'pd_grade': '-1',
            'lgd': '-1'
        }

    def check_source(self):
        sqlstring = f'''
            SELECT
            count(*)
        FROM cc_ln_pre_approval.{self.source_table_name}
        '''
        row_count = self.select_table(
            'feature', sqlstring)['count'].values[0]
        assert row_count > 0

    @property
    def etlSQL(self):
        sqlstring = f'''
        SELECT
            cust_no,
            pd_value,
            pd_grade,
            lgd
        FROM cc_ln_pre_approval.{self.source_table_name}
        '''
        return sqlstring

    @property
    def source_db(self):
        """
        來源表的DB: feature or rawdata
        """
        return 'feature'

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
            pd_value,
            pd_grade,
            lgd
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert table.lgd.min() == -1 or (
            table.lgd.max() <= 1 and table.lgd.min() >= 0
        )
        assert table.pd_value.min() == -1 or (
            table.pd_value.max() <= 1 and table.pd_value.min() >= 0
        )
        assert table.pd_grade.min() == -1 or (
            table.pd_grade.max() <= 15 and table.pd_grade.min() >= 1
        )
