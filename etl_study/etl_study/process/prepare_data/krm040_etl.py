from src.common.prepare_data.table_etl_base import TableETLBase


class KRM040ETL(TableETLBase):
    @property
    def source_table_name(self):
        return "KRM040"

    @property
    def tmp_column_types(self):
        return {
            'krm040_ind': 'char(1)'

        }

    @property
    def tmp_column_defaults(self):
        return {
            'krm040_ind': "'N'"

        }

    def check_source(self):
        sqlstring = f'''
            SELECT
            count(*)
        FROM if_jcic_superset_online.{self.source_table_name}
        '''
        row_count = self.select_table('feature', sqlstring)['count'].values[0]
        assert row_count > 0

    @property
    def etlSQL(self):
        """
        【krm040_ind】
        1.篩選【查詢日 querydate】為近6個月(含6個月)
        2.依上述篩選後給予註記"Y"，其餘為"N"

        """

        sqlstring = f'''
        SELECT id AS cust_no, 'Y' AS krm040_ind
        FROM if_jcic_superset_online.{self.source_table_name}
        WHERE querydate >= DATE_TRUNC('month', ADD_MONTHS(CURRENT_DATE(),-6))
        GROUP BY id

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
            krm040_ind
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)

        assert all(table.krm040_ind.isin(['Y', 'N']))
