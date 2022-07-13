from src.common.prepare_data.table_etl_base import TableETLBase


class CMCH0001ETL(TableETLBase):
    @property
    def source_table_name(self):
        return "CMCH0001"

    @property
    def tmp_column_types(self):
        return {
            'gov_employee': 'char(1)',
            'military_police_firefighters': 'char(1)',
            'cc_monin': 'bigint'
        }

    @property
    def tmp_column_defaults(self):
        return {
            'gov_employee': "'N'",
            'military_police_firefighters': "'N'",
            'cc_monin': "-1"
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
        【gov_employee】
        1.因是月檔關係，故先計算每位卡友最新的【資料日期 dtadt】，篩選
        2.依上述篩選各卡友最新【資料日期 dtadt】的資料月份
        3.篩選【行業別 trdtp】為"11"，則給予註記"Y"，其餘為"N"

        【military_police_firefighters】
        1.因是月檔關係，故先計算每位卡友最新的【資料日期 dtadt】，篩選
        2.依上述篩選各卡友最新【資料日期 dtadt】的資料月份
        3.篩選【行業別 trdtp】為"10"，則給予註記"Y"，其餘為"N"

        【cc_monin】
        1.因是月檔關係，故先計算每位卡友最新的【資料日期 dtadt】，篩選
        2.依上述篩選各卡友最新【資料日期 dtadt】的資料月份
        3.取得【monin】欄位，並命名為cc_monin

        """
        sqlstring = f'''
            SELECT chid cust_no,
                   CASE WHEN trdtp ='11' THEN 'Y' ELSE 'N' END gov_employee,
                   CASE WHEN trdtp ='10' THEN 'Y'
                        ELSE 'N' END military_police_firefighters,
                   CAST(monin AS BIGINT) AS cc_monin
            FROM mlaas_rawdata.{self.source_table_name}
            WHERE dtadt in (SELECT MAX(dtadt) dtadt
                            FROM mlaas_rawdata.{self.source_table_name}
                            )
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
            gov_employee,
            military_police_firefighters,
            cc_monin
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(table.gov_employee.isin(['Y', 'N']))
        assert all(table.military_police_firefighters.isin(['Y', 'N']))
        assert all(table.cc_monin >= -1)
