from src.common.prepare_data.table_etl_base import TableETLBase


class CDCA0001ETL(TableETLBase):
    @property
    def source_table_name(self):
        return "CDCA0001"

    @property
    def tmp_column_types(self):
        return {
            'with_cc': 'char(1)',
            'world_card': 'char(1)',
            'travel_card': 'char(1)',
            'five_profession_card': 'char(1)'
        }

    @property
    def tmp_column_defaults(self):
        return {
            'with_cc': "'N'",
            'world_card': "'N'",
            'travel_card': "'N'",
            'five_profession_card': "'N'"
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
        【with_cc】
        1.篩選【卡片狀態 curcd】為"0"(流通),"9"(毀損)及【正附卡類別 catp2】為0
        2.依上述篩選後給予註記"Y"，其餘為"N"

        【travel_card】
        1.篩選【卡片狀態 curcd】為"0"(流通),"9"(毀損)及【正附卡類別 catp2】為0
        2.加總【認同團體 agno】為"0032"的張數
        3.若張數大於0，則給予註記"Y"，其餘為"N"

        【five_profession_card】
        1.篩選【卡片狀態 curcd】為"0"(流通),"9"(毀損)及【正附卡類別 catp2】為0
        2.加總【認同團體 agno】為"0114","0115","0116","0117","0118"的張數
        3.若張數大於0，則給予註記"Y"，其餘為"N"

        【world_card】
        1.篩選【卡片狀態 curcd】為"0"(流通),"9"(毀損)及【正附卡類別 catp2】為0
        2.加總【認同團體 agno】為"A008","A011","A012"的張數
        3.若張數大於0，則給予註記"Y"，其餘為"N"
        """

        sqlstring = f'''
        SELECT aa.bacno AS cust_no,
               'Y' AS with_cc,
               CASE WHEN aa.world_card > 0 THEN 'Y' ELSE 'N' END AS world_card,
               CASE WHEN aa.travel_card > 0 THEN 'Y' ELSE 'N' END travel_card,
               CASE WHEN aa.five_profession_card > 0 THEN 'Y'
                    ELSE 'N' END AS five_profession_card
        FROM (
              SELECT bacno,
                     SUM(CASE WHEN agno = '0032' THEN 1
                              ELSE 0 END) AS travel_card,
                     SUM(CASE WHEN agno IN ('0114','0115','0116','0117','0118')
                              THEN 1 ELSE 0 END) AS five_profession_card,
                     SUM(CASE WHEN agno IN ('A008','A011','A012') THEN 1
                              ELSE 0 END) AS world_card
              FROM mlaas_rawdata.{self.source_table_name}
              WHERE curcd IN ('0','9') AND catp2='0'
              GROUP BY bacno
             )aa
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
            with_cc,
            world_card,
            travel_card,
            five_profession_card
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)

        assert all(table.with_cc.isin(['Y', 'N']))
        assert all(table.world_card.isin(['Y', 'N']))
        assert all(table.travel_card.isin(['Y', 'N']))
        assert all(table.five_profession_card.isin(['Y', 'N']))
