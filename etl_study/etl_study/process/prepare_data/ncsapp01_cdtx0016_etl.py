from src.common.prepare_data.table_etl_base import TableETLBase


class NCSAPP01_CDTX0016ETL(TableETLBase):
    @property
    def source_table_name(self):
        return "NCSAPP01_CDTX0016"

    @property
    def tmp_column_types(self):
        return {
            'exist_monin': 'bigint'
        }

    @property
    def tmp_column_defaults(self):
        return {
            'exist_monin': "-1"
        }

    def check_source(self):
        sqlstring = '''
            SELECT
            count(*)
        FROM mlaas_rawdata.NCSAPP01
        '''
        row_count_ncsapp01 = self.select_table(
            'rawdata', sqlstring)['count'].values[0]
        assert row_count_ncsapp01 > 0

        sqlstring = '''
            SELECT
            count(*)
        FROM mlaas_rawdata.CDTX0016_HIST
        '''
        row_count_cdtx0016 = self.select_table(
            'rawdata', sqlstring)['count'].values[0]
        assert row_count_cdtx0016 > 0

    @property
    def etlSQL(self):
        """
        【exist_monin】

        使用NCSAPP01大消金平台-信貸申請書資料
        1.篩選【專案別 project】不等於"信貸紓困-勞工"或"行員貸款"
        2.篩選【撥款完成時間 paytm】不等於null值
        3.依【撥款完成時間 paytm】進行各ID的遞減排序，給予案件序號
        4.篩選序號為1，並取得aid、【撥款完成時間 paytm】、【年收入 aannualpay】/12，
          並分別命名為id、pay_date、monin

        使用CDTX0016 卡友通信貸款申請明細日檔
        1.篩選【放款帳號 lacno】不等於空值
        2.依【實際撥款日 t516d】進行各ID的遞減排序，給予案件序號
        3.篩選序號為1，並取得ID、【實際撥款日 t516d】、【平均月收入 monin】，並分別命名為id、pay_date、monin

        合併上述兩張表(使用union all)
        1.依【pay_date】進行各ID的遞減排序，給予案件序號
        2.篩選序號為1，作為顧客行內最終舊貸月收入資料
        """

        sqlstring = '''
            SELECT MERGE_SORT.id AS cust_no,
                   CAST(MERGE_SORT.monin AS BIGINT) AS exist_monin
            FROM (
                    SELECT MERGE.*, ROW_NUMBER()
                           OVER(PARTITION BY id ORDER BY pay_date DESC) AS seq
                    FROM(
                         SELECT paytm AS pay_date, aid AS id, monin
                         FROM (
                                SELECT aid,
                                       paytm,
                                       aannualpay/12 AS monin,
                                       ROW_NUMBER()
                                       OVER(PARTITION BY aid
                                            ORDER BY paytm DESC) AS seq
                                FROM mlaas_rawdata.NCSAPP01
                                WHERE project NOT IN ('信貸紓困-勞工','行員貸款')
                                AND paytm IS NOT NULL
                               ) AS NCSAPP01
                         WHERE seq = 1
                         UNION ALL
                         SELECT t516d AS pay_date, appid AS id ,monin
                         FROM (
                                SELECT t516d,
                                       appid,
                                       monin,
                                       ROW_NUMBER()
                                       OVER(PARTITION BY appid
                                            ORDER BY t516d DESC) AS seq
                                FROM mlaas_rawdata.CDTX0016_HIST
                                WHERE lacno <> ''
                              ) AS TX0016
                         WHERE seq = 1
                         ) AS MERGE
                ) AS MERGE_SORT
            WHERE seq = 1
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
            exist_monin
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(table.exist_monin >= -1)
