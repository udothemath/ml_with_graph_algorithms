from src.common.prepare_data.table_etl_base import TableETLBase


class BAM087ETL(TableETLBase):
    @property
    def source_table_name(self):
        return "BAM087"

    @property
    def tmp_column_types(self):
        return {
            'bam087_ind': 'char(1)',
            'upl_amt': 'float'
        }

    @property
    def tmp_column_defaults(self):
        return {
            'bam087_ind': "'N'",
            'upl_amt': "0"
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
        ETL的SQL於此定義

        【bam087_ind】
        1.篩選【查詢日 querydate】為近6個月(含6個月)
        2.依上述篩選後給予註記"Y"，其餘為"N"

        【upl_amt】
        1.篩選【查詢日_querydate】為近6個月(含6個月)
        且
        2.(1)【擔保品類別 IS_KIND】為「00 純信用」或「01 信用保險」，
              且【科目別ACCOUNT_CODE】為「H 中期放款」、「O 其他」、「G 貸放會金（信用）」、
              「K存單質押放款」、「L 應收保證款項」、「I 長期放款」、「E 其他短期放款」、「C 透支」，
              且【科目別註記ACCOUNT_CODE2】為空值，且符合下面兩項其中一個
              條件：
                 (i)【資金流向註記 IB_MARK】不為空值 ，
                     且【政府專案貸款分類代碼PROJECT_CODE】不為「54原住民微型經濟活動貸款」
                 (ii)【資金流向註記 IB_MARK】為「*」，且【共同借款註記CO_LOAN】為「*」
          (2)或【科目別 account_code】為Y
        且
        3.【銀行代號_bank_code】為808
        4.依上述篩選取得id、【未逾期金額 loan_amt】+【逾期未還金額 pass_due_amt】*1000，
          並分別命名為id、upl_amt
        """

        sqlstring = f'''
       SELECT cust_no, bam087_ind, SUM(CASE WHEN AMT.upl_amt IS NULL THEN 0
                                                 ELSE AMT.upl_amt END) AS upl_amt
            FROM (
                     SELECT id AS cust_no, 'Y' AS bam087_ind, MAX(querydate) AS querydate
                     FROM if_jcic_superset_online.{self.source_table_name}
                         WHERE querydate >= DATE_TRUNC('month',ADD_MONTHS(CURRENT_DATE(),-6))
                     GROUP BY id
                 ) AS IND
            LEFT JOIN (
                           SELECT id, (loan_amt + pass_due_amt)*1000 AS upl_amt, querydate
                           FROM if_jcic_superset_online.{self.source_table_name}
                           WHERE (
                                    (
                                        is_kind IN ('00', '01')
                                        AND account_code IN ('H','O','G','K','L','I','E','C')
                                        AND account_code2 IS NULL
                                        AND (
                                                (ib_mark IS NULL AND project_code != '54')
                                                or
                                                (ib_mark = '*' AND co_loan = '*')
                                            )
                                    )
                                    OR
                                    (account_code = 'Y')
                                )
                           AND LEFT(bank_code,3) = '808'
                           AND QUERYDATE >= DATE_TRUNC('month',ADD_MONTHS(CURRENT_DATE(),-6))
                      )AMT ON IND.cust_no = AMT.id AND IND.querydate = AMT.querydate
              GROUP BY cust_no, bam087_ind



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
            bam087_ind,
            upl_amt
        FROM {self.schema_name}.{target_table_name}
        '''
        table = self.select_table('feature', sqlstring)

        assert all(table.bam087_ind.isin(['Y', 'N']))
        assert all(table.upl_amt >= 0)
