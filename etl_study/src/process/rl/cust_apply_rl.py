from importlib import import_module
from src.common.apply.forking_base import ForkingBase
from src.config import TABLE_SUBFIX
"""
預審邏輯套用到顧客datamart之ETL程式
"""


class CustApply(ForkingBase):
    def __init__(self, input_table_name='cust_datamart' + TABLE_SUBFIX,
                 target_table_name='cust_tmp_result_revolving' + TABLE_SUBFIX, group_id=-1, verbose=False):
        super(CustApply, self).__init__(
            input_db='feature',
            input_table_name=input_table_name,
            target_table_name=target_table_name,
            input_schema_name='cc_ln_pre_approval',
            target_schema_name='cc_ln_pre_approval',
            set_index='cust_no',
            verbose=verbose,
            group_id=group_id
        )

    @property
    def logic_object(self):
        """
        引入要套用的邏輯物件 (繼承LogicBase)

        Example:

        from src.process.cust_logic_one import CustLogic
        return CustLogic
        """
        cust_logic = import_module('src.process.rl.cust_logic_rl')
        return cust_logic.CustLogic

    def get_target_create_SQL(self, schema_name, table_name):
        """
        產生建立產出表的SQL

        Args:
            - schema_name: 產出表schema名稱
            - table_name: 產出表名稱
        Returns:
            - sqlstring: 產製的SQL
        """
        sqlstring = f'''
            CREATE TABLE
            IF NOT EXISTS {schema_name}.{table_name}(
            cust_no CHAR(24),
            "group" CHAR(2),
            product CHAR(2),
            apdlv NUMERIC,
            lgd NUMERIC,
            base_int NUMERIC,
            profit_int NUMERIC,
            pre_net_income BIGINT,
            max_limit BIGINT,
            interest_rate_1 NUMERIC,
            period_1 INT,
            interest_rate_2 NUMERIC,
            period_2 INT,
            fee_amount INT,
            all_rate NUMERIC,
            Primary key(cust_no)
            );'''
        return sqlstring

    def get_forking_partial_select_SQL(
            self, schema_name, table_name, offset, size, index_col, condition_SQL):
        """
        產生batch-by-batch Select時，所用的SQL

        Args:
            - schema_name: input表schema名稱
            - table_name: input表名稱
            - offset: Select的起始行數
            - size: batch的大小
        Returns:
            - select_sql: 選表的SQL
        """
        select_sql = f'''
            SELECT
                {index_col},
                with_cc,
                krm040_ind,
                bam087_ind,
                krm001_ind,
                jas002_ind,
                exist_monin,
                cc_monin,
                salary_monin,
                upl_amt,
                travel_card,
                five_profession_card,
                world_card,
                wm_cust,
                gov_employee,
                military_police_firefighters,
                salary_ind,
                pd_value,
                pd_grade,
                lgd
            FROM {schema_name}.{table_name}
            WHERE {condition_SQL}
            ORDER BY {index_col}
            /*make sure to order by key to avoid tie*/
            LIMIT {size}
            OFFSET {offset}
            '''
        return select_sql

    @property
    def split_condition_extension(self):
        """
        若在SELECT的時候要加入額外條件請覆寫此函式
        此段SQL會併到condition_SQL的後面
        """
        return "pd_value <> -1 AND pd_grade <> -1 AND lgd <> -1 AND " + \
            "(exist_monin > 24000 OR cc_monin > 24000 OR salary_monin > 24000)"

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
            SELECT COUNT(*)
        FROM {schema_name}.{table_name}
        '''
        row_count = self.select_table('feature', sqlstring)['count'][0]
        assert row_count > 0
        assert self._check_group_fee_amount(schema_name, table_name)
        assert self._check_product(schema_name, table_name)
        assert self._check_apdlv(schema_name, table_name)
        assert self._check_lgd(schema_name, table_name)
        assert self._check_pre_net_income(schema_name, table_name)
        assert self._check_max_limit(schema_name, table_name)
        assert self._check_period(schema_name, table_name)

        return True, row_count

    def _check_group_fee_amount(
            self, schema_name, table_name):
        sqlstring = f'''
            SELECT DISTINCT "group", profit_int, interest_rate_1, fee_amount
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(table['group'].isin(['01', '02', '03']))

        assert all(table.loc[table['group'] ==
                   '01', 'fee_amount'].isin([0]))

        assert all(table.loc[table['group'] ==
                   '02', 'fee_amount'].isin([0]))

        assert all(table.loc[table['group'] ==
                   '03', 'fee_amount'].isin([0]))

        return True

    def _check_product(self, schema_name, table_name):
        sqlstring = f'''
            SELECT DISTINCT product
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(table['product'].isin(['01', '02']))
        return True

    def _check_apdlv(self, schema_name, table_name):
        sqlstring = f'''
            SELECT DISTINCT apdlv
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(round(table['apdlv'], 5)
                   .isin([0.0015, 0.00225, 0.00338, 0.00506,
                          0.00759, 0.01139, 0.01709, 0.02563,
                          0.03844, 0.05766, 0.0865, 0.12975,
                          0.19462, 0.29193, 0.43789]))
        return True

    def _check_lgd(self, schema_name, table_name):
        sqlstring = f'''
            SELECT DISTINCT lgd
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(round(table['lgd'], 3).isin([0.441, 0.458, 0.462,
                                                0.513, 0.53, 0.535,
                                                0.598, 0.652]))
        return True

    def _check_pre_net_income(self, schema_name, table_name):
        sqlstring = f'''
            SELECT min(pre_net_income) as min_pre_net_income
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert table['min_pre_net_income'][0] >= 24000
        return True

    def _check_max_limit(self, schema_name, table_name):
        sqlstring = f'''
            SELECT min(max_limit) as min_max_limit, max(max_limit) as max_max_limit
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert table['min_max_limit'][0] >= 0
        assert table['max_max_limit'][0] <= 200000
        return True

    def _check_period(self, schema_name, table_name):
        sqlstring = f'''
            SELECT distinct period_1, period_2
        FROM {schema_name}.{table_name}
        '''
        table = self.select_table('feature', sqlstring)
        assert all(table['period_1'].isin([12]))
        assert all(table['period_2'].isin([0]))
        return True
