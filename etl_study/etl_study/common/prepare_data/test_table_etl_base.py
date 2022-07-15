"""
測試TableETLBase用的物件
"""
import abc
import pandas as pd
from src.common.prepare_data.prepare import Prepare


class TableETLBase:
    """
    測試TableETLBase用的物件
    """
    @property
    @abc.abstractmethod
    def ETLObj(self):
        """
        from src.process.prepare_data.bam087_etl import BAM087ETL

        Example:
        return BAM087ETL
        """
        pass

    @property
    def prepare(self):
        try:
            return self.__prepare
        except BaseException:
            self.__prepare = Prepare([self.etl])
            return self.__prepare

    @property
    def etl(self):
        try:
            return self.__etl
        except BaseException:
            self.__etl = self.ETLObj(alias='_alias_test')
            return self.__etl

    def test_source_table_name(self):
        assert isinstance(self.etl.source_table_name, str)

    def test_tmp_table_name(self):
        assert self.etl.source_table_name + self.etl.alias == self.etl.tmp_table_name

    def test_tmp_column_types(self):
        assert isinstance(self.etl.tmp_column_types, dict)

    def test_tmp_column_defaults(self):
        assert isinstance(self.etl.tmp_column_defaults, dict)

    def test_etlSQL(self):
        assert isinstance(self.etl.etlSQL, str)
        assert 'SELECT' in self.etl.etlSQL
        assert 'FROM' in self.etl.etlSQL

    def test_source_db(self):
        assert isinstance(self.etl.source_db, str)
        assert self.etl.source_db == 'feature' or self.etl.source_db == 'rawdata'

    def test_check_source(self):
        self.prepare.drop_target_table()
        self.prepare.delete_all_tmps()
        print('Initialized')
        self.etl.check_source()

    def test_create_tmp(self):
        self.etl.create_tmp()

    def test_select_n_insert(self):
        self.etl.select_n_insert()

    def test_grant_tmp(self):
        self.etl.grant_tmp()

    def test_show_tmp_in_feature_db(self):
        table = self.etl.show_tmp_in_feature_db()
        assert isinstance(table, pd.DataFrame)

    def test_check_tmp(self):
        self.etl.check_tmp()

    def test_check_target_columns(self):
        self.prepare.join_n_insert()
        print('Stage 2 Success')
        self.etl.check_target_columns(self.prepare.target_table_name)

    def test_truncate_tmp_from_feature_db(self):
        self.etl.truncate_tmp_from_feature_db()

    def test_delete_tmp_from_feature_db(self):
        self.etl.delete_tmp_from_feature_db()
        print('Stage 3 Success')

    def test_drop_target_table(self):
        self.prepare.drop_target_table()
