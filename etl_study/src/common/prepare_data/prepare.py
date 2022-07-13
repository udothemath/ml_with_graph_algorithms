"""
TODO:
- [X] 有可能要再新增 product 作為 primary key => 若有兩種cust_apply_(01/02) & cust_logic_(01/02)，
各自的產出結果可再於final_etl.py進行合併
- [X] full join 改成left join到一個有全部母體顧客的資料表，這樣可以確
保有可預期的data mart primary key。
- [X] add code implementation example along with the instructions of Prepare
- [X] build a tutorial notebook follows the instructures in Prepare.
- [ ] complete docstring of Prepare class
- [X] build up dag.py with entire datamart preparation etl.
- [X] create auto generation of the dag code of the prepare data. -> In datamart_base.py
- [ ] replace table check python code to sql.
- [X] implement a function that regenerate all tmps
(delete -> check_source -> select_n_transform -> insert)
- [X] implement check_all_tmps in Prepare
- [X] suppress get_conn printed info.
- [X] speed up insert using cursor.mogrify()
- [X] simplify check_source
- [X] implement check_target_columns in Prepare
- [X] implement delete_all_tmps in Prepare
- [X] Put all .py related to prepare into a folder named prepare.
- [X] set get_conn connections from airflow to mlaas_tools.
- [X] incorporate with airflow dag and confirm workability
    - [X] confirm PostgresHook workable
    - [X] confirm airflow dag workable
- [X] move here the SQL functions.
- [X] check if running select and insert jointly speed up ETL
- [X] check if running two insert jointly speed up ETL (X)
- [X] put conn getting into a function
- [X] build ETLUnit class with abstract functions defined in stage 1, 2, and 3.
        1. [X] with methods that do check of stage 1
        2. [X] with methods that do insert of stage 1
        3. [X] with methods that "Select X + Transform into X'"
        4. [X] with methods that "check X'"
- [X] A SQL integration class that takes TableETLBase and
        1. [X] create the select/join/transform/insert SQL of stage 2,
        2. [X] do "delete all" of stage 3.
- [X] insert to db table by table => change to mogorify version.
- [X] joining in the feature db
- [X] use target_schema_name and target_table_name in functions with SQL.
- [X] build testing for each example xxx_etl.py
"""
import os
import sys
import importlib
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
ETLBase = getattr(importlib.import_module('src.common.etl_base'), 'ETLBase')


class Prepare(ETLBase):
    """
    這裡放置產出顧客彙總表的ETL函數，用來把預審用欄位從各個來源表匯集為一個彙總表
    (target_table)。

    整個ETL流程我們可以分為 stage 1~3，以下分別說明各個stage所需用到的函數:

    1) Stage 1: 於rawdata_db 檢查某原始來源表(A、B、C...)內容 ->
        拉取資料表並進行資料清理或轉換 ->
        將轉換後結果存入feature_db為中繼表(A'、B'、C'...)。

        1.1) 實作步驟:

            繼承TableETLBase並覆寫 abstractmethods:

            1.1.1) 覆寫source_table_name 設定來源表格名稱

            1.1.2) 覆寫tmp_column_types  設定中繼表(tmp)欄位規格

            1.1.3) 覆寫select_n_transform 撰寫來源表欄位轉換方法

        1.2) 確認實作結果:

            1.2.1) 執行 select_n_transform 進行欄位轉換

            1.2.2) 執行 insert_tmp_into_feature_db 將中繼表存入feature db

            1.2.3) 執行 show_tmp_in_feature_db 確認儲存之中繼表正確。
                (若有問題，執行 delete_tmp_from_feature_db 刪除中繼表後，回到步驟1.1。)

    2) Stage 2: 於feature_db檢查中繼表(A'、B'、C')... ->
        將中繼表A'、B'、C'進行合併、補齊空值後帶入target_table。

        2.1) 實作步驟:

            2.1.1) 覆寫 tmp_column_defaults 設定表格合併時，某表無對應顧客時，
                需將空值填補之預設值。

            2.1.2) 將新的TableETLBase物件納入Prepare中:
                prepare = Prepare([cdca0001_etl, witwo371_etl])

        2.2) 確認實作結果:

            2.2.1) 執行 prepare.check_all_tmps，確認所有來源表之對應中繼表內容已儲存於
            feature db。若有錯誤，執行 prepare.regenerate_all_tmps，重新產生所有中繼表。

            2.2.2) 執行 prepare.join_n_insert，將中繼表合併於顧客資料彙總表(cust_info_table)中。

            2.2.3) 執行 prepare.show_target_table()，確認儲存之彙總表正確。
                (若有問題，執行 prepare.drop_target_table() 刪除彙總表後，回到步驟2.1。)

    3) Stage 3: 於target_table檢查各個來源表所對應之結果欄位內容 -> 將中繼表A'、B'、C'刪除。
            3.1) 實作步驟:

                3.1.1) 於各個TableETLBase覆寫 check_target_columns 進行顧客資料彙總表欄位的檢查
            3.2) 執行:

                3.2.1) 執行 prepare.check_all_target_columns 確認彙總表所有欄位皆正確 ，
                若有錯誤，回到 Stage 1 / Stage 2 檢查欄位產製邏輯是否錯誤。

                3.2.2) 執行 prepare.del_all_tmps 把所有中繼表都刪除
    """

    def __init__(self, etls, target_table_name='cust_datamart'):
        assert len(etls) > 0
        self.etls = etls
        self.target_table_name = target_table_name

    def show_target_table(self):
        sqlstring = f"""
        SELECT *
        FROM {self.schema_name}.{self.target_table_name}
        """
        logging.info(f'[show_target_table] sql: \n{sqlstring}')
        return self.select_table('feature', sqlstring)

    def drop_target_table(self):
        sqlstring = f"""
        DROP TABLE IF EXISTS {self.schema_name}.{self.target_table_name};
        """
        print(sqlstring)
        logging.info(sqlstring)
        self.execute_sql('feature', sqlstring)
        logging.info('[drop_target_table] SUCCESS')
        return True

    def truncate_target_table(self):
        self.__create_target_table()
        sqlstring = f"""
        TRUNCATE {self.schema_name}.{self.target_table_name};
        """
        logging.info(sqlstring)
        self.execute_sql('feature', sqlstring)
        logging.info('[truncate_target_table] SUCCESS')
        return True

    def __grant_target_table(self):
        for user in self.db_users:
            sqlstring = f"""
                GRANT ALL
                ON TABLE {self.schema_name}.{self.target_table_name}
                TO {user};
            """
            print(sqlstring)
            logging.info(f'[__grant_target_table] {sqlstring}')
            self.execute_sql('feature', sqlstring)
            logging.info('[__grant_target_table] SUCCESS')

    def regenerate_all_tmps(self):
        for table_obj in self.etls:
            table_obj.check_source()
            print("[regenerate_all_tmps] DONE check_source")
            table_obj.select_n_insert()
            print("[regenerate_all_tmps] DONE select_n_insert")

        print('[regenerate_all_tmps] SUCCESS')
        return True

    def check_all_tmps(self):
        for table_obj in self.etls:
            table_obj.check_tmp()
        print('[check_all_tmps] SUCCESS')
        logging.info('[check_all_tmps] SUCCESS')
        return True

    def join_n_insert(self):
        assert len(self.etls) > 0
        self.__create_target_table()
        self.truncate_target_table()
        if len(self.etls) > 1:
            sqlstring = LeftJoinSQLGenerator.generate(
                self.etls,
                self.schema_name, self.target_table_name
            )
        elif len(self.etls) == 1:
            sqlstring = f"""
            INSERT INTO {self.schema_name}.{self.target_table_name}
            SELECT *
            FROM {self.schema_name}.{self.etls[0].tmp_table_name}
            """
        logging.info(sqlstring)
        self.execute_sql('feature', sqlstring)
        logging.info('[join_n_insert] SUCCESS')
        return True

    def check_all_target_columns(self):
        sqlstring = f'''
            SELECT
            count(*)
        FROM {self.schema_name}.{self.target_table_name}
        '''
        target_cnt = self.select_table('feature', sqlstring)['count'].values[0]
        sqlstring = f'''
            SELECT
            count(*)
        FROM {self.schema_name}.{self.etls[0].tmp_table_name}
        '''
        pop_cnt = self.select_table('feature', sqlstring)['count'].values[0]
        assert target_cnt == pop_cnt
        logging.info(f'[check_all_target_columns]'
                     'Target {self.schema_name}.{self.target_table_name} Count ({target_cnt})'
                     f'Equals Population Count ({pop_cnt})')
        for table_obj in self.etls:
            table_obj.check_target_columns(self.target_table_name)
        print('[check_all_target_columns] SUCCESS')
        logging.info('[check_all_target_columns] SUCCESS')
        return True

    def delete_all_tmps(self):
        for table_obj in self.etls:
            table_obj.delete_tmp_from_feature_db()
        print('[delete_all_tmps] SUCCESS')
        return True

    def setup(self):
        self.create_all()
        self.truncate_all()
        self.grant_all()
        print('[setup] SUCCESS')
        return True

    def grant_all(self):
        for table_obj in self.etls:
            table_obj.grant_tmp()
        self.__grant_target_table()

    def truncate_all(self):
        self.truncate_all_tmps()
        self.truncate_target_table()

    def create_all(self):
        for table_obj in self.etls:
            table_obj.create_tmp()
            logging.info(f'[create_all] CREATE {table_obj.tmp_table_name}')
        self.__create_target_table()
        logging.info(f'[create_all] CREATE TARGET {self.target_table_name}')

    def truncate_all_tmps(self):
        for table_obj in self.etls:
            table_obj.truncate_tmp_from_feature_db()
        logging.info('[truncate_all_tmps] SUCCESS')
        return True

    def __create_target_table(self):
        column_strs = []
        for table_obj in self.etls:
            for col, type_str in table_obj.tmp_column_types.items():
                column_strs.append(f'{col} {type_str}')
        if len(column_strs) > 0:
            column_sql_str = ',\n\t'.join(column_strs)
            sqlstring = f"""
            CREATE TABLE
            IF NOT EXISTS {self.schema_name}.{self.target_table_name}(
                cust_no char(24),
                {column_sql_str},
                Primary key(cust_no)
            );
            """
        else:
            sqlstring = f"""
            CREATE TABLE
            IF NOT EXISTS {self.schema_name}.{self.target_table_name}(
                cust_no char(24),
                Primary key(cust_no)
            );
            """
        logging.info(f'[__create_target_table] sql: \n{sqlstring}')
        self.execute_sql('feature', sqlstring)
        logging.info('[__create_target_table] SUCCESS')


class LeftJoinSQLGenerator():
    """
    表格合併SQL產生器

    透過以下方法可以產生表格合併的SQL:

    sqlstring = LeftJoinSQLGenerator.generate()
    """
    @staticmethod
    def generate(input_tables, schema_name, target_table_name):

        sqlstring = f"""
        INSERT INTO {schema_name}.{target_table_name}
        SELECT
            t1.cust_no,
            {LeftJoinSQLGenerator.create_full_null_filling_sql(input_tables)}
        {LeftJoinSQLGenerator.create_left_join_sql(input_tables)}
        """
        return sqlstring

    @staticmethod
    def create_cust_no_filling_sql(table_cnt):
        """
        產生cust_no欄位合併用的SQL片段

        Args:
            - table_cnt: 合併的表格數量
        Return:
            - result: SQL片段

        Example:

        INSERT INTO {schema_name}.{target_table_name}
        SELECT
            {LeftJoinSQLGenerator.create_cust_no_filling_sql(len(input_tables))},
            {LeftJoinSQLGenerator.create_null_filling_sql(input_tables)}
        {LeftJoinSQLGenerator.create_left_join_sql(input_tables)}

        """
        assert table_cnt > 1
        result = 'CASE '
        for i in range(1, table_cnt):
            result = result + \
                f'WHEN t{i}.cust_no IS NOT NULL THEN t{i}.cust_no\n'
        result = result + f'ELSE t{i+1}.cust_no END AS cust_no'
        return result

    @staticmethod
    def create_full_null_filling_sql(input_tables):
        """
        產生所有欄位如何補空值的SQL片段 (不含cust_no)

        Args:
            - input_tables: 一個list的 table_etl_base.TableETLBase 物件
        Return:
            - col_null_filling_sqlstring: 所有欄位如何補空值的SQL片段
        """
        sql_args = []
        for i, table_obj in enumerate(input_tables):
            for col_name, default in table_obj.tmp_column_defaults.items():
                sql_args.append((i + 1, col_name, default))

        col_null_filling_sqlstring = ",\n".join(
            [LeftJoinSQLGenerator.create_null_filling_sql(
                *args) for args in sql_args]
        )
        return col_null_filling_sqlstring

    @staticmethod
    def create_null_filling_sql(i, col_name, default):
        """
        產生某一欄位如何補空值的SQL片段

        Args:
            - i : 欄位來自第 i 張來源表
            - col_name: 欄位名稱
            - default: 空值要由何補上 (一個SQL片段 如 'N')
        Return:
            - SQL片段
        """
        return f"""
                CASE
                    WHEN t{i}.{col_name} IS NULL THEN {default}
                    ELSE t{i}.{col_name}
                END"""

    @staticmethod
    def create_left_join_sql(input_tables):
        """
        產生將所有資料表做Left Join的SQL片段

        Args:
            - input_tables: 一個list的 table_etl_base.TableETLBase 物件
        Return:
            - result: SQL片段
        """
        first_table = input_tables[0]
        result = f'''
            FROM {first_table.schema_name}.{first_table.tmp_table_name}
            AS t1'''
        for i, table in enumerate(input_tables[1:]):
            result = result + f"""
            LEFT JOIN {table.schema_name}.{table.tmp_table_name}
            AS t{i+2}
            ON t1.cust_no=t{i+2}.cust_no
            """
        return result
