"""
組件DataMart用的物件
"""
import abc
import logging

from src.common.prepare_data.prepare import Prepare


class DataMartBase:
    """
    組件DataMart用的物件
    """

    def __init__(self, dag=None, operator_name='cust_datamart_etl',
                 target_table_name='cust_datamart'):
        if dag is not None:
            self.dag = dag
            self.dag.doc_md = __doc__
        else:
            self.dag = None
        self.operator_name = operator_name
        # 結果表名稱
        self.target_table_name = target_table_name
        # 設定所有欄位產製的ETL
        self.etl_objects = self.build_etl_objects()
        assert isinstance(self.etl_objects, dict)
        # 指定母體ETL
        assert isinstance(self.population_etl_name, str)
        assert self.population_etl_name in self.etl_objects.keys()
        self.prepare_obj = self.__build_prepare_obj()

    @abc.abstractmethod
    def build_etl_objects(self):
        """
        於此填寫各欄位ETL之物件

        Example:

        return {
            'cm_customer_cm_account_etl': CM_CUSTOMER_CM_ACCOUNTETL(),
            'cdca0001_etl': CDCA0001ETL(),
            'witwo172_etl': WITWO172ETL(),
            'cmch0001_etl': CMCH0001ETL(),
            'cm_acct_dp_payroll_txn_m_etl': CM_ACCT_DP_PAYROLL_TXN_METL()
        }

        """
        pass

    @property
    @abc.abstractmethod
    def population_etl_name(self):
        """
        設定母體: 使用build_etl_objects 中的key名稱來指定

        Example:

        return "cm_customer_cm_account_etl"
        """
        pass

    def __build_prepare_obj(self):
        etls = [self.etl_objects[self.population_etl_name]]
        for etl_name in self.etl_objects.keys():
            if etl_name != self.population_etl_name:
                etls.append(self.etl_objects[etl_name])
        return Prepare(
            etls,
            target_table_name=self.target_table_name
        )

    def run(self):
        prepare = self.__build_prepare_obj()
        prepare.drop_target_table()
        prepare.setup()
        logging.info('[run] Initialized')
        for etl in prepare.etls:
            etl.run()
        logging.info('[run] Stage 1 Success')
        prepare.join_n_insert()
        logging.info('[run] Stage 2 Success')
        prepare.check_all_target_columns()
        prepare.truncate_all_tmps()
        logging.info('[run] Stage 3 Success')

    def setup(self):
        self.prepare_obj.setup()

    def build_task_group(self, upstream_task=None, upstream_connect_to=None):
        from airflow.utils.task_group import TaskGroup
        from airflow.operators.python import PythonOperator
        with TaskGroup(group_id=self.operator_name) as operator_level_task_group:
            task_groups = []
            for key, etl in self.etl_objects.items():
                task_group = etl.build_task_group(
                    group_id=key, dag=self.dag)
                task_groups.append(task_group)
                if upstream_task is not None:
                    if upstream_connect_to is None:
                        # All groups are connected to the same upstream task
                        task_group.set_upstream(upstream_task)
                    else:
                        # Only one group is connected to the upstream task
                        assert isinstance(upstream_connect_to, str)
                        if key == upstream_connect_to:
                            task_group.set_upstream(upstream_task)

            join_n_insert_task = PythonOperator(
                task_id='join_n_insert',
                python_callable=self.prepare_obj.join_n_insert,
                dag=self.dag
            )
            join_n_insert_task.set_upstream(task_groups)

            check_target_task = PythonOperator(
                task_id='check_target',
                python_callable=self.prepare_obj.check_all_target_columns,
                dag=self.dag
            )

            join_n_insert_task >> check_target_task

        return operator_level_task_group
