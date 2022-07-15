"""
多產品利率定價實作框架
"""
import logging
import abc
from src.to_ods import TO_ODS_ETL
from src.trigger import Trigger


class PreApprovalBase:
    """
    提供多產品利率定價實作框架

    可串接多種cust_logic和cust_apply和final_etl.py
    """

    def __init__(self, dag=None, operator_name='pre_approval'):
        if dag is not None:
            self.dag = dag
            self.dag.doc_md = __doc__
        else:
            self.dag = None
        self.operator_name = operator_name
        # 設定各個預審方案結果產製的ETL
        self.__logic_apply_etls = self.logic_apply_etls
        assert isinstance(self.__logic_apply_etls, dict)
        # 設定整併所有方案的ETL程式
        self.__final_etl = self.final_etl
        self.__to_ods_etl = TO_ODS_ETL(
            final_table_name=self.__final_etl.target_table_name
        )
        self.__trigger = Trigger()

    @property
    @abc.abstractmethod
    def logic_apply_etls(self):
        """
        於此設定各個預審方案結果產製的ETL

        Example:

        return {
            'prod1': CustApply(
                input_table_name='cust_datamart',
                target_table_name = 'cust_result_prod_1'),
            'prod2': CustApply(
                input_table_name='cust_datamart',
                target_table_name = 'cust_result_prod_2')
        }
        """
        pass

    @property
    @abc.abstractmethod
    def final_etl(self):
        """
        設定整併所有方案的最終ETL程式

        Example:

        return FinalETL(
            target_table_name = 'esun_cust_loan_preapproval_data'
        )
        """
        pass

    def run(self):
        for prod_name, logic_apply_etl in self.__logic_apply_etls.items():
            logic_apply_etl.run()
            logging.info(f"[run] Finish Logic Apply for {prod_name}")
            logging.info(
                f'[run] Table {logic_apply_etl.target_table_name} Created')

        self.__final_etl.run()
        self.__final_etl.check_target()
        logging.info("[run] Finish Final ETL")
        logging.info(
            f'[run] Table {self.__final_etl.target_table_name} Created')
        self.__to_ods_etl.run()
        logging.info("[run] Finish TO ODS ETL")

    def setup(self):
        for prod_name, logic_apply_etl in self.__logic_apply_etls.items():
            logging.info(
                f"[setup] Start Setting Up the Tables for {prod_name}")
            logic_apply_etl.setup()
        self.__final_etl.setup()

    def build_task_group(self):
        from airflow.utils.task_group import TaskGroup
        from airflow.operators.python import PythonOperator
        with TaskGroup(group_id=self.operator_name) as operator_level_task_group:
            apply_groups = []
            for prod_name, logic_apply_etl in self.__logic_apply_etls.items():
                apply_group = logic_apply_etl.build_task_group(
                    group_id=prod_name, dag=self.dag)
                apply_groups.append(apply_group)

            final_task_group = self.__final_etl.build_task_group(
                group_id='final', dag=self.dag)

            to_ods_task = PythonOperator(
                task_id=f'to_ods_etl_{self.operator_name}',
                python_callable=self.__to_ods_etl.run,
                dag=self.dag
            )

            trigger_task = PythonOperator(
                task_id=f'trigger_{self.operator_name}',
                python_callable=self.__trigger.run,
                dag=self.dag
            )
            apply_groups >> final_task_group >> to_ods_task >> trigger_task
        return operator_level_task_group
