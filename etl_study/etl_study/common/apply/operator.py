"""
進行平行化並將結果進行整合的DAG物件

TODO:
- [X] merge_etl的merging tasks組成task group
"""
from importlib import import_module
from src.common.apply.apply_base import ApplyBase
from src.common.apply.merge_etl import MergeETL
from src.config import SPLIT_DEGREE


class ApplyOperator:
    """
    進行平行化並將結果進行整合的DAG物件
    """

    def __init__(self, apply_module, apply_class,
                 custom_timeout_minutes=24 * 60):
        forking_apply_cls = getattr(import_module(apply_module), apply_class)
        self.__apply_module = apply_module
        self.__apply_class = apply_class
        # 指定模型ETL
        self.__applies = {}

        # 要做幾個分流process，為16的次方次數
        for i in range(16**SPLIT_DEGREE):
            self.__applies[i] = forking_apply_cls(
                group_id=i
            )
        # 拿取create_SQL
        self.__merge_etl = MergeETL(
            forking_apply_cls().input_table_name,
            forking_apply_cls().target_table_name,
            forking_apply_cls().target_create_SQL(),
            split_condition_extension=forking_apply_cls().split_condition_extension
        )
        self.__custom_timeout_minutes = custom_timeout_minutes

    def run(self):
        self.setup()
        for i in range(16**SPLIT_DEGREE):
            self.__applies[i].run()
            self.__applies[i].check_target()
        self.__merge_etl.run()
        print("Apply ETL DONE")

    def setup(self):
        for i in range(16**SPLIT_DEGREE):
            self.__applies[i].setup()
        self.__merge_etl.setup()

    def build_task_group(self, group_id, dag=None):
        """
        Args:
            - group_id: (str) name of the task group
        Return:
            - task_group: (TaskGroup) 回傳的tasks group
        TODO:
        - [X] grouping tasks
        """
        from airflow.utils.task_group import TaskGroup
        from airflow.operators.python import PythonOperator
        with TaskGroup(group_id=group_id) as task_group:
            run_tasks = []
            for i in range(16**SPLIT_DEGREE):
                run_task = ApplyBase.build_bash_operator(
                    apply_module=self.__apply_module,
                    apply_class=self.__apply_class,
                    group_id=i,
                    task_id=f'run_{i}',
                    dag=dag
                )
                run_tasks.append(run_task)

            check_tasks = []
            for i in range(16**SPLIT_DEGREE):
                check_task = PythonOperator(
                    task_id=f'check_{i}',
                    python_callable=self.__applies[i].check_target,
                    dag=dag
                )
                check_tasks.append(check_task)

            merge_task_group = self.__merge_etl.build_task_group(
                group_id='merge', dag=dag)

            for run_task, check_task in zip(run_tasks, check_tasks):
                run_task >> check_task
            check_tasks >> merge_task_group

        return task_group
