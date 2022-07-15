from src.config import PRE_APPROVAL_TIMEOUT
from src.common.pre_approval_base import PreApprovalBase
from src.process.final_etl import FinalETL
from src.common.apply.operator import ApplyOperator


class PreApprovalOperator(PreApprovalBase):
    """
    多產品利率定價實作框架

    可串接多種預審方案運算程式(e.g., cust_apply.py)
    以及整合各預審產出表的ETL程式 (i.e., final_etl.py)
    """

    def __init__(self, dag=None):
        super(PreApprovalOperator, self).__init__(
            dag=dag,
            operator_name='pre_approval'
        )

    @property
    def logic_apply_etls(self):
        """
        於此設定各個預審方案結果產製的ETL

        Example:

        return {
            'prod1': ApplyOperator(
                'src.process.cust_apply',
                'CustApply',
                custom_timeout_minutes=PRE_APPROVAL_TIMEOUT,
                operator_name='prod1'
            )
            'prod2': ApplyOperator(
                'src.process.cust_apply_prod2',
                'CustApply',
                custom_timeout_minutes=PRE_APPROVAL_TIMEOUT,
                operator_name='prod2'
            )
        }

        """
        # prod1為一次撥付型、prod2為循環貸
        return {
            'prod1': ApplyOperator(
                'src.process.one.cust_apply_one',
                'CustApply',
                custom_timeout_minutes=PRE_APPROVAL_TIMEOUT
            ),
            'prod2': ApplyOperator(
                'src.process.rl.cust_apply_rl',
                'CustApply',
                custom_timeout_minutes=PRE_APPROVAL_TIMEOUT
            )

        }

    @property
    def final_etl(self):
        """
        設定整併所有方案的最終ETL程式
        """
        return FinalETL(
            source_table_name=None,
            target_table_name='esun_cust_loan_preapproval_data'
        )
