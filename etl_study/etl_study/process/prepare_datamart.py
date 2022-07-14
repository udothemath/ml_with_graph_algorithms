from src.config import TABLE_SUBFIX
from src.common.prepare_data.datamart_base import DataMartBase
from src.process.prepare_data.cm_customer_cm_account_etl import CM_CUSTOMER_CM_ACCOUNTETL
from src.process.prepare_data.cdca0001_etl import CDCA0001ETL
from src.process.prepare_data.witwo172_etl import WITWO172ETL
from src.process.prepare_data.cmch0001_etl import CMCH0001ETL
from src.process.prepare_data.cm_acct_dp_payroll_txn_m_etl import CM_ACCT_DP_PAYROLL_TXN_METL
from src.process.prepare_data.ncsapp01_cdtx0016_etl import NCSAPP01_CDTX0016ETL
from src.process.prepare_data.bam087_etl import BAM087ETL
from src.process.prepare_data.jas002_etl import JAS002ETL
from src.process.prepare_data.krm001_etl import KRM001ETL
from src.process.prepare_data.krm040_etl import KRM040ETL
from src.process.prepare_data.stm007_etl import STM007ETL
from src.process.prepare_data.upl_pd_lgd_result_etl import UPL_PD_LGD_RESULTETL


class PrepareDataMartOperator(DataMartBase):
    def __init__(self, dag=None,
                 target_table_name='cust_datamart' + TABLE_SUBFIX):
        super(PrepareDataMartOperator, self).__init__(
            dag=dag,
            operator_name='prepare_datamart',
            target_table_name=target_table_name
        )

    def build_etl_objects(self):
        """
        於此填寫ETL物件
        """
        return {
            'cm_customer_cm_account_etl': CM_CUSTOMER_CM_ACCOUNTETL(),
            'cdca0001_etl': CDCA0001ETL(),
            'witwo172_etl': WITWO172ETL(),
            'cmch0001_etl': CMCH0001ETL(),
            'cm_acct_dp_payroll_txn_m_etl': CM_ACCT_DP_PAYROLL_TXN_METL(),
            'ncsapp01_cdtx0016_etl': NCSAPP01_CDTX0016ETL(),
            'bam087_etl': BAM087ETL(),
            'jas002_etl': JAS002ETL(),
            'krm001_etl': KRM001ETL(),
            'krm040_etl': KRM040ETL(),
            'stm007_etl': STM007ETL(),
            'upl_pd_lgd_result_etl': UPL_PD_LGD_RESULTETL()
        }

    @property
    def population_etl_name(self):
        """
        設定母體ETL: 使用build_etl_objects 中的key名稱來指定
        """
        return "cm_customer_cm_account_etl"
