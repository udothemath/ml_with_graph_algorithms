"""
設定參數於此定義
"""
from datetime import timedelta, datetime
from airflow.models import Variable
# ============INPUT============
PROJ_NAME = "cc_ln_pre_approval"    # 修改成你的專案名稱
SCHEMA_NAME = 'cc_ln_pre_approval'  # 修改成你 feature db 的 schema 名稱
VERSION = "v1.0.20"                 # 必須與 git tag 或 PR label 一致，p.s., 版號不能超過2位數

CONNECTION = 'airflow'          # airflow | mlaas_tools


DAG_ARGS = {
    "owner": "esb21375",
    "start_date": datetime(2022, 1, 1),
    "email": [
        "jeffreylin-21375@email.esunbank.com.tw",
        "willyshao514-21313@esunbank.com.tw",
        "weitrtce-16209@email.esunbank.com.tw",
        "mhsu-21978@esunbank.com.tw"
    ],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "trigger_rule": "all_success",
    "depends_on_past": False,  # 先設 False，我們還沒有需要用到backfill
    "wait_for_downstream": False
}

DB_USERS = [                        # 調整此列表以移除或新增專案成員
    'cc_ln_pre_approval',
    'esb21375',
    'esb21313',
    'esb18509',
    'esb21315',
    'esb16209',
    'esb21774',
    'esb21978'
]
TRIGGER_PATH = '/tmp'
# Default Parameters:
DAG_VERSION = '_v14'
SPLIT_DEGREE = 1  # Splitting model_apply into 1(0), 16(1), or 256(2)
INSERT_PER_TASK = 8
PARALLEL_CNT = 1
PRE_APPROVAL_PARALLEL_CNT = 1
BATCH_SIZE = 1000
MAX_BATCH = None
CONCURRENCY = 10
PRE_APPROVAL_TIMEOUT = 1440  # Assume 1 core per task
MODEL_TIMEOUT = 1280  # Assume 1 core per task
SHORT_TIMEOUT_MINUTES = 10  # Set if SHORT_TIMEOUT == True
SHORT_TIMEOUT = False
DEBUG_MODE = False
# In Aicloud:
TABLE_SUBFIX = DAG_VERSION
# In Staging or Production:
try:
    if Variable.get('environment') == 'stage':
        CONCURRENCY = 4
        TABLE_SUBFIX = ''
        # For Faster Testing:
        """
        DEBUG_MODE = True  # fast creation of model input table
        MAX_BATCH = 100   # limit the batch size
        PARALLEL_CNT = 2  # limit cpu usage
        PRE_APPROVAL_PARALLEL_CNT = 2  # limit cpu usage
        BATCH_SIZE = 10   # limit memory usage
        SHORT_TIMEOUT = True  # shorter timeout
        """
    elif Variable.get('environment') == 'prod':
        TABLE_SUBFIX = ''
        TRIGGER_PATH = 'UPLOAD/MLAAS2ODS'
except BaseException:
    print('"environment" Variable only exists on airflow')
# =============================
