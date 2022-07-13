"""
預審計算共用方法
"""
from src.common.apply.logic_base import LogicBase
import math
import numpy as np
from scipy.stats import norm
import numpy_financial as npf


class CommonLogic(LogicBase):
    """預審計算邏輯"""

    def __init__(self, ftp_cost=0.0051, operation_cost=0.0077,
                 capital_adequacy_ratio=0.105, coe=0.1):
        """
        此處定義預審計算過程中的固定參數，如: 資金成本。
        ftp_cost: 資金成本
        operation_cost: 營運成本
        capital_adequacy_ratio: 資本適足率
        coe
        """
        self._ftp_cost = ftp_cost
        self._operation_cost = operation_cost
        self._capital_adequacy_ratio = capital_adequacy_ratio
        self._coe = coe

    @property
    def input_column_names(self):
        """
        input1 和 input2 要調整為預審輸入參數名稱。(建議參數命名與欄位名稱一致)

        with_cc: 是否為卡友
        krm040_ind: 是否有KRM040
        bam087_ind: 是否有BAM087
        krm001_ind: 是否有KRM001
        jas002_ind: 是否有JAS002
        exist_monin: 舊貸月收入資料
        cc_monin: 核卡月收入資料
        salary_monin: 薪轉戶月收入資料
        upl_amt: 行內無擔保額度
        travel_card: 是否有國旅卡
        five_profession_card: 是否有五師卡
        world_card: 是否有世界卡
        wm_cust: 是否為理財會員
        gov_employee: 是否行業別為公務人員
        military_police_firefighters: 是否行業別為軍警消
        salary_ind: 是否為本行薪轉戶
        pd_value: pd值
        pd_grade: pd等級
        lgd

        """

        return ['with_cc', 'krm040_ind', 'bam087_ind', 'krm001_ind',
                'jas002_ind', 'exist_monin', 'cc_monin',
                'salary_monin', 'upl_amt', 'travel_card',
                'five_profession_card', 'world_card', 'wm_cust',
                'gov_employee', 'military_police_firefighters',
                'salary_ind', 'pd_value', 'pd_grade', 'lgd']

    @property
    def output_column_names(self):
        """
        output1 和 output2 要調整為預審輸出參數名稱。(建議參數命名與欄位名稱一致)

        group
        product
        apdlv
        lgd
        base_int
        profit_int
        pre_net_income
        max_limit
        interest_rate_1
        period_1
        interest_rate_2
        period_2
        fee_amount
        all_rate
        list_name
        data_dt
        etl_dt
        """
        return ['group', 'product', 'apdlv', 'lgd', 'base_int', 'profit_int',
                'pre_net_income', 'max_limit', 'interest_rate_1', 'period_1',
                'interest_rate_2', 'period_2', 'fee_amount', 'all_rate']

    def create_pre_loan_type(self, with_cc, salary_ind, krm040_ind,
                             bam087_ind, krm001_ind, jas002_ind):
        """
        判斷顧客為一次撥付型及循環動用型或僅為一次撥付型
        Args:
            with_cc(str): 是否為卡友
            salary_ind(str): 是否為本行薪轉戶
            krm040_ind(str): 是否有KRM040
            bam087_ind(str): 是否有BAM087
            krm001_ind(str): 是否有KRM001
            jas002_ind(str): 是否有JAS002
        Returns(int):
            1: 一次撥付型
            2: 一次撥付型及循環動用型
        """
        if (with_cc == 'Y' and salary_ind == 'Y' and
            (krm040_ind == 'Y' and bam087_ind == 'Y' and
             krm001_ind == 'Y' and jas002_ind == 'N')):
            output = 2
        else:
            output = 1
        return output

    def create_cust_type(self, travel_card, five_profession_card, world_card,
                         wm_cust, gov_employee, military_police_firefighters,
                         salary_ind):
        """
        判斷客群為優質、固定薪、非固定薪及企業主
        Args:
            travel_card(str): 是否有國旅卡
            five_profession_card(str): 是否有五師卡
            world_card(str): 是否有世界卡
            wm_cust(str): 是否為理財會員
            gov_employee(str): 是否行業別為公務人員
            military_police_firefighters(str): 是否行業別為軍警消
            salary_ind(str): 是否為本行薪轉戶
        Returns(int):
            1: 優質客群
            2: 固定薪客群
            3: 非固定薪、企業主
        """
        if (travel_card == 'Y' or five_profession_card == 'Y' or
            world_card == 'Y' or wm_cust == 'Y' or
                gov_employee == 'Y'):
            output = 1
        elif military_police_firefighters == 'Y' or salary_ind == 'Y':
            output = 2
        else:
            output = 3
        return output

    def create_monin(self, exist_monin, salary_monin, cc_monin):
        """
        判斷顧客月收
        如顧客三個月收來源皆為空值，則回傳空值
        Args:
            exist_monin(int): 舊貸月收入資料
            salary_monin(int): 薪轉戶月收入資料
            cc_monin(int): 核卡月收入資料
        Returns(int):
            output: 月收入金額
        """
        def monin_change_type(monin):
            if not isinstance(monin, int):
                if isinstance(monin, float):
                    if math.isnan(monin):
                        monin = -1
                    else:
                        monin = int(monin)
                else:
                    monin = -1
            return monin

        exist_monin = monin_change_type(exist_monin)
        salary_monin = monin_change_type(salary_monin)
        cc_monin = monin_change_type(cc_monin)

        if (exist_monin == -1 and salary_monin == -1 and
                cc_monin == -1):
            output = -1
        elif salary_monin >= 24000:
            output = salary_monin
        elif exist_monin >= 24000:
            output = exist_monin
        elif cc_monin >= 24000:
            output = cc_monin
        else:
            output = max([salary_monin, exist_monin, cc_monin])
        return output

    def calculate_factor(self, pd_grade, lgd, travel_card,
                         five_profession_card, world_card,
                         wm_cust, gov_employee,
                         military_police_firefighters, salary_ind):
        """
        額度倍數：依客群對照PD等級、LGD查詢額度倍數
        若顧客pd_grade、lgd任一為-1則回傳倍數為空值
        Args:
            pd_grade(int)
            lgd(float)
            判斷客群: 1(優質客群)/2(固定薪)/3(非固定薪、企業主)
                travel_card(str): 是否有國旅卡
                five_profession_card(str): 是否有五師卡
                world_card(str): 是否有世界卡
                wm_cust(str): 是否為理財會員
                gov_employee(str): 是否行業別為公務人員
                military_police_firefighters(str): 是否行業別為軍警消
                salary_ind(str): 是否為本行薪轉戶
        Returns(float):
            output: 倍數
        """
        cust_type = self.create_cust_type(travel_card, five_profession_card,
                                          world_card, wm_cust, gov_employee,
                                          military_police_firefighters,
                                          salary_ind)
        lgd = round(lgd, 3)
        if pd_grade == -1 or lgd == -1:
            output = np.NaN

        # 優質客群
        elif cust_type == 1 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.441, 0.458, 0.462):
            output = 22
        elif cust_type == 1 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.513, 0.530, 0.535):
            output = 21
        elif cust_type == 1 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.598, 0.652):
            output = 20
        elif cust_type == 1 and pd_grade in (7, 8, 9) and lgd in (0.441, 0.458, 0.462):
            output = 19
        elif cust_type == 1 and pd_grade in (7, 8, 9) and lgd in (0.513, 0.530, 0.535):
            output = 18
        elif cust_type == 1 and pd_grade in (7, 8, 9) and lgd in (0.598, 0.652):
            output = 17
        elif cust_type == 1 and pd_grade in (10, 11, 12) and lgd in (0.441, 0.458, 0.462):
            output = 16
        elif cust_type == 1 and pd_grade in (10, 11, 12) and lgd in (0.513, 0.530, 0.535):
            output = 15
        elif cust_type == 1 and pd_grade in (10, 11, 12) and lgd in (0.598, 0.652):
            output = 14
        elif cust_type == 1 and pd_grade in (13, 14) and lgd in (0.441, 0.458, 0.462, 0.513, 0.530, 0.535):
            output = 12
        elif cust_type == 1 and ((pd_grade in (13, 14) and lgd in (0.598, 0.652)) or
                                 (pd_grade == 15 and lgd in (0.441, 0.458, 0.462, 0.513, 0.53, 0.535, 0.598, 0.652))):
            output = 10

        # 固定薪客群
        elif cust_type == 2 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.441, 0.458, 0.462):
            output = 21
        elif cust_type == 2 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.513, 0.530, 0.535):
            output = 20
        elif cust_type == 2 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.598, 0.652):
            output = 19
        elif cust_type == 2 and pd_grade in (7, 8, 9) and lgd in (0.441, 0.458, 0.462, 0.513, 0.530, 0.535):
            output = 18
        elif cust_type == 2 and pd_grade in (7, 8, 9) and lgd in (0.598, 0.652):
            output = 17
        elif cust_type == 2 and pd_grade in (10, 11, 12) and lgd in (0.441, 0.458, 0.462, 0.513, 0.530, 0.535):
            output = 15
        elif cust_type == 2 and pd_grade in (10, 11, 12) and lgd in (0.598, 0.652):
            output = 13
        elif cust_type == 2 and pd_grade in (13, 14, 15) and lgd in (0.441, 0.458, 0.462, 0.513, 0.530, 0.535):
            output = 10
        elif cust_type == 2 and pd_grade in (13, 14, 15) and lgd in (0.598, 0.652):
            output = 0

        # 非固定薪/企業主客群
        elif cust_type == 3 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.441, 0.458, 0.462, 0.513, 0.530, 0.535):
            output = 20
        elif cust_type == 3 and pd_grade in (1, 2, 3, 4, 5, 6) and lgd in (0.598, 0.652):
            output = 18
        elif cust_type == 3 and pd_grade in (7, 8, 9) and lgd in (0.441, 0.458, 0.462):
            output = 18
        elif cust_type == 3 and pd_grade in (7, 8, 9) and lgd in (0.513, 0.530, 0.535):
            output = 17
        elif cust_type == 3 and pd_grade in (7, 8, 9) and lgd in (0.598, 0.652):
            output = 16
        elif cust_type == 3 and pd_grade in (10, 11, 12) and lgd in (0.441, 0.458, 0.462):
            output = 15
        elif cust_type == 3 and pd_grade in (10, 11, 12) and lgd in (0.513, 0.530, 0.535):
            output = 13
        elif cust_type == 3 and pd_grade in (10, 11, 12) and lgd in (0.598, 0.652):
            output = 12
        elif cust_type == 3 and pd_grade == 13 and lgd in (0.441, 0.458, 0.462):
            output = 10
        else:
            output = 0
        return output

    def calculate_el_ratio(self, pd_value, lgd):
        """
        計算EL%(pd_value*lgd)
        若顧客pd、lgd任一為-1，則回傳EL%為空值

        Args:
            pd_value(float)
            lgd(float)

        Returns(float):
            output: EL%
        """
        if pd_value == -1 or lgd == -1:
            output = np.NaN
        else:
            output = pd_value * lgd
        return output

    def calculate_k(self, pd_value, lgd):
        """
        計算資本計提率(K值)
        若顧客pd_value、lgd任一為-1則回傳空值

        Args:
            pd_value(float)
            lgd(float)
        Returns(float):
            k: 資本計提率
        """
        if pd_value == -1 or lgd == -1:
            k = np.NaN
        else:
            r = 0.03 * (1 - math.exp(-35 * pd_value)) / (1 - math.exp(-35)) + \
                0.16 * (1 - (1 - math.exp(-35 * pd_value)) / (1 - math.exp(-35)))
            g_pd = norm.ppf(pd_value)
            g_999 = norm.ppf(0.999)
            x = (1 - r)**(-0.5) * g_pd + (r / (1 - r))**(0.5) * g_999
            n_x = norm.cdf(x)
            k = lgd * n_x - pd_value * lgd
        return k

    def calculate_capital_cost(self, pd_value, lgd):
        """
        計算資本成本: K*12.5*capital_adequacy_ratio(資本適足率)*COE
        Args:
            計算K:
                pd_value(float)
                lgd(float)
        Returns(float):
            capital_cost: 資本成本
        """
        k = self.calculate_k(pd_value, lgd)

        if math.isnan(k):
            capital_cost = np.NaN
        else:
            capital_cost = k * 12.5 * self._capital_adequacy_ratio * self._coe
        return capital_cost

    def calculate_base_rate(self, pd_value, lgd):
        """
        計算基礎利率: el_ratio * capital_cost * ftp_cost * operation_cost
        el_ratio: 預期風險損失成本
        capital_cost: 資本成本
        ftp_cost: 資金成本
        opration_cost: 營運成本
        Args:
            計算el_ratio、K:
                pd_value(float)
                lgd(float)
        Returns(float):
            base_rate: 基礎利率
        """
        el_ratio = self.calculate_el_ratio(pd_value, lgd)
        capital_cost = self.calculate_capital_cost(pd_value, lgd)

        if math.isnan(el_ratio) or math.isnan(capital_cost):
            base_rate = np.NaN
        else:
            base_rate = el_ratio + capital_cost + self._ftp_cost + self._operation_cost
        return base_rate

    def calculate_annual_rate(self, max_limit, period_1, period_2,
                              interest_rate_1, interest_rate_2, fee_amount):
        """
        計算總費用年百分率

        Args:
            max_limit(int): 最高可貸額度
            perild_1(int): 第一段期數
            period_2(int): 第二段期數
            interest_rate_1(float): 第一段利率
            interest_rate_2(float): 第二段利率
            fee_amount(int): 預審費用

        Returns:
            annual_rate(float): 總費用年百分率
        """
        period_total = int(period_1 + period_2)
        balance = [0] * period_total
        cap_int = [0] * period_total
        capital = [0] * period_total
        interest = [0] * period_total

        for i in range(period_total):
            term = i + 1
            if i == 0:
                balance[i] = max_limit
            else:
                balance[i] = balance[i - 1] - capital[i - 1]

            if term <= period_1:
                rate = interest_rate_1 / 12
            else:
                rate = interest_rate_2 / 12

            cap_int[i] = round(
                npf.pmt(rate, period_total + 1 - (i + 1), -balance[i]), 0)
            interest[i] = round(balance[i] * rate, 0)
            capital[i] = cap_int[i] - interest[i]

        actual_pay = max_limit - fee_amount
        cashflow = [0] * (period_total + 1)
        cashflow[0] = -actual_pay
        for i in range(period_total):
            cashflow[i + 1] = cap_int[i]

        annual_rate = npf.irr(cashflow) * 12

        return annual_rate

    def run_all(self):
        """
        must override the abstract method
        """
        pass
