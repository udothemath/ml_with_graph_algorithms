"""
This is an abstract class where pre-approval processing functions
for each customer are defined.

Project members should inherent this class as CustLogic class so that
 it can be recognized and connected by
 etl_framework.py

TODO:
    - [ ] 確保 input_table 的 顧客id放在index欄位。
    - [ ] 測試待完整邏輯完成後再進行，使用小包顧客資料作為測資，驗證計算時間。

如何測試: pytest --durations=0
"""
from src.process.common_logic import CommonLogic
import math
import numpy as np


class CustLogic(CommonLogic):
    """預審計算邏輯"""

    def calculate_cust_type_rate(self, pd_grade, lgd, travel_card, five_profession_card,
                                 world_card, wm_cust, gov_employee,
                                 military_police_firefighters,
                                 salary_ind):
        """
        預期利潤率：依客群給予利潤率
            1: 優質客群=>1.6%
            1.1 PD6~8額外加碼0.98%、PD9~PD11額外加碼1.88%、PD12以上額外加碼2.78%；
            1.2 LGD 0.51~0.531額外加碼0.5%、LGD 0.532以上額外加碼1%

            2: 固定薪客群=>2%
            2.1 PD3~5額外加碼0.98%、PD6~PD8額外加碼1.88%、PD9以上額外加碼2.78%；
            2.2 LGD 0.51~0.531額外加碼0.5%、LGD 0.532以上額外加碼1%

            3: 非固定薪、企業主=>4.7%
            3.1 PD6~8額外加碼0.98%、PD9~PD11額外加碼1.88%、PD12以上額外加碼2.78%；
            3.2 LGD 0.51~0.531額外加碼0.5%、LGD 0.532以上額外加碼1%


        Args:
            判斷客群: 1(優質客群)/2(固定薪)/3(非固定薪、企業主)
                travel_card(str): 是否有國旅卡
                five_profession_card(str): 是否有五師卡
                world_card(str): 是否有世界卡
                wm_cust(str): 是否為理財會員
                gov_employee(str): 是否行業別為公務人員
                military_police_firefighters(str): 是否行業別為軍警消
                salary_ind(str): 是否為本行薪轉戶
            判斷利率加碼:
                pd_grade(int)
                lgd(float)

        Returns(float):
            output: 客群預期利潤率
        """
        cust_type = self.create_cust_type(travel_card, five_profession_card,
                                          world_card, wm_cust, gov_employee,
                                          military_police_firefighters,
                                          salary_ind)

        lgd = round(lgd, 3)
        if pd_grade == -1 or lgd == -1:
            output = np.NaN

        # 優質客群
        elif cust_type == 1 and pd_grade in (1, 2, 3, 4, 5) and lgd in (0.441, 0.458, 0.462):
            output = 0.016
        elif cust_type == 1 and pd_grade in (1, 2, 3, 4, 5) and lgd in (0.513, 0.530):
            output = 0.021
        elif cust_type == 1 and pd_grade in (1, 2, 3, 4, 5) and lgd in (0.535, 0.598, 0.652):
            output = 0.026
        elif cust_type == 1 and pd_grade in (6, 7, 8) and lgd in (0.441, 0.458, 0.462):
            output = 0.0258
        elif cust_type == 1 and pd_grade in (6, 7, 8) and lgd in (0.513, 0.530):
            output = 0.0308
        elif cust_type == 1 and pd_grade in (6, 7, 8) and lgd in (0.535, 0.598, 0.652):
            output = 0.0358
        elif cust_type == 1 and pd_grade in (9, 10, 11) and lgd in (0.441, 0.458, 0.462):
            output = 0.0348
        elif cust_type == 1 and pd_grade in (9, 10, 11) and lgd in (0.513, 0.530):
            output = 0.0398
        elif cust_type == 1 and pd_grade in (9, 10, 11) and lgd in (0.535, 0.598, 0.652):
            output = 0.0448
        elif cust_type == 1 and pd_grade in (12, 13, 14, 15) and lgd in (0.441, 0.458, 0.462):
            output = 0.0438
        elif cust_type == 1 and pd_grade in (12, 13, 14, 15) and lgd in (0.513, 0.530):
            output = 0.0488
        elif cust_type == 1 and pd_grade in (12, 13, 14, 15) and lgd in (0.535, 0.598, 0.652):
            output = 0.0538

        # 固定薪客群
        elif cust_type == 2 and pd_grade in (1, 2) and lgd in (0.441, 0.458, 0.462):
            output = 0.020
        elif cust_type == 2 and pd_grade in (1, 2) and lgd in (0.513, 0.530):
            output = 0.025
        elif cust_type == 2 and pd_grade in (1, 2) and lgd in (0.535, 0.598, 0.652):
            output = 0.030
        elif cust_type == 2 and pd_grade in (3, 4, 5) and lgd in (0.441, 0.458, 0.462):
            output = 0.0298
        elif cust_type == 2 and pd_grade in (3, 4, 5) and lgd in (0.513, 0.530):
            output = 0.0348
        elif cust_type == 2 and pd_grade in (3, 4, 5) and lgd in (0.535, 0.598, 0.652):
            output = 0.0398
        elif cust_type == 2 and pd_grade in (6, 7, 8) and lgd in (0.441, 0.458, 0.462):
            output = 0.0388
        elif cust_type == 2 and pd_grade in (6, 7, 8) and lgd in (0.513, 0.530):
            output = 0.0438
        elif cust_type == 2 and pd_grade in (6, 7, 8) and lgd in (0.535, 0.598, 0.652):
            output = 0.0488
        elif cust_type == 2 and pd_grade in (9, 10, 11, 12, 13, 14, 15) and lgd in (0.441, 0.458, 0.462):
            output = 0.0478
        elif cust_type == 2 and pd_grade in (9, 10, 11, 12, 13, 14, 15) and lgd in (0.513, 0.530):
            output = 0.0528
        elif cust_type == 2 and pd_grade in (9, 10, 11, 12, 13, 14, 15) and lgd in (0.535, 0.598, 0.652):
            output = 0.0578

        # 非固定薪/企業主客群
        elif cust_type == 3 and pd_grade in (1, 2, 3, 4, 5) and lgd in (0.441, 0.458, 0.462):
            output = 0.047
        elif cust_type == 3 and pd_grade in (1, 2, 3, 4, 5) and lgd in (0.513, 0.530):
            output = 0.052
        elif cust_type == 3 and pd_grade in (1, 2, 3, 4, 5) and lgd in (0.535, 0.598, 0.652):
            output = 0.057
        elif cust_type == 3 and pd_grade in (6, 7, 8) and lgd in (0.441, 0.458, 0.462):
            output = 0.0568
        elif cust_type == 3 and pd_grade in (6, 7, 8) and lgd in (0.513, 0.530):
            output = 0.0618
        elif cust_type == 3 and pd_grade in (6, 7, 8) and lgd in (0.535, 0.598, 0.652):
            output = 0.0668
        elif cust_type == 3 and pd_grade in (9, 10, 11) and lgd in (0.441, 0.458, 0.462):
            output = 0.0658
        elif cust_type == 3 and pd_grade in (9, 10, 11) and lgd in (0.513, 0.530):
            output = 0.0708
        elif cust_type == 3 and pd_grade in (9, 10, 11) and lgd in (0.535, 0.598, 0.652):
            output = 0.0758
        elif cust_type == 3 and pd_grade in (12, 13, 14, 15) and lgd in (0.441, 0.458, 0.462):
            output = 0.0748
        elif cust_type == 3 and pd_grade in (12, 13, 14, 15) and lgd in (0.513, 0.530):
            output = 0.0798
        elif cust_type == 3 and pd_grade in (12, 13, 14, 15) and lgd in (0.535, 0.598, 0.652):
            output = 0.0848
        else:
            output = np.NaN
        return output

    def calculate_fee(self, travel_card, five_profession_card, world_card,
                      wm_cust, gov_employee, military_police_firefighters,
                      salary_ind):
        """
        判斷優質、固定薪、非固定薪及企業主對應之手續費
            1.優質客群: 5000
            2.固定薪: 6000
            3.非固定薪、企業主: 6000

        Args:
            判斷客群: 1(優質客群)/2(固定薪)/3(非固定薪、企業主)
                travel_card(str): 是否有國旅卡
                five_profession_card(str): 是否有五師卡
                world_card(str): 是否有世界卡
                wm_cust(str): 是否為理財會員
                gov_employee(str): 是否行業別為公務人員
                military_police_firefighters(str): 是否行業別為軍警消
                salary_ind(str): 是否為本行薪轉戶
        Returns(int):
            fee: 手續費
        """
        cust_type = self.create_cust_type(travel_card, five_profession_card,
                                          world_card, wm_cust, gov_employee,
                                          military_police_firefighters,
                                          salary_ind)
        if cust_type == 1:
            fee = 5000
        else:
            fee = 6000
        return fee

    def calculate_first_level_rate(self, travel_card, five_profession_card,
                                   world_card, wm_cust, gov_employee,
                                   military_police_firefighters, salary_ind):
        """
        判斷優質、固定薪、非固定薪及企業主對應之第一段利率
            1.優質客群: 1.38%
            2.固定薪: 1.68%
            3.非固定薪、企業主: 1.68%

        Args:
            判斷客群: 1(優質客群)/2(固定薪)/3(非固定薪、企業主)
                travel_card(str): 是否有國旅卡
                five_profession_card(str): 是否有五師卡
                world_card(str): 是否有世界卡
                wm_cus(str)t: 是否為理財會員
                gov_employee(str): 是否行業別為公務人員
                military_police_firefighters(str): 是否行業別為軍警消
                salary_ind(str): 是否為本行薪轉戶

        Returns(float):
            first_level_rate: 第一段利率
        """
        cust_type = self.create_cust_type(travel_card, five_profession_card,
                                          world_card, wm_cust, gov_employee,
                                          military_police_firefighters,
                                          salary_ind)
        if cust_type == 1:
            first_level_rate = 0.0138
        else:
            first_level_rate = 0.0168
        return first_level_rate

    def calculate_amt(self, upl_amt, exist_monin, salary_monin, cc_monin,
                      pd_grade, lgd, travel_card, five_profession_card,
                      world_card, wm_cust, gov_employee,
                      military_police_firefighters, salary_ind):
        """
        計算額度：預估收入*額度倍數-行內無擔
            若預估收入小於24000或為空值則調整月收為24000
            若額度倍數為空值則調整額度倍數為20
            若計算出額度為負則回傳0
        Args:
            upl_amt(float): 行內無擔保額度
            計算預估收入:
                exist_monin(int): 舊貸戶月收
                salary_monin(int): 薪轉月收
                cc_monin(int): 核卡月收
            計算額度倍數:
                pd_grade(int)
                lgd(float)
                travel_card(str): 是否有國旅卡
                five_profession_card(str): 是否有五師卡
                world_card(str): 是否有世界卡
                wm_cust(str): 是否為理財會員
                gov_employee(str): 是否行業別為公務人員
                military_police_firefighters(str): 是否行業別為軍警消
                salary_ind(str): 是否為本行薪轉戶
        Returns(float):
            max_limit: 預審額度
        """

        monin = self.create_monin(exist_monin, salary_monin, cc_monin)
        factor = self.calculate_factor(pd_grade, lgd, travel_card,
                                       five_profession_card, world_card,
                                       wm_cust, gov_employee,
                                       military_police_firefighters,
                                       salary_ind)
        if math.isnan(monin) or monin < 24000:
            monin = 24000
        if math.isnan(factor):
            factor = 20
        max_limit = monin * factor - upl_amt
        if max_limit > 3000000:
            max_limit = 3000000
        elif max_limit < 50000:
            max_limit = 0
        return max(0, max_limit)

    def calculate_total_rate(self, travel_card, five_profession_card,
                             world_card, wm_cust, gov_employee,
                             military_police_firefighters,
                             salary_ind, pd_value, lgd, pd_grade):
        """
        計算定價利率: cust_type_rate + base_rate
        cust_type_rate: 客群預期利潤率
        base_rate: 基礎利率
        Args:
            計算客群預期利潤率:
                pd_grade：pd等級
                lgd
                travel_card(str): 是否有國旅卡
                five_profession_card(str): 是否有五師卡
                world_card(str): 是否有世界卡
                wm_cust(str): 是否為理財會員
                gov_employee(str): 是否行業別為公務人員
                military_police_firefighters(str): 是否行業別為軍警消
                salary_ind(str): 是否為本行薪轉戶
            計算基礎利率:
                pd_value(float)
                lgd(float)
        Returns(float):
            total_rate: 定價利率
        """
        cust_type_rate = self.calculate_cust_type_rate(pd_grade, lgd, travel_card,
                                                       five_profession_card,
                                                       world_card, wm_cust,
                                                       gov_employee,
                                                       military_police_firefighters,
                                                       salary_ind)
        base_rate = self.calculate_base_rate(pd_value, lgd)

        if math.isnan(cust_type_rate) or math.isnan(base_rate):
            total_rate = 0.0618
        else:
            total_rate = cust_type_rate + base_rate
        return total_rate

    def run_all(self, with_cc, krm040_ind, bam087_ind, krm001_ind, jas002_ind,
                exist_monin, cc_monin, salary_monin, upl_amt,
                travel_card, five_profession_card, world_card, wm_cust,
                gov_employee, military_police_firefighters,
                salary_ind, pd_value, pd_grade, lgd):
        """
        輸入顧客因子、串接邏輯、輸出最終預審結果

        Args:
            with_cc(str): 是否為卡友
            krm040_ind(str): 是否有KRM040
            bam087_ind(str): 是否有BAM087
            krm001_ind(str): 是否有KRM001
            jas002_ind(str): 是否有JAS002
            exist_monin(int): 舊貸月收入資料
            cc_monin(int): 核卡月收入資料
            salary_monin(int): 薪轉戶月收入資料
            upl_amt(float): 行內無擔保額度
            travel_card(str): 是否有國旅卡
            five_profession_card(str): 是否有五師卡
            world_card(str): 是否有世界卡
            wm_cust(str): 是否為理財會員
            gov_employee(str): 是否行業別為公務人員
            military_police_firefighters(str): 是否行業別為軍警消
            salary_ind(str): 是否為本行薪轉戶
            pd_value(float): pd值
            pd_grade(int): pd等級
            lgd(float)

        Returns:
            group(str): 客群群組(01優質客群/02固定薪/03非固定薪、企業主)
            product(str): 適用產品別 (01一次撥付型 、02一次撥付+循環動用型)
            apdlv(float): 試算 APD
            lgd(float): 試算 LGD
            base_int(float): 基礎利率
            profit_int(float): 預期利潤率
            pre_net_income(int): 預估月收入
            max_limit(int): 最高可貸額度
            interest_rate_1(float): 第一段利率
            period_1(int): 第一段期數
            interest_rate_2(float): 第二段利率
            period_2(int): 第二段起始期數
            fee_amount(int): 預審費用
            all_rate(float): 總費用年百分率
        """
        assert isinstance(with_cc, str)
        assert isinstance(krm040_ind, str)
        assert isinstance(bam087_ind, str)
        assert isinstance(krm001_ind, str)
        assert isinstance(jas002_ind, str)
        assert isinstance(exist_monin, int) or isinstance(exist_monin, float)
        assert isinstance(cc_monin, int) or isinstance(cc_monin, float)
        assert isinstance(salary_monin, int) or isinstance(salary_monin, float)
        assert isinstance(upl_amt, int) or isinstance(upl_amt, float)
        assert isinstance(travel_card, str)
        assert isinstance(five_profession_card, str)
        assert isinstance(world_card, str)
        assert isinstance(wm_cust, str)
        assert isinstance(gov_employee, str)
        assert isinstance(military_police_firefighters, str)
        assert isinstance(salary_ind, str)
        assert isinstance(pd_value, int) or isinstance(pd_value, float)
        assert isinstance(pd_grade, int)
        assert isinstance(lgd, int) or isinstance(lgd, float)

        group = self.create_cust_type(travel_card, five_profession_card,
                                      world_card, wm_cust, gov_employee,
                                      military_police_firefighters,
                                      salary_ind)
        # 調整group格式為兩位文字
        group = str(group).zfill(2)
        product = self.create_pre_loan_type(with_cc, salary_ind, krm040_ind,
                                            bam087_ind, krm001_ind, jas002_ind)
        # 調整group格式為兩位文字
        product = str(product).zfill(2)
        # apdlv取至小數點第6位
        apdlv = round(pd_value, 6)
        # lgd取至小數點第3位
        lgd = round(lgd, 3)
        base_int = self.calculate_base_rate(pd_value, lgd)
        profit_int = self.calculate_cust_type_rate(pd_grade, lgd, travel_card,
                                                   five_profession_card,
                                                   world_card, wm_cust,
                                                   gov_employee,
                                                   military_police_firefighters,
                                                   salary_ind)
        pre_net_income = self.create_monin(exist_monin, salary_monin, cc_monin)
        max_limit = self.calculate_amt(upl_amt, exist_monin, salary_monin,
                                       cc_monin, pd_grade, lgd, travel_card,
                                       five_profession_card, world_card,
                                       wm_cust, gov_employee,
                                       military_police_firefighters,
                                       salary_ind)
        max_limit = int(max_limit)
        interest_rate_1 = self.calculate_first_level_rate(travel_card,
                                                          five_profession_card,
                                                          world_card, wm_cust,
                                                          gov_employee,
                                                          military_police_firefighters,
                                                          salary_ind)
        period_1 = 3
        interest_rate_2 = self.calculate_total_rate(travel_card,
                                                    five_profession_card,
                                                    world_card, wm_cust,
                                                    gov_employee,
                                                    military_police_firefighters,
                                                    salary_ind, pd_value, lgd, pd_grade)
        # interest_rate_2取至小數第四位
        round_interest_rate_2 = '{:.4f}'.format(round(interest_rate_2, 4))
        # interest_rate_2小數第四位調整為 8
        interest_rate_2 = float(str(round_interest_rate_2)[:-1] + '8')
        period_2 = 60 - period_1
        fee_amount = self.calculate_fee(travel_card, five_profession_card,
                                        world_card, wm_cust, gov_employee,
                                        military_police_firefighters,
                                        salary_ind)

        # 調整空值與型態
        if math.isnan(pre_net_income):
            pre_net_income = -1
        else:
            pre_net_income = int(pre_net_income)

        if math.isnan(max_limit):
            max_limit = -1
        else:
            max_limit = int(max_limit / 10000) * 10000

        # 確認預審輸出欄位的數值後再計算總費用年百分率
        all_rate = self.calculate_annual_rate(max_limit, period_1, period_2,
                                              interest_rate_1, interest_rate_2,
                                              fee_amount)

        return [group, product, apdlv, lgd, base_int, profit_int,
                pre_net_income, max_limit, interest_rate_1, period_1,
                interest_rate_2, period_2, fee_amount, all_rate]
