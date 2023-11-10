# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:48
# @Author: FC
# @Email:  18817289038@163.com
import pandas as pd
import numpy as np
import sys

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
    FinancialBalanceSheetName as FBSN,
    FinancialIncomeSheetName as FISN,
    FinancialCashFlowSheetName as FCFSN
)


# 估值因子
class FundamentalValueFactor(FactorBase):
    """408001000: 合并报表； 408006000：母公司报表 """

    def __init__(self):
        super(FundamentalValueFactor, self).__init__()

    @classmethod
    def Value001(cls,
                 data: pd.DataFrame,
                 net_profit_in: str = FISN.Net_Pro_In.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市盈率倒数(EP_TTM)：市盈率（包含少数股东权益）倒数
        :param data:
        :param net_profit_in:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value013(cls,
                 data: pd.DataFrame,
                 net_profit_in: str = FISN.Net_Pro_In.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市盈率倒数(最新财报)(EP_LR)：市盈率（包含少数股东权益）倒数
        :param data:
        :param net_profit_in:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_in] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value002(cls,
                 data: pd.DataFrame,
                 net_profit_cut: str = FISN.Net_Pro_Cut.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        扣非市盈率倒数(EP_cut_TTM): 市盈率（扣除非经常性损益）倒数
        :param data:
        :param net_profit_cut:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_cut] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value012(cls,
                 data: pd.DataFrame,
                 net_profit_ex: str = FISN.Net_Pro_Ex.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市盈率倒数(归属母公司净利润)(E2P_TTM)：市盈率（不包含少数股东权益）倒数
        :param data:
        :param net_profit_ex:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_ex] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value010(cls,
                 data: pd.DataFrame,
                 net_profit_ex: str = FISN.Net_Pro_Ex.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市盈率增长倒数(PEG_TTM)：市盈率（考虑利润的增长）倒数
        :param data:
        :param net_profit_ex:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_profit_ex] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value003(cls,
                 data: pd.DataFrame,
                 net_asset_ex: str = FBSN.Net_Asset_Ex.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False) -> FactorInfo:
        """
        市净率倒数(最新财报)(BP_LR)：市净率的倒数
        :param data:
        :param net_asset_ex:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_asset_ex] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value011(cls,
                 data: pd.DataFrame,
                 net_asset_ex: str = FBSN.Net_Asset_Ex.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False) -> FactorInfo:
        """
        市净率倒数(TTM)(BP_TTM)：市净率的倒数
        :param data:
        :param net_asset_ex:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_asset_ex] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value004(cls,
                 data: pd.DataFrame,
                 operator_income: str = FISN.Op_Income.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市销率倒数(TTM)(SP_TTM)：市销率倒数
        :param data:
        :param operator_income:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_income] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value015(cls,
                 data: pd.DataFrame,
                 operator_income: str = FISN.Op_Income.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市销率倒数(最新财报)(SP_LR)：市销率倒数
        :param data:
        :param operator_income:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_income] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value005(cls,
                 data: pd.DataFrame,
                 net_cash_flow: str = FCFSN.Net_CF.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市现率倒数(TTM)(NCFP_TTM)：市现率倒数（净现金流）
        :param data:
        :param net_cash_flow:净现金流
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[net_cash_flow] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value006(cls,
                 data: pd.DataFrame,
                 operator_net_cash_flow: str = FCFSN.Op_Net_CF.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市现率倒数(经营现金流，TTM)(OCFP_TTM)：市现率倒数（经营现金流）
        :param data:
        :param operator_net_cash_flow:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_net_cash_flow] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value007(cls,
                 data: pd.DataFrame,
                 free_cash_flow: str = FCFSN.Free_Cash_Flow.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市现率倒数(自由现金流，最新财报)(FCFP_LR)：市现率倒数（自由现金流）
        :param data:
        :param free_cash_flow:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[free_cash_flow] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value014(cls,
                 data: pd.DataFrame,
                 free_cash_flow: str = FCFSN.Free_Cash_Flow.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        市现率倒数(自由现金流，TTM)(FCFP_TTM)：市现率倒数（自由现金流）
        :param data:
        :param free_cash_flow:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[free_cash_flow] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value008(cls,
                 data: pd.DataFrame,
                 Surplus_Reserves: str = FBSN.Surplus_Reserves.value,
                 Undistributed_Profit: str = FBSN.Undistributed_Profit.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        股息率倒数(DP_TTM)：股息率（近12个月现金红利和）
        股息 = 期末留存收益 - 期初留存收益
        留存收益 = 盈余公积 + 未分配利润
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data["RE"] = data[Surplus_Reserves] + data[Undistributed_Profit]
        data[func_name] = data["RE"] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    @classmethod
    def Value009(cls,
                 data: pd.DataFrame,
                 operator_income: str = FISN.Op_Income.value,
                 total_mv: str = PVN.TOTAL_MV.value,
                 switch: bool = False):
        """
        企业价值倍数倒数(最新财报，扣除现金)(EV2EBITDA_LR)：企业价值（扣除现金）/息税折旧摊销前利润
        企业价值 = 总市值 + 负债总计 - 无息负债 - 货币资金
        :param data:
        :param operator_income:
        :param total_mv:
        :param switch:
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[operator_income] / data[total_mv]
        data_fact = data[func_name].copy(deep=True)
        data = data.reset_index()

        F = FactorInfo()
        F.data_raw = data[[SN.ANN_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, func_name]]
        F.data = data_fact
        F.factor_type = data['type'][0]
        F.factor_category = cls().__class__.__name__
        F.factor_name = func_name

        return F

    ####################################################################################################################
    @classmethod
    def Value003_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value011_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FBSN.Net_Asset_Ex.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FBSN.Net_Asset_Ex.value] = financial_ttm

        # switch freq
        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')
        res.reset_index(inplace=True)

        return res

    @classmethod  # TODO
    def REP_data_raw(cls,
                     sta: int = 20130101,
                     end: int = 20200401,
                     f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FBSN.Net_Asset_Ex.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FBSN.Net_Asset_Ex.value] = financial_ttm

        # switch freq
        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')
        res.reset_index(inplace=True)

        return res

    @classmethod  # TODO
    def CCP_data_raw(cls,
                     sta: int = 20130101,
                     end: int = 20200401,
                     f_type: str = '408001000'):
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FBSN.Net_Asset_Ex.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FBSN.Net_Asset_Ex.value] = financial_ttm

        # switch freq
        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')
        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value012_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_EXCL_MIN_INT_INC": f"\"{FISN.Net_Pro_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FISN.Net_Pro_Ex.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_Ex.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_Ex.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value001_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FISN.Net_Pro_In.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_In.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_In.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value013_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_INCL_MIN_INT_INC": f"\"{FISN.Net_Pro_In.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_In.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value010_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_EXCL_MIN_INT_INC": f"\"{FISN.Net_Pro_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FISN.Net_Pro_Ex.value)
        # 利润同比增长率
        profit_growth = financial_ttm.groupby(KN.STOCK_ID.value).apply(lambda x: x / x.shift(4) - 1)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_Ex.value] = financial_ttm * profit_growth

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_Ex.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value002_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"NET_PROFIT_AFTER_DED_NR_LP": f"\"{FISN.Net_Pro_Cut.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FISN.Net_Pro_Cut.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Net_Pro_Cut.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Net_Pro_Cut.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value007_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"CFT": {"FREE_CASH_FLOW": f"\"{FCFSN.Free_Cash_Flow.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data = cls()._switch_freq(data_=financial_data, name=FCFSN.Free_Cash_Flow.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value014_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"CFT": {"FREE_CASH_FLOW": f"\"{FCFSN.Free_Cash_Flow.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FCFSN.Free_Cash_Flow.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.Free_Cash_Flow.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FCFSN.Free_Cash_Flow.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value005_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"CFT": {"NET_INCR_CASH_CASH_EQU": f"\"{FCFSN.Net_CF.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FCFSN.Net_CF.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.Net_CF.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FCFSN.Net_CF.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value006_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"CFT": {"NET_CASH_FLOWS_OPER_ACT": f"\"{FCFSN.Op_Net_CF.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FCFSN.Op_Net_CF.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FCFSN.Op_Net_CF.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FCFSN.Op_Net_CF.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value004_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FISN.Op_Income.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Op_Income.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value015_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Op_Income.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value008_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"BST": {"SURPLUS_RSRV": f"\"{FBSN.Surplus_Reserves.value}\"",
                            "UNDISTRIBUTED_PROFIT": f"\"{FBSN.Undistributed_Profit.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        # TTM
        surplus_reserves = cls()._switch_ttm(financial_data, FBSN.Surplus_Reserves.value)
        undistributed_profit = cls()._switch_ttm(financial_data, FBSN.Undistributed_Profit.value)
        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FBSN.Surplus_Reserves.value] = surplus_reserves
        financial_data[FBSN.Undistributed_Profit.value] = undistributed_profit

        Sur_Res = cls()._switch_freq(data_=financial_data, name=FBSN.Surplus_Reserves.value, limit=120)
        Undis_Pro = cls()._switch_freq(data_=financial_data, name=FBSN.Undistributed_Profit.value, limit=120)
        financial_data_new = pd.merge(Sur_Res[[FBSN.Surplus_Reserves.value, SN.REPORT_DATE.value, 'type']],
                                      Undis_Pro[[FBSN.Undistributed_Profit.value, SN.REPORT_DATE.value, 'type']],
                                      on=[KN.TRADE_DATE.value, KN.STOCK_ID.value, SN.REPORT_DATE.value, 'type'],
                                      how='inner')
        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data_new, price_data], axis=1, join='inner')
        res = res[res.columns.drop_duplicates()]

        res.reset_index(inplace=True)

        return res

    @classmethod
    def Value009_data_raw(cls,
                          sta: int = 20130101,
                          end: int = 20200401,
                          f_type: str = '408001000'):
        sql_keys = {"IST": {"OPER_REV": f"\"{FISN.Op_Income.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)
        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value])

        # 过滤未上市公司
        data_ = pd.merge(financial_data, cls().list_date, on=[KN.STOCK_ID.value], how='left')
        financial_data = data_[data_[KN.TRADE_DATE.value] >= data_[KN.LIST_DATE.value]]

        financial_ttm = cls()._switch_ttm(financial_data, FISN.Op_Income.value)

        financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        financial_data[FISN.Op_Income.value] = financial_ttm

        financial_data = cls()._switch_freq(data_=financial_data, name=FISN.Op_Income.value, limit=120)

        price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

        # 数据合并
        res = pd.concat([financial_data, price_data], axis=1, join='inner')

        res.reset_index(inplace=True)

        return res


