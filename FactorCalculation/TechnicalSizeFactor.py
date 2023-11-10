import pandas as pd
import numpy as np
import sys

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo

from constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN
)


class TechnicalSizeFactor(FactorBase):

    @classmethod
    def Size001(cls,
                data: pd.DataFrame,
                liq_mv: str = PVN.LIQ_MV.value):
        """
        流动市值
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data.sort_index(inplace=True)

        data[func_name] = data[liq_mv]
        return data[['code', func_name]]

    def total_market_value(self,
                           data: pd.DataFrame,
                           market_name: str = 'mv'):
        """
        总市值对数
        :return:
        """
        factor_name = sys._getframe().f_code.co_name
        data[factor_name] = np.log(data[market_name])
        return data[['code', factor_name]]

    @classmethod
    def Size001_data_raw(cls,
                         sta: int = 20130101,
                         end: int = 20200401):
        return cls()._csv_data(data_name=[PVN.LIQ_MV.value])


