import pandas as pd
import numpy as np
import sys


class LiquidationFactor(object):
    """
    流动性因子
    """

    def turnover(self,
                 data: pd.DataFrame,
                 amount_name: str = 'amount',
                 mv_name: str = 'mv',
                 n: int = 1):
        """
        N日换手率
        :return:
        """
        data[f'amount_{n}'] = data[amount_name].rolling(n).sum()
        data[f'mv_{n}'] = data[mv_name].rolling(n).mean()
        data[f'turnover_{n}'] = data[f'amount_{n}'] / data[f'mv_{n}']

        return self.data_filter1(
            data[['code', f'turnover_{n}']], rolling=n, factor_name=f'turnover_{n}'
        )


