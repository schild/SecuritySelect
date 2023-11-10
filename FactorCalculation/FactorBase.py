# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:49
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import os
import copy
from typing import Callable, Dict, Any

from Data.GetData import SQL
from constant import (
    KeyName as KN,
    SpecialName as SN,
    FilePathName as FPN,
    ExchangeName as EN
)


class FactorBase(object):

    def __init__(self):
        self.Q = SQL()
        self.list_date = SQL().list_date_csv()

    # 财务数据转换，需要考虑未来数据
    def _switch_freq(self,
                     data_: pd.DataFrame,
                     name: str,
                     limit: int = 120,
                     date_sta: str = '20130101',
                     date_end: str = '20200401',
                     exchange: str = EN.SSE.value) -> pd.Series:
        """

        :param data_:
        :param name: 需要转换的财务指标
        :param limit: 最大填充时期，默认二个季度
        :param date_sta:
        :param date_end:
        :param exchange:
        :return:
        """

        def _reindex(data: pd.DataFrame, name_: str):
            """填充有风险哦"""
            # data_re = data.reindex(trade_date[KN.TRADE_DATE.value])
            data_re = pd.merge(data, trade_date, on=KN.TRADE_DATE.value, how='outer')
            data_re.loc[:, data_re.columns != name_] = data_re.loc[:, data_re.columns != name_].fillna(method='ffill')

            return data_re

        sql_trade_date = self.Q.trade_date_SQL(date_sta=date_sta,
                                               date_end=date_end,
                                               exchange=exchange)
        trade_date = self.Q.query(sql_trade_date)

        # 保留最新数据
        data_sub = data_.groupby(KN.STOCK_ID.value,
                                 group_keys=False).apply(
            lambda x: x.sort_values(
                by=[KN.TRADE_DATE.value, SN.REPORT_DATE.value]).drop_duplicates(subset=[KN.TRADE_DATE.value],
                                                                                keep='last'))
        data_sub.reset_index(inplace=True)

        # 交易日填充
        data_trade_date = data_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(_reindex, name)
        res = data_trade_date.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value]).sort_index()

        # 历史数据有限填充因子值
        res[name] = res[name].groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x.ffill(limit=limit))

        res.dropna(subset=[name], inplace=True)
        if 'index' in res.columns:
            res.drop(columns='index', inplace=True)
        return res

    # 读取因子计算所需常用数据
    def _csv_data(self,
                  data_name: list,
                  file_path: str = FPN.factor_inputData.value,
                  file_name: str = "FactorPool1",
                  date: str = KN.TRADE_DATE.value,
                  stock_id: str = KN.STOCK_ID.value):
        return pd.read_csv(
            os.path.join(file_path, f'{file_name}.csv'),
            usecols=[date, stock_id] + data_name,
        )

    # 读取指数数据
    def csv_index(self,
                  data_name: list,
                  file_path: str = FPN.factor_inputData.value,
                  file_name: str = 'IndexInfo',
                  index_name: str = '',
                  date: str = KN.TRADE_DATE.value,):
        index_data = pd.read_csv(
            os.path.join(file_path, f'{file_name}.csv'),
            usecols=[date, 'index_name'] + data_name,
        )
        return index_data[index_data['index_name'] == index_name]

    # 读取分钟数据(数据不在一个文件夹中)，返回回调函数结果
    def csv_HFD_data(self,
                     data_name: list,
                     func: Callable = None,
                     fun_kwargs: dict = {},
                     file_path: str = FPN.HFD_Stock_M.value,
                     sub_file: str = '') -> Dict[str, Any]:
        if not sub_file:
            Path = file_path
        elif sub_file == '1minute':
            Path = FPN.HFD_Stock_M.value
        else:
            Path = os.path.join(file_path, sub_file)
        data_dict = {}
        file_names = os.listdir(Path)

        i = 1
        for file_name in file_names:
            i += 1
            if file_name[-3:] == 'csv':
                try:
                    data_df = pd.read_csv(os.path.join(Path, file_name), usecols=['code', 'time'] + data_name)
                except Exception as e:
                    continue
                data_df['date'] = file_name[:-4]
                data_df.rename(columns={'code': 'stock_id'}, inplace=True)
                res = func(data_df, **fun_kwargs)
                data_dict[file_name[:-4]] = res
            # if i == 3:
            #     break

        return data_dict

    def _switch_ttm(self, data_: pd.DataFrame, name: str):
        """
        计算TTM，groupby后要排序
        """

        def _pros_ttm(data_sub: pd.DataFrame, name_: str):
            data_sub[f'{name_}_TTM'] = data_sub[name_].diff(1)
            res_ = data_sub[data_sub['M'] == '03'][name_].append(
                data_sub[data_sub['M'] != '03'][f'{name_}_TTM']
            )
            res_ = res_.droplevel(level=KN.STOCK_ID.value).sort_index().rolling(4).sum()
            return res_

        data_copy = copy.deepcopy(data_)
        data_copy['M'] = data_copy[SN.REPORT_DATE.value].apply(lambda x: x[5:7])
        data_copy.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value], inplace=True)
        data_copy.sort_index(inplace=True)

        res = data_copy[[name, 'M']].groupby(KN.STOCK_ID.value).apply(_pros_ttm, name)

        res.index = res.index.swaplevel(0, 1)
        res.name = name
        return res


if __name__ == '__main__':
    A = FactorBase()
    # A.csv_HFD_data()
