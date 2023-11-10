import cx_Oracle
from typing import AnyStr
import pandas as pd
import os
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN,
    ExchangeName as EN)

path = 'A:\\数据\\'

file_dict = {
    "Industry_Index": '行业指数标识.csv',
    "Industry_Size": '规模指数标识.csv',
    "Daily_stock_market": 'AStockData_new.csv',
    "Stock_connect_index": "陆股通指数数据.csv",
    "N_Capital_position": "北上资金沪股通持仓明细.csv"
}


class SQL(object):
    con_wind = cx_Oracle.connect('ZD_RESEARCH_WIND/zdresearchwind1234$@47.112.235.108:1521/zd_research')

    FS = {"BST": "ASHAREBALANCESHEET",
          "IST": "ASHAREINCOME",
          "CFT": "ASHARECASHFLOW"}

    def __init__(self):
        pass

    def finance_SQL(self,
                    sql_keys,
                    date_sta: int,
                    date_end: int,
                    table_type: str,
                    ):
        """

        :param sql_keys: {财务报表表名缩写：{财务内容关键字：缩写，}}，财务报表表名缩写：资产负债表[BST], 利润表[IST]，现金流量表[CFT]
        :param date_sta: 数据起始时间
        :param date_end: 数据截止时间
        :param table_type: 报表类型：合并报表，母公司报表/调成，更改前
        :return: 返回Oracle SQL语句
        """
        sql_dict = {}
        sql_key_dict = {}

        for key_, values_ in sql_keys.items():

            if values_ == {}:
                continue

            sheet_name = self.FS[key_]
            keys_str = ""
            keys_str2 = ""
            for K_, ab in values_.items():
                keys_str += f", {K_} {ab}"
                keys_str2 += f", {key_}.{ab}"
            sql_key_dict[key_] = keys_str2

            sql_dict[key_] = self._sql_finance(keys_str, sheet_name, table_type, date_sta, date_end)

        if len(sql_dict) == 1:
            key_list = list(sql_dict.keys())
            return f"SELECT * FROM ({sql_dict[key_list[0]]})"

        elif len(sql_dict) == 2:
            key_list = list(sql_dict.keys())
            return (
                "SELECT {key_0}.* {key_value_1} "
                "FROM ({sql0}) {key_0} "
                "LEFT JOIN ({sql1}) {key_1} "
                "ON {key_0}.\"{code}\" = {key_1}.\"{code}\" "
                "AND {key_0}.\"{date1}\" = {key_1}.\"{date1}\" "
                "AND {key_0}.\"{date2}\" = {key_1}.\"{date2}\" "
                "AND {key_0}.\"type\" = {key_1}.\"type\" ".format(
                    code=KN.STOCK_ID.value,
                    date1=SN.ANN_DATE.value,
                    date2=SN.REPORT_DATE.value,
                    key_0=key_list[0],
                    key_1=key_list[1],
                    key_value_1=sql_key_dict[key_list[1]],
                    sql0=sql_dict[key_list[0]],
                    sql1=sql_dict[key_list[1]],
                )
            )

        elif len(sql_dict) == 3:
            key_list = list(sql_dict.keys())
            return (
                "SELECT {key_0}.* {key_value_1} {key_value_2} "
                "FROM ({sql0})  {key_0} "
                "LEFT JOIN ({sql1}) {key_1} "
                "ON {key_0}.\"{code}\" = {key_1}.\"{code}\" "
                "AND {key_0}.\"{date1}\" = {key_1}.\"{date1}\" "
                "AND {key_0}.\"{date2}\" = {key_1}.\"{date2}\" "
                "AND {key_0}.\"type\" = {key_1}.\"type\" "
                "LEFT JOIN ({sql2}) {key_2} "
                "ON {key_1}.\"code\" = {key_2}.\"{code}\" "
                "AND {key_1}.\"{date1}\" = {key_2}.\"{date1}\" "
                "AND {key_1}.\"{date2}\" = {key_2}.\"{date2}\" "
                "AND {key_1}.\"type\" = {key_2}.\"type\" ".format(
                    code=KN.STOCK_ID.value,
                    date1=SN.ANN_DATE.value,
                    date2=SN.REPORT_DATE.value,
                    key_0=key_list[0],
                    key_1=key_list[1],
                    key_2=key_list[2],
                    key_value_1=sql_key_dict[key_list[1]],
                    key_value_2=sql_key_dict[key_list[2]],
                    sql0=sql_dict[key_list[0]],
                    sql1=sql_dict[key_list[1]],
                    sql2=sql_dict[key_list[2]],
                )
            )

        else:
            print("SQL ERROR!")
            return ""

    def stock_index_SQL(self,
                        bm_index: str,
                        date_sta: str = '20130101',
                        date_end: str = '20200401'):
        return (
            "SELECT "
            "to_char(to_date(TRADE_DT ,'yyyy-MM-dd'), 'yyyy-MM-dd') \"{date}\" , "
            "S_DQ_CLOSE \"{close}\", S_DQ_OPEN \"{open}\", S_DQ_HIGH \"{high}\", S_DQ_LOW \"{low}\" "
            "FROM AINDEXEODPRICES "
            "WHERE S_INFO_WINDCODE = \'{bm_code}\' "
            "AND TRADE_DT BETWEEN {sta} AND {end} "
            "ORDER BY TRADE_DT ".format(
                date=KN.TRADE_DATE.value,
                close=PVN.CLOSE.value,
                open=PVN.OPEN.value,
                high=PVN.HIGH.value,
                low=PVN.LOW.value,
                bm_code=bm_index,
                sta=date_sta,
                end=date_end,
            )
        )

    def _sql_finance(self,
                     keys: str,
                     f_table: str,
                     table_type: str,
                     date_sta: int,
                     date_end: int):
        """
        :param keys: ,关键字 缩写, 关键字 缩写
        :param f_table: 财务报表名称
        :param table_type: 报表类型
        :param date_sta:数据起始时间
        :param date_end:数据截止时间
        :return: 返回Oracle sql语句
        """
        return (
            "SELECT "
            "S_INFO_WINDCODE \"{code}\", "
            "to_char(to_date(ANN_DT ,'yyyy-MM-dd'), 'yyyy-MM-dd') \"{date1}\", "
            "to_char(to_date(REPORT_PERIOD ,'yyyy-MM-dd'), 'yyyy-MM-dd') \"{date2}\", "
            "STATEMENT_TYPE \"type\" {keys} "
            "FROM {f_table} "
            "WHERE STATEMENT_TYPE = {table_type} "
            "AND REPORT_PERIOD BETWEEN {sta} AND {end} "
            "AND regexp_like(S_INFO_WINDCODE, '^[0-9]') "
            "ORDER BY \"{date1}\" ".format(
                code=KN.STOCK_ID.value,
                date1=SN.ANN_DATE.value,
                date2=SN.REPORT_DATE.value,
                keys=keys,
                f_table=f_table,
                table_type=table_type,
                sta=date_sta,
                end=date_end,
            )
        )

    # 交易日
    def trade_date_SQL(self,
                       date_sta: str = '20130101',
                       date_end: str = '20200401',
                       exchange: str = EN.SSE.value):

        return (
            "SELECT to_char(to_date(TRADE_DAYS ,'yyyy-MM-dd'), 'yyyy-MM-dd') \"{date}\" "
            "FROM ASHARECALENDAR "
            "WHERE S_INFO_EXCHMARKET = \'{exchange}\' "
            "AND TRADE_DAYS BETWEEN {sta} AND {end} "
            "ORDER BY TRADE_DAYS ".format(
                date=KN.TRADE_DATE.value,
                exchange=exchange,
                sta=date_sta,
                end=date_end,
            )
        )

    def trade_date_csv(self, file_path: str = FPN.Trade_Date.value, file_name: str = 'TradeDate.csv'):
        return pd.read_csv(os.path.join(file_path, file_name))

    def list_date_csv(self, file_path: str = FPN.List_Date.value, file_name: str = 'ListDate.csv'):
        return pd.read_csv(os.path.join(file_path, file_name))

    # 个股上市日期
    def list_date_SQL(self):
        return (
            "SELECT S_INFO_WINDCODE \"{code}\", "
            "to_char(to_date(S_INFO_LISTDATE ,'yyyy-MM-dd'), 'yyyy-MM-dd') \"{list_date}\" "
            "FROM ASHAREDESCRIPTION "
            "WHERE regexp_like(S_INFO_WINDCODE, '^[0-9]')".format(
                code=KN.STOCK_ID.value, list_date=KN.LIST_DATE.value
            )
        )

    def query(self, sql):
        """
        Oracle SQL 查询
        :param sql:
        :return:
        """
        return pd.read_sql(sql, self.con_wind)
