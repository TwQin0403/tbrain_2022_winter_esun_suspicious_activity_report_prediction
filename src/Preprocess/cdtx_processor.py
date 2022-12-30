"""
cdtx processor

register: 註冊函數器
get_basic_info: 讀取basic_info
compute_templete_feats: agg的基本設定
get_cdtx_type_features: 註冊函數 計算每天的templete_feat的特徵
cdtx_processor: 定義porcessor class
"""
from functools import reduce
import pandas as pd
import src.dataloader as dataloader

LOADER = dataloader.DataLoader()
USE_FUNCTIONS = []


def register(feat_name, active=True):
    """
    註冊函數器
    Args:
        feat_name (str): 要註冊函數
        active (bool, optional): 是否註冊. Defaults to True.
    """
    def decorate(func):
        print("REGISTER FUNCTION: {}".format(feat_name))
        if active:
            USE_FUNCTIONS.append(func)
        return func

    return decorate


def get_basic_info(loader=LOADER):
    """
    讀取basic_info
    Args:
        loader (dataloader.DataLoader, optional):  Defaults to LOADER.

    Returns:
        basic_info (pd.DataFrame)
    """
    basic_info = loader.load_input('basic_info.joblib')
    basic_info = basic_info[['alert_key', 'date', 'cust_id', 'sar_flag']]
    return basic_info


def compute_templete_feats(cdtx, cdtx_type):
    """
    agg的基本設定
    Args:
        cdtx (pd.DataFrame)
        cdtx_type (str)

    Returns:
        cdtx (pd.DataFrame)
    """
    cdtx = cdtx.groupby(['cust_id', 'date']).agg({
        "amt": ['count', 'sum', 'median'],
    }).reset_index()
    cols = [
        'cust_id',
        'date',
        'cdtx_amt_count',
        'cdtx_amt_sum',
        'cdtx_amt_median',
    ]
    cols = ['cust_id', 'date'
            ] + [col + "_{}".format(cdtx_type) for col in cols[2:]]
    cdtx.columns = cols
    return cdtx


@register(feat_name='cdtx_type_features')
def get_cdtx_type_features():
    """
    註冊函數 計算每天的templete_feat的特徵
    Returns:
        cdtx (pd.DataFrame)
    """
    cdtx = LOADER.load_train_data('cdtx.csv')
    tx_types = cdtx[['cust_id', 'country',
                     'cur_type']].groupby(['country',
                                           'cur_type']).count().index
    cdtx_list = []
    for tx_type in tx_types:
        cdtx_type = "country_{}_cur_{}".format(tx_type[0], tx_type[1])
        cdtx_split = cdtx[(cdtx['country'] == tx_type[0])
                          & (cdtx['cur_type'] == tx_type[1])].copy()
        cdtx_list.append(compute_templete_feats(cdtx_split, cdtx_type))
    cdtx = reduce(
        lambda df1, df2: pd.merge(
            df1, df2, on=['cust_id', 'date'], how='outer'), cdtx_list)
    return cdtx


class cdtx_processor:
    """定義porcessor class
    method
    get_feats: 計算特徵的method 將註冊函數一個個執行
    """
    def __init__(self):
        self.name = 'cdtx_type'

    def get_feats(self):
        """
        計算特徵的method 將註冊函數一個個執行
        Returns:
            cdtx (pd.DataFrame)
        """
        df_list = [func() for func in USE_FUNCTIONS]
        cdtx = reduce(
            lambda df1, df2: pd.merge(
                df1, df2, on=['cust_id', 'date'], how='outer'), df_list)
        cdtx = cdtx.fillna(0)
        cdtx = cdtx.sort_values('date').reset_index(drop=True)
        return cdtx
