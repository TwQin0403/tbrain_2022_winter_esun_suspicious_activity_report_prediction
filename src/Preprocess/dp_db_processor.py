"""
dp_db processor

register: 註冊函數器
get_basic_info: 讀取basic_info
get_dp: 讀取dp 計算匯率轉換的amt
compute_templete_feats: agg的基本設定
get_dp_tx_type_features: 註冊函數 計算每天的templete_feat的特徵
dp_db_processor: 定義porcessor class
"""
import pandas as pd
from functools import reduce
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


def get_dp(loader=LOADER):
    """
    讀取dp 計算匯率轉換的amt
    Args:
        loader (dataloader.DataLoader, optional): Defaults to LOADER.

    Returns:
        dp (pd.DataFrmae)
    """
    dp = loader.load_train_data('dp.csv')
    dp['tx_amt_twd'] = dp['tx_amt'] * dp['exchg_rate']
    return dp


def get_basic_info(loader=LOADER):
    """
    讀取basic_info
    Args:
        loader (dataloader.DataLoader, optional):  Defaults to LOADER.

    Returns:
        basic_info (pd.DataFrame)
    """
    basic_info = loader.load_input('dp_label.joblib')
    basic_info = basic_info[['alert_key', 'date', 'cust_id', 'sar_flag']]
    return basic_info


def compute_templete_feats(dp, dp_type):
    """
    agg的基本設定
    Args:
        dp (pd.DataFrame)
        dp_type (str)

    Returns:
        dp (pd.DataFrame)
    """
    dp = dp.groupby(['cust_id', 'tx_date']).agg({
        "tx_amt_twd": ['count', 'sum', 'median'],
        "ATM": ['count', 'mean', 'sum'],
        "cross_bank": ['count', 'mean', 'sum'],
    }).reset_index()
    cols = [
        'cust_id', 'date', 'tx_amt_count_db', 'tx_amt_twd_sum_db',
        'tx_amt_twd_median_db', 'ATM_count_db', 'ATM_mean_db', 'ATM_sum_db',
        'cross_bank_count_db', 'cross_bank_mean_db', 'cross_bank_sum_db'
    ]
    cols = ['cust_id', 'date'
            ] + [col + "_{}".format(dp_type) for col in cols[2:]]
    dp.columns = cols
    return dp


@register(feat_name='dp_tx_type_features')
def get_dp_tx_type_features():
    """
    註冊函數 計算每天的templete_feat的特徵
    Returns:
        dp (pd.DataFrame)
    """
    dp = get_dp()
    dp = dp[dp['debit_credit'] == 'DB']
    tx_types = dp[['cust_id', 'tx_type', 'info_asset_code'
                   ]].groupby(['tx_type', 'info_asset_code']).count().index
    dp_list = []
    for tx_type in tx_types:
        dp_type = "type_{}_asset_{}".format(tx_type[0], tx_type[1])
        dp_split = dp[(dp['tx_type'] == tx_type[0])
                      & (dp['info_asset_code'] == tx_type[1])].copy()
        dp_list.append(compute_templete_feats(dp_split, dp_type))
    dp = reduce(
        lambda df1, df2: pd.merge(
            df1, df2, on=['cust_id', 'date'], how='outer'), dp_list)
    return dp


class dp_db_processor:
    """定義porcessor class
    method
    get_feats: 計算特徵的method 將註冊函數一個個執行
    """
    def __init__(self):
        self.name = 'dpdb'

    def get_feats(self):
        """
        計算特徵的method 將註冊函數一個個執行
        Returns:
            dp (pd.DataFrame)
        """
        df_list = [func() for func in USE_FUNCTIONS]
        dp = reduce(
            lambda df1, df2: pd.merge(
                df1, df2, on=['cust_id', 'date'], how='outer'), df_list)
        dp = dp.fillna(0)
        dp = dp.sort_values('date').reset_index(drop=True)
        return dp
