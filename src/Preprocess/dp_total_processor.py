"""
dp_total processor

register: 註冊函數器
get_basic_info: 讀取basic_info
get_dp: 讀取dp 計算匯率轉換的amt
get_dp_total_pair: 註冊函數 計算每天的templete_feat的特徵
dp_total_processor: 定義porcessor class
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


@register(feat_name='dp_total_pair')
def get_dp_total_pair():
    """
    註冊函數 計算每天的templete_feat的特徵
    Returns:
        dp_feats (pd.DataFrame)
    """
    dp = get_dp()
    # compute the dp_cr total value
    dp_cr = dp[dp['debit_credit'] == 'CR']
    dp_cr = dp_cr.groupby(['cust_id', 'tx_date']).agg({
        "tx_amt_twd": ['sum']
    }).reset_index()
    cols = ['cust_id', 'date', 'cr_total']
    dp_cr.columns = cols
    # compute the dp_db total_value
    dp_db = dp[dp['debit_credit'] == 'DB']
    dp_db = dp_db.groupby(['cust_id', 'tx_date']).agg({
        "tx_amt_twd": ['sum']
    }).reset_index()
    cols = ['cust_id', 'date', 'db_total']
    dp_db.columns = cols
    # merge the results
    dp_feats = pd.merge(dp_db, dp_cr, on=['cust_id', 'date'], how='outer')
    dp_feats = dp_feats.fillna(0)
    dp_feats = dp_feats.reset_index(drop=True)
    dp_feats['dp_balanced'] = dp_feats['db_total'] - dp_feats['cr_total']
    return dp_feats


class dp_total_processor:
    """定義porcessor class
    method
    get_feats: 計算特徵的method 將註冊函數一個個執行
    """
    def __init__(self):
        self.name = 'dptotal'

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
