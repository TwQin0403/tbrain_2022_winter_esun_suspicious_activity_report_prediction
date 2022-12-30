"""
custinfo_processor
custinfo_processor: 定義porcessor class
"""
import pandas as pd
import src.dataloader as dataloader


class custinfo_processor:
    """定義porcessor class
    處理group features
    get_feats: 計算特徵的method
    """
    def __init__(self):
        self.loader = dataloader.DataLoader()
        self.custinfo = self.loader.load_train_data('custinfo.csv')
        self.basic_info = self.loader.load_input('basic_info.joblib')
        self.basic_info = self.basic_info[[
            'alert_key', 'date', 'cust_id', 'sar_flag'
        ]]
        self.cdtx = self.loader.load_train_data('cdtx.csv')
        self.ccba = self.loader.load_train_data('ccba.csv')

    def get_feats(self):
        """
        計算特徵的method
        Returns:
            group_feats (pd.DataFrame)
        """
        group_df = pd.merge(self.basic_info,
                            self.custinfo[['alert_key', 'total_asset']],
                            on='alert_key',
                            how='left')
        group_dfs = group_df.groupby(['cust_id', 'total_asset'])
        group_dfs = [df[1] for df in group_dfs]
        results = []
        for group, df in enumerate(group_dfs):
            df = df.sort_values(by='date')
            df['group'] = group
            df['group_no'] = [i + 1 for i in range(len(df))]
            df['is_last_group'] = [0 for i in range(len(df) - 1)] + [1]
            cust_id = df['cust_id'].iloc[0]
            cust_df = group_df[group_df['cust_id'] == cust_id].copy()
            # compute last minus first
            f_asset = cust_df['total_asset'].iloc[0]
            c_asset = df['total_asset'].iloc[0]
            df['asset_change_total'] = c_asset - f_asset
            # compute the last - min
            df['asset_change_max'] = c_asset - cust_df['total_asset'].min()
            df['asset_change_max_ratio'] = c_asset / (
                1 + cust_df['total_asset'].min())

            # compute the last - max
            df['asset_change_min'] = c_asset - cust_df['total_asset'].max()
            df['asset_change_min_ratio'] = cust_df['total_asset'].max() / (
                1 + cust_df['total_asset'])
            # compute groups cdtx sum
            tmp_df = self.cdtx[
                (self.cdtx['cust_id'] == df['cust_id'].iloc[0])
                & (self.cdtx['date'].isin(df['date'].to_list()))].copy()
            tmp_df = tmp_df.reset_index(drop=True)

            tmp_df['group_cumsum_amt'] = tmp_df['amt'].cumsum()
            tmp_df = tmp_df[['cust_id', 'date',
                             'group_cumsum_amt']].groupby(['cust_id',
                                                           'date']).tail(1)
            tmp_df.columns = ['cust_id', 'date', 'group_cum_amt']
            df = pd.merge(df, tmp_df, on=['cust_id', 'date'], how='left')
            df['group_cum_amt_asset_ratio'] = df['group_cum_amt'] / (1 +
                                                                     c_asset)
            # compute the cycam ratio
            cust_ccba = self.ccba[self.ccba['cust_id'] == cust_id].copy()
            if len(cust_ccba) > 0:
                cycam = cust_ccba['cycam'].iloc[0]
                if cycam > 0:
                    df['asset_ratio'] = c_asset / cycam
                    df['asset_change_ratio'] = (c_asset - f_asset) / cycam
                    df['asset_change_max_ratio'] = (
                        c_asset - cust_df['total_asset'].min()) / cycam
                    df['asset_change_min_ratio'] = (
                        c_asset - cust_df['total_asset'].max()) / cycam
                    df['group_cum_amt_ratio'] = df['group_cum_amt'] / cycam
            else:
                df['asset_ratio'] = -1
                df['asset_change_ratio'] = -1
                df['asset_change_max_ratio'] = -1
                df['asset_change_min_ratio'] = -1
                df['group_cum_amt_ratio'] = -1

            results.append(df)
        group_feats = pd.concat(results)
        return group_feats
