"""
    用來產生需要的特徵的class集合
    AlertKeyPredGenerator: 產生第二階段預測當天是否會產生alert_key的預測值
    AbnormalDpGenerator: 產生dp的異常值
    AbnormalCdtxGenerator: 產生cdtx的異常值
    Aggregator: 產生聚合每天的特徵的class
    DataGenerator: 產生最後用來訓練的特徵
"""
from collections import OrderedDict
from functools import reduce
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest

from config import IS_ALERT_PARAMS
import Preprocess.custinfo_processor as custinfo_processor
import Preprocess.dp_cr_processor as dp_cr_processor
import Preprocess.dp_db_processor as dp_db_processor
import Preprocess.dp_total_processor as dp_total_processor
import Preprocess.cdtx_processor as cdtx_processor

import dataloader
import utils

LOADER = dataloader.DataLoader()


class AlertKeyPredGenerator():
    """
    產生第二階段預測當天是否會產生alert_key的預測
    其他get_feats method 會回傳is_alert_key_pred並存在Preprocess
    methods:
    generate_is_alert_key: 產生y值 is_alert_key
    get_train_feats: 產生X值 訓練特徵
    train: 執行第二階段訓練
    get_features: 取得前30落後項的預測值
    get_feats: 得到第三階段的特徵值
    """
    def __init__(self):
        """
        讀入需要的資料
        """
        self.basic_info = LOADER.load_input('basic_info.joblib')
        self.dp_cr_processor = dp_cr_processor.dp_cr_processor()
        self.dp_db_processor = dp_db_processor.dp_db_processor()
        self.cdtx_processor = cdtx_processor.cdtx_processor()
        self.is_alert_key_label = self.generate_is_alert_key()
        self.dp_abnormal_feats = LOADER.load_input('stage2_dp_abnormal.joblib')
        self.cdtx_abnormal_feats = LOADER.load_input(
            'stage2_cdtx_abnormal.joblib')
        self.get_train_feats()

    def generate_is_alert_key(self):
        """
        產生y值 is_alert_key
        Input: None
        Returns: is_alert_key pd.DataFrame: (cust_id, date) 是否為alert_key
        """
        basic_info = utils.get_labels()
        basic_info = basic_info.sort_values(by='date')
        possible_date = list(basic_info['date'].unique())
        is_alert_key = basic_info.groupby('cust_id')
        is_alert_key = [df[1] for df in is_alert_key]
        results = []
        for df in is_alert_key:
            templete_df = pd.DataFrame({'date': possible_date})
            templete_df['cust_id'] = df['cust_id'].iloc[0]
            templete_df['is_alert_key'] = 0
            templete_df.loc[templete_df['date'].isin(df['date'].to_list()),
                            'is_alert_key'] = 1
            results.append(templete_df)
        is_alert_key = pd.concat(results)
        is_alert_key = is_alert_key.reset_index(drop=True)
        return is_alert_key

    def get_train_feats(self):
        """
        產生第二階段的訓練特徵
        Input: None
        Returns: None
        """
        dp_cr_feats = self.dp_cr_processor.get_feats()
        dp_db_feats = self.dp_db_processor.get_feats()
        cdtx_feats = self.cdtx_processor.get_feats()
        cdtx_feats = cdtx_feats.reset_index(drop=True)
        cdtx_feats = cdtx_feats.fillna(0)
        feats = [
            dp_cr_feats, dp_db_feats, cdtx_feats, self.dp_abnormal_feats,
            self.cdtx_abnormal_feats
        ]
        feats = reduce(
            lambda df1, df2: pd.merge(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True),
                                      on=['cust_id', 'date'],
                                      how='outer'), feats)
        feats = feats.fillna(0)
        self.raw_feats = feats.copy()
        # generate alert key label
        basic_info = LOADER.load_input('basic_info.joblib')
        feats['is_alert_key'] = 0
        feats['tmp_set'] = [
            (cust_id, date)
            for cust_id, date in zip(feats['cust_id'], feats['date'])
        ]
        tmp_idx = [
            (cust_id, date)
            for cust_id, date in zip(basic_info['cust_id'], basic_info['date'])
        ]
        feats['is_alert_key'] = np.where((feats['tmp_set'].isin(tmp_idx)), 1,
                                         0)
        feats = feats.drop('tmp_set', axis=1)
        # delete samples after the sar_flag = 1
        del_idx = []
        sar_df = self.basic_info[self.basic_info['sar_flag'] == 1].copy()
        is_sar_idx = [
            (cust_id, date)
            for cust_id, date in zip(sar_df['cust_id'], sar_df['date'])
        ]

        for (is_sar_id, date) in is_sar_idx:
            tmp_df = feats[(feats['cust_id'] == is_sar_id)
                           & (feats['date'] > date)].copy()
            del_idx += list(tmp_df.index)
        feats = feats.drop(del_idx)
        feats = feats[feats['date'] >= 30]
        feats = feats.reset_index(drop=True)
        self.train_feats = feats

    def train(self):
        """
        執行第二階段訓練
        Input: None
        Returns: None
        """
        X = self.train_feats.drop(['cust_id', 'date', 'is_alert_key'],
                                  axis=1).copy()
        y = self.train_feats[['is_alert_key']].copy()
        kf = StratifiedKFold(n_splits=10)
        kf.get_n_splits(X)
        models = []
        pred_val = []
        for train_index, test_index in kf.split(X, y):
            templete_df = self.train_feats[['cust_id', 'date',
                                            'is_alert_key']].copy()
            templete_df = templete_df.loc[test_index]
            X_train, X_val = X.loc[train_index], X.loc[test_index]
            y_train, y_val = y.loc[train_index], y.loc[test_index]

            train_dataset = lgb.Dataset(X_train, y_train)
            val_dataset = lgb.Dataset(
                X_val,
                y_val,
                reference=train_dataset,
            )
            gbm = lgb.train(
                params=IS_ALERT_PARAMS,
                train_set=train_dataset,
                valid_sets=[train_dataset, val_dataset],
                num_boost_round=5000,
                verbose_eval=100,
                early_stopping_rounds=600,
            )
            models.append(gbm)
            # get val pred results
            templete_df['pred'] = gbm.predict(X_val)
            pred_val.append(templete_df)
        self.models = models
        self.pred_val = pred_val

        df_list = []
        for i, pred_df in enumerate(pred_val):
            fold = pred_df.copy()
            df_list.append(fold)

        preds = pd.concat(df_list)
        self.preds = preds
        LOADER.save_input(preds, 'stage3_is_alert_key.joblib')

    def get_features(self, scores, cust_id, date):
        """
        取得前30落後項的預測值
        Args:
            scores (pd.DataFrame): is_alert_key
            cust_id (str): cust_id
            date (int): date

        Returns:
           result (dict): is_alert_pred的30落後項
        """
        result = {'cust_id': [cust_id], 'date': [date]}
        use_values = scores[(scores['cust_id'] == cust_id)
                            & (scores['date'] <= date)].copy()
        for fetch_date in range(30):
            date_idx = date - fetch_date
            date_df = use_values[use_values['date'] == date_idx].copy()
            if len(date_df) > 0:
                value = date_df['pred'].iloc[0]
            else:
                value = 0
            result.update({"is_alert_pred_{}".format(fetch_date): value})
        return result

    def get_feats(self):
        """
        得到第三階段的特徵, 並在input存下檔案
        Input: None
        Returns:
            is_alert_key_pred (pd.DataFrame): is_alert_key 預測值
        """
        try:
            is_alert_key = LOADER.load_input('stage3_is_alert_key.joblib')
        except FileNotFoundError:
            self.train()
            is_alert_key = self.preds
        basic_info = utils.get_labels()
        # generate step_3 features
        results = []
        count = 0
        for cust_id, date in zip(basic_info['cust_id'], basic_info['date']):
            count += 1
            if count % 100 == 0:
                print(count)
            feat = self.get_features(is_alert_key, cust_id, date)
            results.append(feat)
        is_alert_key_pred = pd.DataFrame(results)
        is_alert_key_pred['cust_id'] = is_alert_key_pred['cust_id'].apply(
            lambda x: x[0])
        is_alert_key_pred['date'] = is_alert_key_pred['date'].apply(
            lambda x: x[0])
        LOADER.save_input(is_alert_key_pred, 'stage3_is_alert_key_pred.joblib')
        return is_alert_key_pred


class AbnormalDpGenerator:
    """
    產生dp的異常值
    methods:
    get_train_feats: 產生X值訓練特徵
    train: 訓練產生異常值
    get_features: 取得前30落後項的預測值
    get_feats: 得到特徵

    """
    def __init__(self):
        """
        讀入資料
        """
        self.basic_info = utils.get_labels()
        self.basic_info = self.basic_info[self.basic_info['date'] >= 30]
        self.dp_cr_processor = dp_cr_processor.dp_cr_processor()
        self.dp_db_processor = dp_db_processor.dp_db_processor()
        # self.dp_total_processor = dp_total_processor.dp_total_processor()
        self.get_train_feats()

    def get_train_feats(self):
        """
        產生X值訓練特徵
        Input: None
        Returns: None
        """
        dp_cr_feats = self.dp_cr_processor.get_feats()
        dp_db_feats = self.dp_db_processor.get_feats()
        # dp_total_feats = self.dp_total_processor.get_feats()
        feats = [dp_cr_feats, dp_db_feats]
        self.train_feats = reduce(
            lambda df1, df2: pd.merge(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True),
                                      on=['cust_id', 'date'],
                                      how='outer'), feats)
        self.train_feats = self.train_feats.fillna(0)

    def train(self):
        """
        訓練產生異常值
        Returns:
            dp_abnormal (pd.DataFrame): dp_abnormal的值
        """
        date_list = list(self.train_feats['date'].unique())
        basic_info = utils.get_labels()
        templete_df = pd.DataFrame(
            {"cust_id": list(basic_info['cust_id'].unique())})
        final_df = {"cust_id": list(basic_info['cust_id'].unique())}
        for a_date in date_list:
            use_df = templete_df.copy()
            print(a_date)
            a_df = self.train_feats[self.train_feats['date'] == a_date].copy()
            print(a_df.shape)
            use_df = pd.merge(use_df, a_df, on='cust_id', how='left')
            use_df = use_df.fillna(0)
            print(use_df.shape)
            X = use_df.drop('cust_id', axis=1)
            clf = IsolationForest(random_state=0, n_estimators=400, n_jobs=-1)
            clf.fit(X)
            final_df.update(
                {'dp_abnormal_{}'.format(a_date): abs(clf.score_samples(X))})
        dp_abnormal = pd.DataFrame(final_df)
        return dp_abnormal

    def get_features(self, scores, cust_id, date):
        """
        取得前30落後項的預測值
        Args:
            scores (pd.DataFrame): dp_abnormal
            cust_id (str): cust_id
            date (int): date

        Returns:
           result (dict): dp_abnormal的30落後項
        """
        use_cols = ["dp_abnormal_{}".format(date - idx) for idx in range(30)]
        use_values = scores[scores['cust_id'] == cust_id].copy()
        use_values = use_values[['cust_id'] + use_cols]
        use_values.columns = ['cust_id'] + [
            "dp_abnormal_{}".format(int(col.split('_')[2]) - date)
            for col in use_values.columns[1:]
        ]
        use_values['date'] = date
        return use_values

    def get_feats(self):
        """
        得到第2,3階段的特徵, 並在input存下檔案
        Input: None
        Returns:
            stage_2_dp_abnormal (pd.DataFrame): 第二階段dp_abnormal
            stage_3_dp_abnormal (pd.DataFrame): 第三階段dp_abnormal
        """
        dp_abnormal = self.train()
        self.dp_abnormal = dp_abnormal
        step_2_templete = self.train_feats[
            self.train_feats['date'] >= 30].copy()
        # generate step_2 features

        results = []
        count = 0
        for cust_id, date in zip(step_2_templete['cust_id'],
                                 step_2_templete['date']):
            count += 1
            if count % 1000 == 0:
                print(count)
            feat = self.get_features(dp_abnormal, cust_id, date)
            results.append(feat)
        stage_2_dp_abnormal = pd.concat(results)
        LOADER.save_input(stage_2_dp_abnormal, 'stage2_dp_abnormal.joblib')

        # generate step_3 features
        results = []
        count = 0
        for cust_id, date in zip(self.basic_info['cust_id'],
                                 self.basic_info['date']):
            count += 1
            if count % 100 == 0:
                print(count)
            feat = self.get_features(dp_abnormal, cust_id, date)
            results.append(feat)
        stage_3_dp_abnormal = pd.concat(results)
        LOADER.save_input(stage_3_dp_abnormal, 'stage3_dp_abnormal.joblib')
        return stage_2_dp_abnormal, stage_3_dp_abnormal


class AbnormalCdtxGenerator():
    """
    產生cdtx的異常值
    methods:
    get_train_feats: 產生X值訓練特徵
    train: 訓練產生異常值
    get_features: 取得前30落後項的預測值
    get_feats: 得到特徵

    """
    def __init__(self):
        """
        讀入資料
        """
        self.basic_info = utils.get_labels()
        self.basic_info = self.basic_info[self.basic_info['date'] >= 30]
        self.cdtx_processor = cdtx_processor.cdtx_processor()
        self.get_train_feats()

    def get_train_feats(self):
        """
        產生X值訓練特徵
        Input: None
        Returns: None
        """
        cdtx_feats = self.cdtx_processor.get_feats()
        cdtx_feats = cdtx_feats.reset_index(drop=True)
        cdtx_feats = cdtx_feats.fillna(0)
        self.train_feats = cdtx_feats

    def train(self):
        """
        訓練產生異常值
        Returns:
            cdtx_abnormal (pd.DataFrame): cdtx_abnormal的值
        """
        date_list = list(self.train_feats['date'].unique())
        basic_info = utils.get_labels()
        templete_df = pd.DataFrame(
            {"cust_id": list(basic_info['cust_id'].unique())})
        final_df = {"cust_id": list(basic_info['cust_id'].unique())}
        for a_date in date_list:
            use_df = templete_df.copy()
            print(a_date)
            a_df = self.train_feats[self.train_feats['date'] == a_date].copy()
            print(a_df.shape)
            use_df = pd.merge(use_df, a_df, on='cust_id', how='left')
            use_df = use_df.fillna(0)
            print(use_df.shape)
            X = use_df.drop('cust_id', axis=1)
            clf = IsolationForest(random_state=0, n_estimators=400, n_jobs=-1)
            clf.fit(X)
            final_df.update(
                {'cdtx_abnormal_{}'.format(a_date): abs(clf.score_samples(X))})
        cdtx_abnormal = pd.DataFrame(final_df)
        return cdtx_abnormal

    def get_features(self, scores, cust_id, date):
        """
        取得前30落後項的預測值
        Args:
            scores (pd.DataFrame): cdtx_abnormal
            cust_id (str): cust_id
            date (int): date

        Returns:
           result (dict): cdtx_abnormal的30落後項
        """
        use_cols = ["cdtx_abnormal_{}".format(date - idx) for idx in range(30)]
        use_values = scores[scores['cust_id'] == cust_id].copy()
        use_values = use_values[['cust_id'] + use_cols]
        use_values.columns = ['cust_id'] + [
            "cdtx_abnormal_{}".format(int(col.split('_')[2]) - date)
            for col in use_values.columns[1:]
        ]
        use_values['date'] = date
        return use_values

    def get_feats(self):
        """
        得到第2,3階段的特徵, 並在input存下檔案
        Input: None
        Returns:
            stage_2_cdtx_abnormal (pd.DataFrame): 第二階段cdtx_abnormal
            stage_3_cdtx_abnormal (pd.DataFrame): 第三階段cdtx_abnormal
        """
        cdtx_abnormal = self.train()
        self.cdtx_abnormal = cdtx_abnormal
        step_2_templete = self.train_feats[
            self.train_feats['date'] >= 30].copy()
        # generate step_2 features
        results = []
        count = 0
        for cust_id, date in zip(step_2_templete['cust_id'],
                                 step_2_templete['date']):
            count += 1
            if count % 1000 == 0:
                print(count)
            feat = self.get_features(cdtx_abnormal, cust_id, date)
            results.append(feat)
        stage_2_cdtx_abnormal = pd.concat(results)
        LOADER.save_input(stage_2_cdtx_abnormal, 'stage2_cdtx_abnormal.joblib')
        # generate step_3 features
        results = []
        count = 0
        for cust_id, date in zip(self.basic_info['cust_id'],
                                 self.basic_info['date']):
            count += 1
            if count % 100 == 0:
                print(count)
            feat = self.get_features(cdtx_abnormal, cust_id, date)
            results.append(feat)
        stage_3_cdtx_abnormal = pd.concat(results)
        LOADER.save_input(stage_3_cdtx_abnormal, 'stage3_cdtx_abnormal.joblib')
        return stage_2_cdtx_abnormal, stage_3_cdtx_abnormal


class Aggregator:
    """
    產生聚合每天的特徵的class
    methods:

    get_raw_feat_agg_single: 某天(date)的前面average_days的聚合資料
    get_raw_feats_agg: 產生聚合資料
    fit_transform:  轉換
    """
    def __init__(self):
        self.basic_info = utils.get_labels()

    def get_raw_feat_agg_single(self, data, date, name, get_raw_feats_agg_cols,
                                average_days):
        """
        某天(date)的前面average_days的聚合資料
        Args:
            data (pd.DataFrame): 要聚合資料
            date (int): 天
            name (str): 命名名稱
            get_raw_feats_agg_cols (list): 聚合的columns名稱
            average_days (int): 聚合幾天

        Returns:
            use_data (pd.DataFrame): 聚合資料
        """
        templete_df = self.basic_info[['alert_key', 'cust_id', 'date']].copy()
        new_cols, compute_dicts = get_raw_feats_agg_cols(data, name)
        use_data = data[(data['date'] <= date)
                        & (data['date'] >= date - average_days)].copy()
        use_data = use_data.groupby(['cust_id'
                                     ]).agg(compute_dicts).reset_index()
        use_data.columns = new_cols
        use_data['date'] = date
        use_data = pd.merge(templete_df,
                            use_data,
                            on=['cust_id', 'date'],
                            how='inner')
        return use_data

    def get_raw_feats_agg(self, data, name, get_raw_feats_agg_cols,
                          average_days):
        """
        產生聚合資料
        Args:
            data (pd.DataFrame): 要聚合資料
            name (str): 命名名稱
            get_raw_feats_agg_cols (list): 聚合的columns名稱
            average_days (int): 聚合幾天

        Returns:
            raw_feats (pd.DataFrame): 聚合資料
        """
        date_list = list(self.basic_info['date'].unique())
        raw_results = []
        for a_date in date_list:
            a_feats = self.get_raw_feat_agg_single(data, a_date, name,
                                                   get_raw_feats_agg_cols,
                                                   average_days)
            raw_results.append(a_feats)
        raw_feats = pd.concat(raw_results)
        return raw_feats

    def fit_transform(self,
                      data,
                      name,
                      get_raw_feats_agg_cols,
                      average_days=30):
        """
        Args:
            data (pd.DataFrame): 要聚合資料
            name (str): 命名名稱
            get_raw_feats_agg_cols (list): 聚合的columns名稱
            average_days (int): 聚合幾天

        Returns:
            agg_feats (pd.DataFrame): 聚合資料
        """
        agg_feats = self.get_raw_feats_agg(data, name, get_raw_feats_agg_cols,
                                           average_days)
        return agg_feats


class DataGenerator:
    """
    產生特徵值
    methods:
    generate_custinfo_feats: 產生custinfo features
    generate_dp_cr_feats: 產生 dp_cr features
    generate_dp_db_feats: 產生 dp_db features
    generate_cdtx_feats: 產生 cdtx features
    generate_is_alert_key_feats: 產生 is_alert_key features
    generate_dp_total_short: 產生 dp_total_short features
    generate_dp_total_long: 產生 dp_total_long features
    generate_is_large_amount: 產生 is_large_amount features
    generate_dp_abnormal: 讀取 dp_abnormal features
    generate_cdtx_abnormal: 讀取 cdtx_abnormal features
    generate_is_alert_pred_feats: 讀取 is_alert_pred features
    generate_is_alert_key: 產生is_alert_key
    get_feats: 計算全部特徵並merge在一起
    """
    def __init__(self):
        """
        讀入資料
        """
        self.custinfo_processor = custinfo_processor.custinfo_processor()
        self.dp_cr_processor = dp_cr_processor.dp_cr_processor()
        self.dp_db_processor = dp_db_processor.dp_db_processor()
        self.dp_total_processor = dp_total_processor.dp_total_processor()
        self.cdtx_processor = cdtx_processor.cdtx_processor()
        self.basic_info = utils.get_labels()
        self.register_funcs = [
            self.generate_custinfo_feats,
            self.generate_dp_cr_feats,
            self.generate_dp_db_feats,
            self.generate_cdtx_feats,
            self.generate_is_alert_key_feats,
            self.generate_dp_total_short,
            self.generate_dp_total_long,
            self.generate_is_large_amount,
            self.generate_dp_abnormal,
            self.generate_cdtx_abnormal,
            self.generate_is_alert_pred_feats,
        ]
        self.feats = []

    def generate_custinfo_feats(self):
        """
        產生custinfo features
        Returns:
           custinfo_group_feats (pd.DataFrame): 特徵值
        """
        basic_info = self.basic_info[['alert_key', 'cust_id', 'date']].copy()
        custinfo = LOADER.load_train_data('custinfo.csv')
        basic_info = pd.merge(basic_info,
                              custinfo.drop(['cust_id', 'total_asset'],
                                            axis=1),
                              on='alert_key',
                              how='left')
        custinfo_group_feats = self.custinfo_processor.get_feats()
        custinfo_group_feats = pd.merge(basic_info,
                                        custinfo_group_feats.drop(['sar_flag'],
                                                                  axis=1),
                                        on=['alert_key', 'cust_id', 'date'],
                                        how='left')
        custinfo_group_feats = custinfo_group_feats.reset_index(drop=True)
        self.feats.append(custinfo_group_feats)
        return custinfo_group_feats

    def generate_dp_cr_feats(self, average_days=30):
        """
        產生 dp_cr features
        Args:
            average_days (int, optional): 聚合幾天. Defaults to 30.
        """
        def get_raw_feats_agg_cols(data, name):
            compute_dicts = OrderedDict()
            compute_dicts.update({"date": ['count']})
            old_cols = data.columns[2:]
            new_cols = []
            for col in old_cols:
                new_cols.append(col + "_{}_sum".format(name))
                new_cols.append(col + "_{}_mean".format(name))
                new_cols.append(col + '_{}_max'.format(name))
            compute_dicts.update(
                {col: ['sum', 'mean', 'max']
                 for col in old_cols})
            new_cols = ['cust_id', '{}_total_count'.format(name)] + new_cols
            return new_cols, compute_dicts

        raw_feats = self.dp_cr_processor.get_feats()
        aggregator = Aggregator()
        dp_cr_feats = aggregator.fit_transform(raw_feats, 'dpcr',
                                               get_raw_feats_agg_cols,
                                               average_days)
        dp_cr_feats = dp_cr_feats.reset_index(drop=True)
        self.feats.append(dp_cr_feats)
        return dp_cr_feats

    def generate_dp_db_feats(self, average_days=30):
        """
        產生 dp_db features
        Args:
            average_days (int, optional): 聚合幾天. Defaults to 30.
        """
        def get_raw_feats_agg_cols(data, name):
            compute_dicts = OrderedDict()
            compute_dicts.update({"date": ['count']})
            old_cols = data.columns[2:]
            new_cols = []
            for col in old_cols:
                new_cols.append(col + "_{}_sum".format(name))
                new_cols.append(col + "_{}_mean".format(name))
                new_cols.append(col + '_{}_max'.format(name))
            compute_dicts.update(
                {col: ['sum', 'mean', 'max']
                 for col in old_cols})
            new_cols = ['cust_id', '{}_total_count'.format(name)] + new_cols
            return new_cols, compute_dicts

        raw_feats = self.dp_db_processor.get_feats()
        aggregator = Aggregator()
        dp_db_feats = aggregator.fit_transform(raw_feats, 'dpdb',
                                               get_raw_feats_agg_cols,
                                               average_days)
        dp_db_feats = dp_db_feats.reset_index(drop=True)
        self.feats.append(dp_db_feats)
        return dp_db_feats

    def generate_cdtx_feats(self, average_days=30):
        """
        產生 cdtx features
        Args:
            average_days (int, optional): 聚合幾天. Defaults to 30.
        """
        def get_raw_feats_agg_cols(data, name):
            compute_dicts = OrderedDict()
            compute_dicts.update({"date": ['count']})
            old_cols = data.columns[2:]
            new_cols = []
            for col in old_cols:
                new_cols.append(col + "_{}_sum".format(name))
                new_cols.append(col + "_{}_mean".format(name))
                new_cols.append(col + '_{}_max'.format(name))
            compute_dicts.update(
                {col: ['sum', 'mean', 'max']
                 for col in old_cols})
            new_cols = ['cust_id', '{}_total_count'.format(name)] + new_cols
            return new_cols, compute_dicts

        raw_feats = self.cdtx_processor.get_feats()
        aggregator = Aggregator()
        cdtx_feats = aggregator.fit_transform(raw_feats, 'cdtx',
                                              get_raw_feats_agg_cols,
                                              average_days)
        cdtx_feats = cdtx_feats.reset_index(drop=True)
        self.feats.append(cdtx_feats)
        return cdtx_feats

    def generate_is_alert_key_feats(self, average_days=30):
        """
        產生 is_alert_key features
        Args:
            average_days (int, optional): 聚合幾天. Defaults to 30.
        """
        def get_raw_feats_agg_cols(data, name):
            compute_dicts = OrderedDict()
            old_cols = data.columns[2:]
            new_cols = []
            for col in old_cols:
                new_cols.append(col + "_{}_sum".format(name))
            compute_dicts.update({col: ['sum'] for col in old_cols})
            new_cols = ['cust_id'] + new_cols
            return new_cols, compute_dicts

        raw_feats = self.generate_is_alert_key()
        aggregator = Aggregator()
        is_alert_key_feats = aggregator.fit_transform(raw_feats, 'alertcount',
                                                      get_raw_feats_agg_cols,
                                                      average_days)
        is_alert_key_feats = is_alert_key_feats.reset_index(drop=True)
        self.feats.append(is_alert_key_feats)
        return is_alert_key_feats

    def generate_dp_total_short(self, average_days=7):
        """
        產生 dp_total_short features
        Args:
            average_days (int, optional): 聚合幾天. Defaults to 7.
        """
        def get_raw_feats_agg_cols(data, name):
            compute_dicts = OrderedDict()
            old_cols = data.columns[2:]
            new_cols = []
            for col in old_cols:
                new_cols.append(col + "_{}_mean".format(name))
            compute_dicts.update({col: ['mean'] for col in old_cols})
            new_cols = ['cust_id'] + new_cols
            return new_cols, compute_dicts

        raw_feats = self.dp_total_processor.get_feats()
        aggregator = Aggregator()
        dp_total_short_feats = aggregator.fit_transform(
            raw_feats, 'dpshort', get_raw_feats_agg_cols, average_days)
        dp_total_short_feats = dp_total_short_feats.reset_index(drop=True)
        self.feats.append(dp_total_short_feats)
        return dp_total_short_feats

    def generate_dp_total_long(self, average_days=30):
        """
        產生 dp_total_long features
        Args:
            average_days (int, optional): 聚合幾天. Defaults to 30.
        """
        def get_raw_feats_agg_cols(data, name):
            compute_dicts = OrderedDict()
            old_cols = data.columns[2:]
            new_cols = []
            for col in old_cols:
                new_cols.append(col + "_{}_mean".format(name))
            compute_dicts.update({col: ['mean'] for col in old_cols})
            new_cols = ['cust_id'] + new_cols
            return new_cols, compute_dicts

        raw_feats = self.dp_total_processor.get_feats()
        aggregator = Aggregator()
        dp_total_long_feats = aggregator.fit_transform(raw_feats, 'dplong',
                                                       get_raw_feats_agg_cols,
                                                       average_days)
        dp_total_long_feats = dp_total_long_feats.reset_index(drop=True)
        self.feats.append(dp_total_long_feats)
        return dp_total_long_feats

    def generate_dp_abnormal(self):
        """
        產生 dp_abnormal features
        """
        basic_info = self.basic_info[['alert_key', 'cust_id', 'date']].copy()
        dp_abnormal = LOADER.load_input('stage3_dp_abnormal.joblib')
        dp_abnormal = dp_abnormal[~dp_abnormal.duplicated()]
        dp_abnormal = dp_abnormal[
            ['cust_id', 'date'] +
            ['dp_abnormal_{}'.format(-1 * i) for i in range(5)]]
        dp_abnormal = pd.merge(basic_info.reset_index(drop=True),
                               dp_abnormal.reset_index(drop=True),
                               on=['cust_id', 'date'],
                               how='inner')
        dp_abnormal = dp_abnormal.reset_index(drop=True)
        self.feats.append(dp_abnormal)
        return dp_abnormal

    def generate_cdtx_abnormal(self):
        """
        產生 cdtx_abnormal features
        """
        basic_info = self.basic_info[['alert_key', 'cust_id', 'date']].copy()
        cdtx_abnormal = LOADER.load_input('stage3_cdtx_abnormal.joblib')
        cdtx_abnormal = cdtx_abnormal[~cdtx_abnormal.duplicated()]
        cdtx_abnormal = cdtx_abnormal[
            ['cust_id', 'date'] +
            ['cdtx_abnormal_{}'.format(-1 * i) for i in range(5)]]
        cdtx_abnormal = pd.merge(basic_info.reset_index(drop=True),
                                 cdtx_abnormal.reset_index(drop=True),
                                 on=['cust_id', 'date'],
                                 how='left')
        cdtx_abnormal = cdtx_abnormal.reset_index(drop=True)
        self.feats.append(cdtx_abnormal)
        return cdtx_abnormal

    def generate_is_alert_pred_feats(self):
        """
        產生 is_alert_key_pred features
        """
        basic_info = self.basic_info[['alert_key', 'cust_id', 'date']].copy()
        is_alert_pred_feats = LOADER.load_input(
            'stage3_is_alert_key_pred.joblib')
        is_alert_pred_feats = is_alert_pred_feats[~is_alert_pred_feats.
                                                  duplicated()]
        is_alert_pred_feats = is_alert_pred_feats[
            ['cust_id', 'date'] +
            ['is_alert_pred_{}'.format(i) for i in range(5)]]
        is_alert_pred_feats = pd.merge(
            basic_info.reset_index(drop=True),
            is_alert_pred_feats.reset_index(drop=True),
            on=['cust_id', 'date'],
            how='left')
        is_alert_pred_feats = is_alert_pred_feats.reset_index(drop=True)
        self.feats.append(is_alert_pred_feats)
        return is_alert_pred_feats

    def generate_is_large_amount(self):
        """
        產生 is_large_amount features
        """
        dp = LOADER.load_train_data('dp.csv')
        custinfo = LOADER.load_train_data('custinfo.csv')
        basic_info = LOADER.load_input('basic_info.joblib')
        basic_info = pd.merge(basic_info[['alert_key', 'cust_id', 'date']],
                              custinfo[['alert_key', 'total_asset']],
                              on='alert_key',
                              how='left')
        dp['tx_twd_amt'] = dp['tx_amt'] * dp['exchg_rate']
        tmp_df = dp[[
            'cust_id', 'tx_date', 'debit_credit', 'tx_twd_amt', 'cross_bank',
            'ATM'
        ]].copy()
        tmp_df.columns = [
            'cust_id', 'date', 'debit_credit', 'tx_twd_amt', 'cross_bank',
            'ATM'
        ]

        dp_asset = pd.merge(
            tmp_df.reset_index(drop=True),
            basic_info[['alert_key', 'cust_id', 'date',
                        'total_asset']].reset_index(drop=True),
            on=['cust_id', 'date'],
            how='inner')
        dp_asset['tx_twd_amt_total_asset_ratio'] = dp_asset['tx_twd_amt'] / (
            1 + dp_asset['total_asset'])

        is_large_amount = {
            "alert_key": [],
            'cust_id': [],
            'date': [],
            'is_large_ATM': [],
            'is_large_cross_bank': [],
            'is_large_CR': [],
            'is_large_DB': []
        }
        dp_asset_group = dp_asset.groupby('alert_key')
        dp_asset_group = [df[1] for df in dp_asset_group]
        for df in dp_asset_group:
            alert_key = df['alert_key'].iloc[0]
            cust_id = df['cust_id'].iloc[0]
            date = df['date'].iloc[0]
            tmp_df = df[df['tx_twd_amt_total_asset_ratio'] > 1]
            if len(tmp_df) > 0:
                debit_credit = tmp_df['debit_credit'].to_list()
                cross_bank = tmp_df['cross_bank'].to_list()
                atm = tmp_df['ATM'].to_list()
            else:
                debit_credit = []
                cross_bank = []
                atm = []
            is_large_amount['alert_key'].append(alert_key)
            is_large_amount['cust_id'].append(cust_id)
            is_large_amount['date'].append(date)
            is_large_amount['is_large_ATM'].append(int(1 in atm))
            is_large_amount['is_large_cross_bank'].append(int(1 in cross_bank))
            is_large_amount['is_large_CR'].append(int('CR' in debit_credit))
            is_large_amount['is_large_DB'].append(int('DB' in debit_credit))
        is_large_amount = pd.DataFrame(is_large_amount)
        self.feats.append(is_large_amount)
        return is_large_amount

    def get_feats(self):
        """
        計算全部特徵並merge在一起
        Returns:
            feats (pd.DataFrame): 特徵值
        """
        for register_func in self.register_funcs:
            print(register_func)
            register_func()
        feats = reduce(
            lambda df1, df2: pd.merge(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True),
                                      on=['alert_key', 'cust_id', 'date'],
                                      how='outer'), self.feats)
        print(feats.head())
        print(feats.shape)
        return feats

    def generate_is_alert_key(self):
        """
        產生is_alert_key
        Returns:
            is_alert_key (pd.DataFrame): is_alert_key
        """
        basic_info = utils.get_labels()
        basic_info = basic_info.sort_values(by='date')
        possible_date = list(basic_info['date'].unique())
        is_alert_key = basic_info.groupby('cust_id')
        is_alert_key = [df[1] for df in is_alert_key]
        results = []
        for df in is_alert_key:
            templete_df = pd.DataFrame({'date': possible_date})
            templete_df['cust_id'] = df['cust_id'].iloc[0]
            templete_df['is_alert_key'] = 0
            templete_df.loc[templete_df['date'].isin(df['date'].to_list()),
                            'is_alert_key'] = 1
            results.append(templete_df)
        is_alert_key = pd.concat(results)
        is_alert_key = is_alert_key.reset_index(drop=True)
        return is_alert_key
