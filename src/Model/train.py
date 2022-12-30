"""
訓練的程式碼

train_step1: step 1訓練 產生dp_abnormal跟cdtx abnormal 在Preprocess
train_step2: step 2訓練 產生is_alert_pred在 Preprocess
train: step3 訓練 並產生sample_submission在Model
Returns:
    _type_: _description_
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import src.utils as utils

import src.dataloader as dataloader
import src.generator as generator
import src.config as config

LOADER = dataloader.DataLoader()


def train_step1():
    """
    step 1訓練 產生dp_abnormal跟cdtx abnormal 在Preprocess
    """
    cdtx_ab = generator.AbnormalCdtxGenerator()
    cdtx_ab.get_feats()
    dp_ab = generator.AbnormalDpGenerator()
    dp_ab.get_feats()


def train_step2():
    """
    step 2訓練 產生is_alert_pred在 Preprocess
    """
    alert_generator = generator.AlertKeyPredGenerator()
    alert_generator.train()
    alert_generator.get_feats()


def train(random_state=None, name='0'):
    """
    step3 訓練 並產生sample_submission在Model
    Args:
        random_state (int, optional): cam choose random seed. Defaults to None.
        name (str, optional): name. Defaults to '0'.

    Returns:
        _type_: _description_
        submission (pd.DataFrame): submission
        preds (pd.DataFrame): prediction for validation
        models (Object): models
    """
    try:
        feats = LOADER.load_input('current_feats.joblib')
    except FileNotFoundError:
        print("Start generate feats")
        data_generator = generator.DataGenerator()
        feats = data_generator.get_feats()
    feats = feats[feats['date'] >= 30]
    y = LOADER.load_train_data('y.csv')

    data = pd.merge(y, feats, on='alert_key', how='right')
    print(data.head())
    train_data = data[data['sar_flag'].notna()].copy()
    train_data = train_data.reset_index(drop=True)
    train_data = train_data.fillna(0)
    X = train_data.drop(['alert_key', 'sar_flag', 'cust_id', 'date', 'group'],
                        axis=1).copy()
    y = train_data[['sar_flag']].copy()
    groups = np.array(train_data['group'].to_list())
    params = config.LGB_BASIC_SETTING
    if random_state is not None:
        kf = StratifiedGroupKFold(n_splits=10,
                                  shuffle=True,
                                  random_state=random_state)
    else:
        kf = StratifiedGroupKFold(n_splits=10)

    models = []
    pred_val = []
    for train_index, test_index in kf.split(X, y, groups):
        templete_df = train_data[['alert_key', 'sar_flag']].copy()
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
            params=params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            categorical_feature=[
                'AGE', 'risk_rank', 'occupation_code', 'is_last_group',
                'is_large_ATM', 'is_large_cross_bank', 'is_large_CR',
                'is_large_DB'
            ],
            feval=utils.recall_n_eval_lgb,
            num_boost_round=3000,
            verbose_eval=100,
            early_stopping_rounds=600,
        )
        templete_df['pred'] = gbm.predict(X_val)
        pred_val.append(templete_df)
        models.append(gbm)
    sample_submission = utils.generate_sample_submisssion(
        data,
        drop_cols=['sar_flag', 'cust_id', 'date', 'group'],
        models=models,
        val_type='mean')
    submission = sample_submission.sort_values(by='probability',
                                               ascending=False)
    LOADER.save_model(submission, 'submission_{}'.format(name), cls_type='csv')
    df_list = []
    for _, pred_df in enumerate(pred_val):
        fold = pred_df.copy()
        fold = fold[['alert_key', 'pred']]
        df_list.append(fold)
    preds = pd.concat(df_list)
    preds.sort_values(by='pred')

    return submission, preds, models
