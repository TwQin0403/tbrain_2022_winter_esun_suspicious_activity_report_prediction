"""
做雜事
get_labels: 讀取basic_info
positive_sigmoid: 計算sigmoid用
negative_sigmoid: 計算sigmoid用
sigmoid: 計算sigmoid用
recall_n_eval_lgb: The implementation of eval function
recall_n_eval: The implementation of eval function
generate_sample_submisssion: 用來產生sample_submission
get_validation_score: 在publice公布後 用來算public score的
generate_basic_info: 用來產生basic_info
"""
import numpy as np
import pandas as pd
import dataloader
import warnings

warnings.simplefilter("ignore", UserWarning)

LOADER = dataloader.DataLoader()


def get_labels():
    """
    讀取basic_info

    Returns:
        dp_label (pd.DataFrame): _description_
    """
    #
    dp_label = LOADER.load_input('basic_info.joblib')
    dp_label = dp_label[['alert_key', 'date', 'cust_id', 'sar_flag']]
    return dp_label


def positive_sigmoid(x):
    """
    計算sigmoid用

    Args:
        x (np.array): 要計算的值

    Returns:
        (np.array): 計算值
    """

    return 1 / (1 + np.exp(-x))


def negative_sigmoid(x):
    """
    計算sigmoid用

    Args:
        x (np.array): 要計算的值

    Returns:
        (np.array): 計算值
    """
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    """
    計算sigmoid用

    Args:
        x (np.array): 要計算的值

    Returns:
        (np.array): 計算值
    """
    positive = x >= 0
    negative = ~positive
    result = np.empty_like(x)
    result[positive] = positive_sigmoid(x[positive])
    result[negative] = negative_sigmoid(x[negative])
    return result


def recall_n_eval_lgb(z, data):
    """
    The implementation of eval function
    Args:
        z (np.array):
        data (lightgbm.Dataset)

    Returns:
        標準的eval function return
    """
    y_true = data.get_label()
    y_hat = sigmoid(z)
    y_true = y_true.reshape(-1, 1)
    y_hat = y_hat.reshape(-1, 1)
    y = np.concatenate((y_true, y_hat), axis=1)
    y = pd.DataFrame(y)
    y.columns = ['y_true', 'y_hat']
    y.columns = ['y_true', 'y_hat']

    target_y = y[y['y_true'] == 1].sort_values(by='y_hat')['y_hat'].iloc[1]
    recall_n_mins_1 = (len(y[y['y_true'] == 1]) - 1) / len(
        y[y['y_hat'] >= target_y])
    return 'recall_n', recall_n_mins_1, True


def recall_n_eval(y_hat, y_true):
    """
    The implementation of eval function
    方便用np.array直接算分
    Args:
        y_hat (np.array):
        y_true (np.array)

    Returns:
        分數
    """
    y_true = y_true.reshape(-1, 1)
    y_hat = y_hat.reshape(-1, 1)
    y = np.concatenate((y_true, y_hat), axis=1)
    y = pd.DataFrame(y)
    y.columns = ['y_true', 'y_hat']
    target_y = y[y['y_true'] == 1].sort_values(by='y_hat')['y_hat'].iloc[1]
    recall_n_mins_1 = (len(y[y['y_true'] == 1]) - 1) / len(
        y[y['y_hat'] >= target_y])
    return recall_n_mins_1


def generate_sample_submisssion(data, drop_cols, models, val_type='mean'):
    """
    用來產生sample_submission
    Args:
        data (pd.DataFrame): 預測資料
        drop_cols (list): 要去掉的columns
        models (list of models): 訓練好的模型
        val_type (str, optional): 要用那個模型預測 'mean'代表取算術平均. Defaults to 'mean'.

    Returns:
        submission (pd.DataFrame): 提交檔案
    """
    sample_submission = LOADER.load_train_data('submit_sample.csv')
    test_feats = data[data['sar_flag'].isna()].copy()
    test_feats = test_feats.drop(drop_cols, axis=1)
    X_test = test_feats.drop('alert_key', axis=1).copy()
    model_preds = pd.DataFrame(
        {'alert_key': test_feats['alert_key'].to_list()})
    for i, model in enumerate(models):
        pred = model.predict(X_test)
        model_preds['pred_{}'.format(i + 1)] = pred
    if val_type == 'mean':
        model_preds['preds'] = model_preds.drop('alert_key',
                                                axis=1).mean(axis=1)
    else:
        model_preds['preds'] = model_preds[val_type].to_list()
    # model_preds['preds'] = model_preds.drop('alert_key', axis=1).max(axis=1)
    map_preds = {
        alert_key: pred
        for alert_key, pred in zip(model_preds['alert_key'],
                                   model_preds['preds'])
    }

    def map_the_preds(alert_key):
        if alert_key in map_preds.keys():
            return map_preds[alert_key]
        else:
            return 0

    sample_submission['probability'] = sample_submission['alert_key'].apply(
        map_the_preds)
    return sample_submission


def get_validation_score(sample_submission):
    # 用來產生public score
    """
    在publice公布後 用來算public score的

    Args:
        sample_submission (pd.DataFrame): 提交檔案

    Returns:
        ans (pd.DataFrame)
    """
    ans = LOADER.load_train_data('public_y_answer.csv')
    ans = pd.merge(ans, sample_submission, on='alert_key', how='left')
    print(
        recall_n_eval(np.array(ans['probability']), np.array(ans['sar_flag'])))
    return ans


def generate_basic_info():
    """
    用來產生basic_info
    Returns:
        basic_info (pd.DataFrame): 把alert_key與(cust_id, date)
        連結起來的檔案，方便整理
    """
    basic_info = LOADER.load_train_data('y.csv')
    custinfo = LOADER.load_train_data('custinfo.csv')
    basic_info = pd.merge(custinfo, basic_info, on='alert_key', how='left')
    # merge date
    alert_date = LOADER.load_train_data('alert_date.csv')
    public_x_alert_date = LOADER.load_train_data('public_x_alert_date.csv')
    private_x_alert_date = LOADER.load_train_data('private_x_alert_date.csv')

    all_alert_date = pd.concat(
        [alert_date, public_x_alert_date, private_x_alert_date])

    basic_info = pd.merge(basic_info,
                          all_alert_date,
                          on='alert_key',
                          how='left')
    basic_info = basic_info[['alert_key', 'cust_id', 'date', 'sar_flag']]
    basic_info = basic_info.sort_values(by='alert_key')
    basic_info = basic_info.reset_index(drop=True)
    return basic_info
