"""
執行訓練流程
"""
import utils
import dataloader
import Model.train as train

LOADER = dataloader.DataLoader()

if __name__ == '__main__':
    # generate basic_info
    basic_info = utils.generate_basic_info()
    LOADER.save_input(basic_info, 'basic_info.joblib')

    # generate step1 meta data
    try:
        LOADER.load_input('stage3_cdtx_abnormal.joblib')
        LOADER.load_input('stage3_dp_abnormal.joblib')
    except FileNotFoundError:
        train.train_step1()

    # generate step2 meta data
    try:
        LOADER.load_input('stage3_is_alert_key_pred.joblib')
    except FileNotFoundError:
        train.train_step2()

    # step3 training
    # generate submission file in Model folder
    train.train()
