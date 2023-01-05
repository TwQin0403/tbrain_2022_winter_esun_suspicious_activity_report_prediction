# tbrain_2022_winter_esun_suspicious_activity_report_prediction

the 6th place solution for 2022 winter e-sun bank suspicious activity report prediction.

## 檔案用途:
- src/Preprocess/: 存放前處理的code
- src/Model/: 存放訓練相關code
- src/train_data/: 存放原始資料
- src/dataloader.py:處理io
- src/utils.py: 處理雜事
- src/config.py:參數設定
- src/generator.py:處理資料產生
- setup.py: 打包src用
- requirements.txt: 需要的套件
- src/main.py: 執行整個訓練流程

## 執行流程:
```
# 更改檔案名稱(見下方)
$ 將alert_date.csv, ccba.csv, cdtx.csv, custinfo.csv, dp.csv, private_x_alert_date.csv, public_x_alert_date.csv, remit.csv, y.csv放入train_data資料夾
# 創建虛擬環境
$ virtualenv e_sun_2022
# 進入虛擬環境
$ source e_sun_2022/bin/activate
# 安裝所需套件
$ pip install -r requirements.txt 
# 打包模塊
$ pip install -e .
# training and inference
$ cd src
$ python main.py
```

## 更改檔案名稱

### Not add private version
```
# train_x_alert_date.csv -> alert_date.csv
# public_train_x_ccba_full_hashed.csv -> ccba.csv
# public_train_x_cdtx0001_full_hashed.csv -> cdtx.csv
# public_train_x_custinfo_full_hashed.csv -> custinfo.csv
# public_train_x_dp_full_hashed.csv -> dp.csv
# public_train_x_remit1_full_hashed.csv -> remit.csv
# train_y_answer.csv -> y.csv
```

### add private version
```
# 將public_train_x_ccba_full_hashed.csv 與  private_x_ccba_full_hashed.csv concat reset_index -> ccba.csv
# 將public_train_x_cdtx_full_hashed.csv 與  private_x_cdtx0001_full_hashed.csv concat reset_index -> cdtx.csv
# 將public_train_x_custinfo_full_hashed.csv 與  private_x_custinfo_full_hashed.csv concat reset_index -> custinfo.csv
# 將public_train_x_dp_full_hashed.csv 與  private_x_dp_full_hashed.csv concat reset_index -> dp.csv
# 將public_train_x_remit_full_hashed.csv 與  private_x_remit1_full_hashed.csv concat reset_index -> remit.csv
# 將train_y_answer.csv 與  public_y_answer concat reset_index -> y.csv
```

## Remark
最後的submission是用沒有加入public的submission * 0.6 + 加入public * 0.4 混合提交。 要得到加入public的submission的結果需要自己手動更新dp, ccba, cdtx, custinfo等檔案
