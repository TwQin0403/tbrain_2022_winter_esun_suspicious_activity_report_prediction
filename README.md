# tbrain_2022_winter_esun_suspicious_activity_report_prediction

the 5th place solution for 2022 winter e-sun bank suspicious activity report prediction.

## 訓練方式

### 訓練流程
![](https://raw.githubusercontent.com/TwQin0403/tbrain_2022_winter_esun_suspicious_activity_report_prediction/main/docs/fig/train_process.png)

大致分成三個步驟：
- 計算dp表格與cdtx表格的異常值
- 計算某一(cust_id, date)的is_alert_key_pred
- 混合其他特徵做最終的訓練

### Step1 訓練
![](https://raw.githubusercontent.com/TwQin0403/tbrain_2022_winter_esun_suspicious_activity_report_prediction/main/docs/fig/step1.png)

- 使用dp資訊計算有轉帳訊息的異常程度(by date) -&gt; 得到dp_abnormal資料
- 使用cdtx資訊計算有交易訊息的異常程度(by date) -&gt;得到cdtx_abnormal 資料

### Step2 訓練:
![](https://raw.githubusercontent.com/TwQin0403/tbrain_2022_winter_esun_suspicious_activity_report_prediction/main/docs/fig/step2.png)

- 使用dp_abnormal, cdtx_abnormal, dp, cdtx資料去預測當天(cust_id, date)有alert_key的機率-&gt; 得到 is_alert_key_pred 資料

### Step3 訓練：
- 使用 dp_abnormal, cdtx_abnormal is_alert_key_pred, dp資訊, cdtx資訊 custinfo資訊, ccba資訊等 去預測 是否會被提報為SAR

- 用total_asset作為分組來做用StratifiedGroupKFold的CV

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
