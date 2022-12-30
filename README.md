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

## Remark
最後的submission是用沒有加入public的submission * 0.6 + 加入public * 0.4 混合提交。 要得到加入public的submission的結果需要自己手動更新dp, ccba, cdtx, custinfo等檔案
