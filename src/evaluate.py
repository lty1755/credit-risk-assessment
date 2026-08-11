import pandas as pd
import lightgbm as lgb
import re
from preprocess import load_and_preprocess_data


def generate_submission(model, train_features):
    print("正在讀取測試集...")
    test_df = pd.read_csv("data/application_test.csv")

    # 儲存 SK_ID_CURR 用於最終產出
    test_ids = test_df['SK_ID_CURR']

    # 使用與訓練階段相同的預處理邏輯
    # 注意：這裡直接讀取並預處理測試集
    # 為了確保特徵名稱與 train 階段一致，我們需要進行一些對齊
    test_processed = load_and_preprocess_data("data/application_test.csv")

    # 過濾特殊字元 (與 train 階段一致)
    test_processed = test_processed.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    # 對齊特徵：確保測試集的欄位順序與訓練集完全一致
    # 訓練集有的欄位測試集也要有，沒有的要補 0
    features = train_features.columns.tolist()

    # 只保留訓練集有的特徵，缺失的補 0
    for col in features:
        if col not in test_processed.columns:
            test_processed[col] = 0

    # 只取訓練集出現過的欄位，並按照順序排列
    test_final = test_processed[features]

    print("正在進行預測...")
    preds = model.predict(test_final, num_iteration=model.best_iteration)

    # 產出提交檔案
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': preds})
    submission.to_csv("submission.csv", index=False)
    print("預測完成！已產出 submission.csv 到專案根目錄。")


if __name__ == "__main__":
    # 為了重新訓練並獲取模型，我們直接調用 train 中的訓練邏輯
    from train import train_credit_risk_model

    model, X_train, y_train, _ = train_credit_risk_model()

    generate_submission(model, X_train)