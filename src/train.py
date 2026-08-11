import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.preprocess import load_and_preprocess_data


def calculate_psi(expected, actual, num_bins=10):
    """
    計算群體穩定度指標 (PSI)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected) + 1e-4
    actual_pct = actual_counts / len(actual) + 1e-4

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value


def optimize_cost_matrix(y_true, y_pred_proba):
    """
    商業成本矩陣最佳化 (呆帳成本 10x vs 機會成本 1x)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    min_cost = float('inf')

    cost_fn = 10.0  # 違約放款成本 (Type II Error)
    cost_fp = 1.0  # 拒絕好客戶成本 (Type I Error)

    for th in thresholds:
        y_pred = (y_pred_proba >= th).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_cost = (fn * cost_fn) + (fp * cost_fp)

        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = th

    return best_threshold, min_cost


def train_credit_risk_model():
    print("正在載入與預處理所有附表與主檔數據...")
    df = load_and_preprocess_data("data/application_train.csv")

    # --- 關鍵：依據 SK_ID_CURR（申請流水號）排序，模擬時間序列 (OOT) ---
    df = df.sort_values('SK_ID_CURR').reset_index(drop=True)

    X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = df['TARGET']

    # 前 80% 作為訓練集（過去），後 20% 作為 OOT 驗證集（未來）
    split_index = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    print(f"訓練集樣本數: {len(X_train)}, OOT 驗證集樣本數: {len(X_val)}")

    # 計算類別不平衡權重
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"計算得出的正負樣本權重比 (scale_pos_weight): {scale_pos_weight:.2f}")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1
    }

    print("--- 訓練 OOT 時間序列 LightGBM 模型 ---")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50)]
    )

    # 預測與評估
    val_preds = model.predict(X_val)
    auc = roc_auc_score(y_val, val_preds)
    print(f"OOT 驗證集 ROC-AUC: {auc:.4f}")

    # 商業成本矩陣最佳化
    best_th, lowest_cost = optimize_cost_matrix(y_val, val_preds)
    print(f"💼 【商業成本矩陣】OOT 最佳放款門檻: {best_th:.4f} (最低總成本: {lowest_cost:.1f})")

    # PSI 模型監控
    train_preds = model.predict(X_train)
    psi_score = calculate_psi(train_preds, val_preds)
    print(f"📊 【MLOps 監控】OOT 預測分佈 PSI: {psi_score:.4f} (小於 0.1 代表分佈穩定)")

    print("正在將模型與驗證集樣本打包儲存為 .pkl 檔...")
    os.makedirs('models', exist_ok=True)

    # 1. 儲存訓練好的 LightGBM 模型
    joblib.dump(model, 'models/best_model.pkl')

    # 2. 儲存一小張驗證集樣本（例如取前 100 筆給網頁互動使用，檔案才不會太大）
    X_val_sample = X_val.head(100)
    joblib.dump(X_val_sample, 'models/X_val_sample.pkl')

    print("✅ 打包完成！'models/best_model.pkl' 與 'models/X_val_sample.pkl' 已建立。")


    return model, X_val, y_val

if __name__ == "__main__":
    train_credit_risk_model()