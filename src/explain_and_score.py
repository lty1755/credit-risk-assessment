import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from train import train_credit_risk_model


def run_explainability_and_scoring():
    print("正在訓練模型以進行可解釋性分析...")
    model, X_val, y_val, preds = train_credit_risk_model()

    # --- 1. SHAP 可解釋性分析 ---
    print("\n--- 正在計算 SHAP 導向的全局與個體歸因 ---")
    # 建立 TreeExplainer
    explainer = shap.TreeExplainer(model)
    # 取前 1000 筆驗證集資料來計算 SHAP 值（加快繪圖速度）
    X_val_sample = X_val.iloc[:1000]
    shap_values = explainer.shap_values(X_val_sample)

    # 針對二分類問題，shap_values 通常是一個 list，取正類（違約）的 shap 值
    if isinstance(shap_values, list):
        shap_vals_positive = shap_values[1]
    else:
        shap_vals_positive = shap_values

    # 模擬產出單一拒貸客戶的「拒貸歸因報告」
    # 挑選驗證集中預約機率最高的第一位客戶
    risky_idx = np.argmax(preds[:1000])
    print(f"\n[個體拒貸歸因報告範例 - 驗證集第 {risky_idx} 位客戶]")
    print(f"該客戶預測違約機率: {preds[risky_idx] * 100:.2f}% (超過 47.80% 截斷點 -> 拒絕放款)")

    # 列出對該客戶違約機率推波助瀾最大的前 3 個特徵
    customer_shap = shap_vals_positive[risky_idx]
    feature_importance_df = pd.DataFrame({
        'Feature': X_val_sample.columns,
        'SHAP_Value': customer_shap,
        'Feature_Value': X_val_sample.iloc[risky_idx].values
    }).sort_values(by='SHAP_Value', ascending=False)

    print("推高該客戶風險評分的前三大關鍵風險因子：")
    print(feature_importance_df.head(3).to_string(index=False))

    # --- 2. 信用評分轉換與風險分級 (Scorecard & Risk Tiers) ---
    print("\n--- 正在轉換為傳統信用評分與風險分級 (Tier A-D) ---")
    # 將違約機率 (PD) 轉換為傳統信用分數 (Score 300 ~ 850 分)
    # 邏輯：違約機率越低，分數越高
    # 公式示範：Score = A - B * ln(ODD)，這裡簡化成線性對應或標準轉換
    min_pd, max_pd = preds.min(), preds.max()
    # 將機率反轉並縮放到 300 ~ 850 分
    credit_scores = 300 + (1 - preds) * (850 - 300)

    val_result = pd.DataFrame({
        'PD': preds,
        'Credit_Score': credit_scores
    })

    # 定義風險分級策略 (Risk Tiers)
    # Tier A (極低風險): 分數 >= 750 (優質客戶，享有最低利率)
    # Tier B (低風險): 700 <= 分數 < 750 (標準利率)
    # Tier C (中風險): 600 <= 分數 < 700 (提高利率 / 額度控管)
    # Tier D (高風險): 分數 < 600 或 違約機率 > 0.478 (拒絕放款)
    def assign_tier(row):
        if row['PD'] > 0.4780:
            return 'Tier D (高風險-拒絕放款)'
        elif row['Credit_Score'] >= 750:
            return 'Tier A (極低風險-優利專案)'
        elif row['Credit_Score'] >= 700:
            return 'Tier B (低風險-標準定價)'
        else:
            return 'Tier C (中風險-風險定價/加碼)'

    val_result['Risk_Tier'] = val_result.apply(assign_tier, axis=1)

    tier_counts = val_result['Risk_Tier'].value_counts()
    print("\n驗證集客戶風險分級分佈統計：")
    print(tier_counts)

    print("\n【恭喜】信用風險模型專案的全套核心邏輯（工程、訓練、評估、解釋、定價）已完整展示！")


if __name__ == "__main__":
    run_explainability_and_scoring()