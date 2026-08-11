import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from train import train_credit_risk_model


def evaluate_risk_metrics():
    print("正在訓練模型以獲取驗證集預測結果...")
    model, X_val, y_val, preds = train_credit_risk_model()

    # 1. 計算 ROC-AUC
    auc = roc_auc_score(y_val, preds)
    gini = 2 * auc - 1
    print(f"\n--- 模型整體指標 ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Gini 係數: {gini:.4f}")

    # 2. 計算 KS 統計量 (Kolmogorov-Smirnov)
    # KS = 最大 (累積好客戶比例 - 累積壞客戶比例)
    fpr, tpr, thresholds = roc_curve(y_val, preds)
    ks_statistic = max(tpr - fpr)
    print(f"KS 統計量 (KS Statistic): {ks_statistic:.4f} ({ks_statistic * 100:.2f}%)")

    # 3. 計算最佳截斷點 (Optimal Cutoff Threshold) 透過 Youden's J 統計量 (J = TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\n--- 業務決策指標 ---")
    print(f"最佳預測機率截斷點 (Optimal Threshold): {optimal_threshold:.4f}")
    print(f"意義：當預測違約機率大於 {optimal_threshold * 100:.2f}% 時，系統建議拒絕放款。")

    return optimal_threshold


if __name__ == "__main__":
    evaluate_risk_metrics()