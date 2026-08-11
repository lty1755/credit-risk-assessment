from train import train_credit_risk_model
from evaluate_risk import evaluate_risk_metrics
from explain_and_score import run_explainability_and_scoring


def main():
    print("==========================================")
    print("  開始執行信用風險風控專案完整流程...     ")
    print("==========================================")

    # 1. 訓練模型
    print("\n[階段 1/3] 模型訓練與評估...")
    model, X_val, y_val, preds = train_credit_risk_model()

    # 2. 計算風控指標
    print("\n[階段 2/3] 計算 KS 統計量與最佳決策截斷點...")
    evaluate_risk_metrics()

    # 3. 執行可解釋性與分級定價
    print("\n[階段 3/3] SHAP 可解釋性歸因與風險分級定價...")
    run_explainability_and_scoring()

    print("\n==========================================")
    print("  全部流程執行完畢！                      ")
    print("==========================================")


if __name__ == "__main__":
    main()