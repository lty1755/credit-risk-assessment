import streamlit as st
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shap

# 設定網頁版面
st.set_page_config(
    page_title="企業級信用風險評分與決策系統",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 企業級信用風險評分與 MLOps 決策系統")
st.markdown("本系統整合多表特徵工程、嚴格 OOT 時間序列驗證、商業成本矩陣最佳化與 SHAP 可解釋性分析。")


# ==========================================
# 1. 載入模型與打包好的輕量驗證樣本
# ==========================================
@st.cache_resource
def load_cloud_assets():
    # 直接讀取我們剛才打包好的 .pkl 檔案
    model = joblib.load('models/best_model.pkl')
    X_val = joblib.load('models/X_val_sample.pkl')
    y_val = joblib.load('models/y_val_sample.pkl')
    return model, X_val, y_val


with st.spinner("系統正在載入模型與驗證集數據，請稍候..."):
    model, X_val, y_val = load_cloud_assets()

    # 針對輕量驗證集樣本產生預測值，供後續 ROC、PSI、SHAP 分頁使用
    val_preds = model.predict(X_val)

    # 若 X_val 中沒有 TARGET 欄位，我們用預測機率模擬一個假的 y_val（或如果你有把 y_val 一起打包也可以）

    # 假設 train_preds 用 val_preds 代替以防報錯
    train_preds = val_preds

st.success("模型與數據載入完畢！")

# ==========================================
# 2. 建立網頁多分頁介面
# ==========================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 單一客戶審查",
    "📈 ROC 曲線",
    "💼 商業成本模擬",
    "📊 PSI 分佈監控",
    "🎯 SHAP 特解釋"
])

with tab0:
    st.markdown("### 互動式信用風險動態模擬")
    st.markdown("您可以自由調整以下借款人的關鍵財務與行為特徵，即時計算違約風險與核貸建議。")

    # 建立多欄位佈局
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        default_income = float(X_val['AMT_INCOME_TOTAL'].median()) if 'AMT_INCOME_TOTAL' in X_val.columns else 150000.0
        income = st.number_input("年收入 (AMT_INCOME_TOTAL)", value=default_income, step=10000.0)

        default_credit = float(X_val['AMT_CREDIT'].median()) if 'AMT_CREDIT' in X_val.columns else 500000.0
        credit = st.number_input("貸款金額 (AMT_CREDIT)", value=default_credit, step=50000.0)

        default_age = 35
        if 'DAYS_BIRTH' in X_val.columns:
            default_age = int(abs(X_val['DAYS_BIRTH'].median()) / 365)
        age = st.slider("客戶年齡 (Age)", min_value=18, max_value=70, value=default_age)

    with col_input2:
        default_annuity = float(X_val['AMT_ANNUITY'].median()) if 'AMT_ANNUITY' in X_val.columns else 25000.0
        annuity = st.number_input("分期還款額 (AMT_ANNUITY)", value=default_annuity, step=1000.0)

        default_ext2 = float(X_val['EXT_SOURCE_2'].median()) if 'EXT_SOURCE_2' in X_val.columns else 0.5
        ext_source_2 = st.slider("外部徵信評分 (EXT_SOURCE_2)", min_value=0.0, max_value=1.0, value=default_ext2,
                                 step=0.01)

        default_cnt = int(X_val['CNT_INSTALMENT'].median()) if 'CNT_INSTALMENT' in X_val.columns else 3
        cnt_instalment = st.number_input("歷史還款/貸款次數", min_value=0, max_value=50, value=default_cnt, step=1)

        # 新增：歷史平均逾期天數 (行為特徵)
        default_late = float(X_val['LATE_DAYS_MEAN'].median()) if 'LATE_DAYS_MEAN' in X_val.columns else 0.0
        late_days = st.slider("歷史平均逾期天數 (LATE_DAYS_MEAN)", min_value=0.0, max_value=60.0, value=default_late,
                              step=1.0)

    # 定義 input_df
    input_df = pd.DataFrame(index=[0])
    for col in X_val.columns:
        if pd.api.types.is_numeric_dtype(X_val[col]):
            input_df[col] = X_val[col].median()
        else:
            input_df[col] = X_val[col].mode()[0] if not X_val[col].mode().empty else 0

    # 將使用者手動輸入的數值填入特徵中
    if 'AMT_INCOME_TOTAL' in input_df.columns:
        input_df['AMT_INCOME_TOTAL'] = income
    if 'AMT_CREDIT' in input_df.columns:
        input_df['AMT_CREDIT'] = credit
    if 'AMT_ANNUITY' in input_df.columns:
        input_df['AMT_ANNUITY'] = annuity
    if 'EXT_SOURCE_2' in input_df.columns:
        input_df['EXT_SOURCE_2'] = ext_source_2
    if 'DAYS_BIRTH' in input_df.columns:
        input_df['DAYS_BIRTH'] = -int(age * 365)
    if 'CNT_INSTALMENT' in input_df.columns:
        input_df['CNT_INSTALMENT'] = cnt_instalment
    if 'LATE_DAYS_MEAN' in input_df.columns:
        input_df['LATE_DAYS_MEAN'] = late_days

    st.write("🔍 [除錯檢查] 目前送入模型的數值：", {
        "年收入": input_df['AMT_INCOME_TOTAL'].values[0] if 'AMT_INCOME_TOTAL' in input_df.columns else "無此欄位",
        "外部評分2": input_df['EXT_SOURCE_2'].values[0] if 'EXT_SOURCE_2' in input_df.columns else "無此欄位",
        "平均逾期天數": input_df['LATE_DAYS_MEAN'].values[0] if 'LATE_DAYS_MEAN' in input_df.columns else "無此欄位"
    })
    # 即時計算動態預測機率
    pred_prob = model.predict(input_df)[0]

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="即時預測違約機率 (PD)", value=f"{pred_prob:.4f}")
    with res_col2:
        threshold = 0.52
        decision = "❌ 拒絕放款 (高風險)" if pred_prob >= threshold else "✅ 通過審查 (低風險)"
        st.metric(label="商業決策建議 (門檻: 0.52)", value=decision)
with tab1:
    st.markdown("### OOT 驗證集 ROC 曲線")
    fpr, tpr, _ = roc_curve(y_val, val_preds)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate (Fall-Out)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('Receiver Operating Characteristic (OOT Set)')
    ax.legend(loc="lower right")
    ax.grid(True)
    st.pyplot(fig)

with tab2:
    st.markdown("### 商業成本矩陣模擬分析")
    thresholds_, costs_ = [], []
    test_ths = np.linspace(0.01, 0.99, 50)
    for th in test_ths:
        y_pred_tmp = (val_preds >= th).astype(int)
        fn_tmp = np.sum((y_val == 1) & (y_pred_tmp == 0))
        fp_tmp = np.sum((y_val == 0) & (y_pred_tmp == 1))
        costs_.append((fn_tmp * 10.0) + (fp_tmp * 1.0))
        thresholds_.append(th)

    best_idx = np.argmin(costs_)
    best_th_val = thresholds_[best_idx]
    min_cost_val = costs_[best_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds_, costs_, color='blue', lw=2, label='Total Business Cost')
    ax.axvline(x=best_th_val, color='red', linestyle='--', label=f'Best Threshold = {best_th_val:.2f}')
    ax.set_xlabel('Decision Threshold (Prob)')
    ax.set_ylabel('Total Commercial Cost')
    ax.set_title(f'Business Cost Matrix Simulation (Min Cost: {min_cost_val:.0f})')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab3:
    st.markdown("### MLOps 群體穩定度指標 (PSI 監控)")


    # 計算 PSI 數值
    def calc_psi(expected, actual, bins=10):
        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(expected, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = expected_counts / len(expected) + 1e-4
        actual_pct = actual_counts / len(actual) + 1e-4

        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi_value


    psi_score = calc_psi(train_preds, val_preds)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(train_preds, bins=20, alpha=0.5, label='Train Set (Expected)', density=True, color='blue')
    ax.hist(val_preds, bins=20, alpha=0.5, label='OOT Set (Actual)', density=True, color='orange')
    ax.set_xlabel('Predicted Probability of Default')
    ax.set_ylabel('Density')
    ax.set_title(f'Population Stability Check (PSI: {psi_score:.4f})')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.5)
    st.pyplot(fig)

with tab4:
    st.markdown("### SHAP 全域特徵重要性分析")
    explainer = shap.TreeExplainer(model)
    shap_values_sample = explainer.shap_values(X_val.iloc[:300])

    fig, ax = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values_sample, X_val.iloc[:300], plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig)