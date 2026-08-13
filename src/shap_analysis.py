import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# %%
# ---------- 重新訓練單一模型(不做 CV,用全部資料訓練，SHAP 分析只需要一個代表性模型)----------
train = pd.read_parquet(os.path.join(DATA_DIR, 'application_train_full.parquet'))

CATEGORICAL_COLS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
]
feature_cols = [c for c in train.columns if c not in ['SK_ID_CURR', 'TARGET']]
X = train[feature_cols].copy()
for col in CATEGORICAL_COLS:
    X[col] = X[col].astype('category')
y = train['TARGET']

neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / pos

with open(os.path.join(BASE_DIR, 'best_params.json')) as f:
    best_params = json.load(f)

# 用 80% 訓練，保留 20% 當作之後 SHAP 個體解釋的樣本池
from sklearn.model_selection import train_test_split
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_estimators=2000,   # 這裡不用 early stopping,直接用調參找到的合理樹數量附近的值
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    **best_params,
)
model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)
print('模型訓練完成')

# %%
# ---------- SHAP 計算 ----------
# TreeExplainer 對樹模型做了特別優化,計算速度遠比通用 Explainer 快
explainer = shap.TreeExplainer(model)

# SHAP 值計算對整個訓練集會很慢，先用抽樣(例如 5000 筆)做全局分析
sample_idx = np.random.RandomState(42).choice(len(X_holdout), size=5000, replace=False)
X_sample = X_holdout.iloc[sample_idx]

shap_values = explainer.shap_values(X_sample)
print('SHAP 值計算完成, shape:', np.array(shap_values).shape)
# %%
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='shap')

# ---------- 全局解釋：Summary Plot ----------
plt.figure()
shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '..', 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Summary Plot 已存檔: shap_summary.png')

# %%
# 也印出文字版排名，方便直接寫進報告
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_sample.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False)

print('\nTop 20 重要特徵 (依 SHAP 平均絕對值排序):')
print(importance_df.head(20).to_string(index=False))
# %%
# ---------- 挑一個高風險客戶做個體解釋 ----------
# 用 X_sample 對應的預測機率，挑一個被判定風險最高的真實違約案例
pred_proba_sample = model.predict_proba(X_sample)[:, 1]
y_sample_true = y_holdout.iloc[sample_idx].values

# 篩選：模型判定機率高、且真的是違約客戶(TP,預測對的高風險案例最有說服力)
high_risk_mask = (pred_proba_sample > 0.7) & (y_sample_true == 1)
high_risk_candidates = np.where(high_risk_mask)[0]
print(f'符合條件(機率>0.7 且真實違約)的候選人數: {len(high_risk_candidates)}')

# 挑其中機率最高的一個做示範
target_idx = high_risk_candidates[np.argmax(pred_proba_sample[high_risk_candidates])]
print(f'選定案例: 預測違約機率 = {pred_proba_sample[target_idx]:.4f}')

# %%
# ---------- Waterfall Plot ----------
explanation = shap.Explanation(
    values=shap_values[target_idx],
    base_values=explainer.expected_value,
    data=X_sample.iloc[target_idx],
    feature_names=X_sample.columns.tolist(),
)

plt.figure()
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '..', 'shap_waterfall_example.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Waterfall Plot 已存檔: shap_waterfall_example.png')

# %%
# ---------- 文字版拒貸歸因報告 ----------
case_shap = pd.Series(shap_values[target_idx], index=X_sample.columns)
top_risk_factors = case_shap.sort_values(ascending=False).head(5)  # 推高風險的前5個因素
top_protective_factors = case_shap.sort_values(ascending=True).head(3)  # 降低風險的前3個因素

print('\n=== 拒貸歸因報告 ===')
print(f'客戶預測違約機率: {pred_proba_sample[target_idx]:.2%}')
print('\n主要風險推升因素:')
for feat, val in top_risk_factors.items():
    actual_value = X_sample.iloc[target_idx][feat]
    print(f'  - {feat}: 數值={actual_value}, SHAP貢獻=+{val:.4f}')

print('\n主要風險緩解因素:')
for feat, val in top_protective_factors.items():
    actual_value = X_sample.iloc[target_idx][feat]
    print(f'  - {feat}: 數值={actual_value}, SHAP貢獻={val:.4f}')