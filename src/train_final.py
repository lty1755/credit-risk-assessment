import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# %%
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

# 讀入 Optuna 找到的最佳參數
with open(os.path.join(BASE_DIR, 'best_params.json')) as f:
    best_params = json.load(f)
print('使用參數:', best_params)

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_final = np.zeros(len(X))
fold_aucs_final = []
models_final = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=3000,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **best_params,   # 展開 Optuna 找到的 learning_rate, num_leaves 等參數
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_COLS,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds_final[val_idx] = val_pred

    fold_auc = roc_auc_score(y_val, val_pred)
    fold_aucs_final.append(fold_auc)
    models_final.append(model)
    print(f'Fold {fold+1} AUC: {fold_auc:.5f} (best_iteration: {model.best_iteration_})')

overall_auc_final = roc_auc_score(y, oof_preds_final)
print(f'\n平均 Fold AUC: {np.mean(fold_aucs_final):.5f} (+/- {np.std(fold_aucs_final):.5f})')
print(f'整體 OOF AUC: {overall_auc_final:.5f}')

print('\n--- 三個版本總結 ---')
print('Logistic Regression baseline: 0.76788')
print('LightGBM 預設參數:            0.78384')
print(f'LightGBM Optuna 調參後:        {overall_auc_final:.5f}')

# 存起來，之後 Stage 4(風控指標)、Stage 5(SHAP) 都要用
np.save(os.path.join(DATA_DIR, 'oof_preds_final.npy'), oof_preds_final)
train[['SK_ID_CURR', 'TARGET']].assign(oof_pred=oof_preds_final).to_csv(
    os.path.join(DATA_DIR, 'oof_predictions.csv'), index=False
)
print('已存 OOF 預測結果，供後續風控指標分析使用')

# %%
# ---------- 對 Kaggle 測試集產出預測，用於 Public Leaderboard 提交 ----------
test = pd.read_parquet(os.path.join(DATA_DIR, 'application_test_full.parquet'))
print('test shape:', test.shape)

X_test = test[feature_cols].copy()
for col in CATEGORICAL_COLS:
    X_test[col] = X_test[col].astype('category')

# 用 5 個 fold 模型分別預測，取平均(bagging)，比單一模型的預測更穩健
test_preds = np.zeros(len(X_test))
for model in models_final:
    test_preds += model.predict_proba(X_test)[:, 1] / len(models_final)

submission = pd.DataFrame({
    'SK_ID_CURR': test['SK_ID_CURR'],
    'TARGET': test_preds,
})
submission_path = os.path.join(DATA_DIR, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f'submission.csv 已存至: {submission_path}')
print(submission.head())
print('\n預測機率分佈:')
print(submission['TARGET'].describe())