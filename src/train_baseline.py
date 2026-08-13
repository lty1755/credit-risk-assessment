import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# %%
# ---------- 讀取特徵表 ----------
train = pd.read_parquet(os.path.join(DATA_DIR, 'application_train_full.parquet'))
print('train shape:', train.shape)

# SK_ID_CURR 只是識別碼，TARGET 是標籤，其餘都是特徵
feature_cols = [c for c in train.columns if c not in ['SK_ID_CURR', 'TARGET']]
X = train[feature_cols]
y = train['TARGET']

print('特徵數:', len(feature_cols))
print('類別型特徵:', X.select_dtypes(include='category').columns.tolist())

# %%
print(X.dtypes.value_counts())
print()
print('object 型態欄位:')
print(X.select_dtypes(include='object').columns.tolist())
# %%
CATEGORICAL_COLS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
]
NUMERIC_COLS = [c for c in feature_cols if c not in CATEGORICAL_COLS]
print('類別型:', len(CATEGORICAL_COLS), '數值型:', len(NUMERIC_COLS))

# ---------- Logistic Regression 前處理 ----------
from sklearn.impute import SimpleImputer

X_num = X[NUMERIC_COLS].copy()
X_cat = X[CATEGORICAL_COLS].copy()

# 數值型：中位數填補缺失值
num_imputer = SimpleImputer(strategy='median')
X_num_imputed = pd.DataFrame(
    num_imputer.fit_transform(X_num), columns=NUMERIC_COLS, index=X.index
)

# 類別型：先填 'Missing'，再 One-Hot
X_cat_filled = X_cat.fillna('Missing')
X_cat_encoded = pd.get_dummies(X_cat_filled, columns=CATEGORICAL_COLS, dummy_na=False)
print('One-Hot 後類別特徵欄數:', X_cat_encoded.shape[1])

# 合併數值 + One-Hot 類別
X_baseline = pd.concat([X_num_imputed, X_cat_encoded], axis=1)
print('Baseline 最終特徵數:', X_baseline.shape[1])

# %%
# ---------- StratifiedKFold 5-fold 交叉驗證 ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_baseline))      # out-of-fold 預測，每筆資料只會被預測一次(在它所屬的驗證 fold)
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_baseline, y)):
    X_tr, X_val = X_baseline.iloc[train_idx], X_baseline.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 標準化：Logistic Regression 對特徵尺度敏感，一定要做
    # 注意：scaler 只能用訓練 fold fit，驗證 fold 只能 transform，避免資料洩漏
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # 處理類別不平衡，等同於樣本加權
        random_state=42,
    )
    model.fit(X_tr_scaled, y_tr)

    val_pred = model.predict_proba(X_val_scaled)[:, 1]
    oof_preds[val_idx] = val_pred

    fold_auc = roc_auc_score(y_val, val_pred)
    fold_aucs.append(fold_auc)
    print(f'Fold {fold+1} AUC: {fold_auc:.5f}')

overall_auc = roc_auc_score(y, oof_preds)
print(f'\n平均 Fold AUC: {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})')
print(f'整體 OOF AUC: {overall_auc:.5f}')
# %%
import lightgbm as lgb

# LightGBM 需要把類別欄位轉回 category dtype 才能被正確辨識(前面提過 parquet 讀回來型態跑掉了)
X_lgbm = X.copy()
for col in CATEGORICAL_COLS:
    X_lgbm[col] = X_lgbm[col].astype('category')

# 不平衡處理：負樣本數 / 正樣本數
neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / pos
print(f'scale_pos_weight: {scale_pos_weight:.3f}')

oof_preds_lgbm = np.zeros(len(X_lgbm))
fold_aucs_lgbm = []
models = []  # 存起來給之後 SHAP 分析用

for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgbm, y)):
    X_tr, X_val = X_lgbm.iloc[train_idx], X_lgbm.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=32,
        max_depth=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_COLS,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds_lgbm[val_idx] = val_pred

    fold_auc = roc_auc_score(y_val, val_pred)
    fold_aucs_lgbm.append(fold_auc)
    models.append(model)
    print(f'Fold {fold + 1} AUC: {fold_auc:.5f} (best_iteration: {model.best_iteration_})')

overall_auc_lgbm = roc_auc_score(y, oof_preds_lgbm)
print(f'\n平均 Fold AUC: {np.mean(fold_aucs_lgbm):.5f} (+/- {np.std(fold_aucs_lgbm):.5f})')
print(f'整體 OOF AUC: {overall_auc_lgbm:.5f}')
print(f'\n對比 Baseline (Logistic Regression): {overall_auc:.5f}')
print(f'AUC 提升: {overall_auc_lgbm - overall_auc:.5f}')