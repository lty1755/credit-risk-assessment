import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
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

# 調參階段只用 3-fold，速度換取能多試幾組參數
skf_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
train_idx, val_idx = next(iter(skf_tune.split(X, y)))
X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 3000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=CATEGORICAL_COLS,
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_pred)


# %%
study = optuna.create_study(direction='maximize', study_name='home_credit_lgbm')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print('\n最佳 AUC (單 fold):', study.best_value)
print('最佳參數:')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')

# 存起來，之後正式訓練跟報告都要用
import json
with open(os.path.join(BASE_DIR, 'best_params.json'), 'w') as f:
    json.dump(study.best_params, f, indent=2)
print('\n已存到 src/best_params.json')