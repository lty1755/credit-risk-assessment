# %%
import sys
import numpy as np
sys.path.append('..')  # 讓 python 找得到 src/utils.py
from src.utils import reduce_mem_usage
import pandas as pd
import gc

# %%
app_train = pd.read_csv('../data/application_train.csv')
app_train = reduce_mem_usage(app_train)

print(app_train.shape)
print(app_train['TARGET'].value_counts(normalize=True))

# %%
# 快速看一下缺失值最嚴重的前 20 欄
missing = app_train.isnull().mean().sort_values(ascending=False)
print(missing.head(20))
# %%
# 檢查 OWN_CAR_AGE 缺失是否等於沒有車
car_check = pd.crosstab(app_train['FLAG_OWN_CAR'], app_train['OWN_CAR_AGE'].isnull())
print(car_check)

# %%
# 檢查 DAYS_EMPLOYED 的異常值
print(app_train['DAYS_EMPLOYED'].describe())
print('\n異常值(365243)筆數:', (app_train['DAYS_EMPLOYED'] == 365243).sum())
print('異常值佔比:', (app_train['DAYS_EMPLOYED'] == 365243).mean())

# 通常會發現這批異常值幾乎都是 NAME_INCOME_TYPE == 'Pensioner'（退休人士）
print(app_train.loc[app_train['DAYS_EMPLOYED'] == 365243, 'NAME_INCOME_TYPE'].value_counts())

# %%
# 檢查 AMT_INCOME_TOTAL 是否有極端離群值
print(app_train['AMT_INCOME_TOTAL'].describe())
print(app_train['AMT_INCOME_TOTAL'].sort_values(ascending=False).head(5))
# %%
from src.features import clean_application

app_train_clean = clean_application(app_train)
print(app_train_clean.shape)
print(app_train_clean[['AGE_YEARS', 'YEARS_EMPLOYED', 'DAYS_EMPLOYED_ANOM', 'HOUSING_INFO_MISSING_COUNT']].describe())

# 確認異常值真的被清掉了
print('清理後 YEARS_EMPLOYED 最大值:', app_train_clean['YEARS_EMPLOYED'].max())
# %%
from src.features import add_financial_ratios

app_train_fe = add_financial_ratios(app_train_clean)
print(app_train_fe.shape)

new_cols = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM',
            'CREDIT_GOODS_RATIO', 'INCOME_PER_PERSON', 'EXT_SOURCE_MEAN']
print(app_train_fe[new_cols].describe())

# 檢查有沒有 inf 殘留
print('inf 檢查:', np.isinf(app_train_fe[new_cols]).sum().sum())

# %%
# 快速驗證：這些新特徵跟 TARGET 的相關性方向合不合理
correlations = app_train_fe[new_cols + ['TARGET']].corr()['TARGET'].sort_values()
print(correlations)
# %%
from src.features import aggregate_bureau, merge_bureau_features
import gc

bureau = pd.read_csv('../data/bureau.csv')
bureau = reduce_mem_usage(bureau)
print('bureau shape:', bureau.shape)
print('bureau 涵蓋幾個不重複客戶:', bureau['SK_ID_CURR'].nunique())

# %%
bureau_agg = aggregate_bureau(bureau)
print('聚合後 shape:', bureau_agg.shape)
print(bureau_agg.head())

# 用完就刪掉原始表釋放記憶體
del bureau
gc.collect()

# %%
app_train_bureau = merge_bureau_features(app_train_fe, bureau_agg)
print('merge 後 shape:', app_train_bureau.shape)
print('BUREAU_HAS_RECORD 分佈:')
print(app_train_bureau['BUREAU_HAS_RECORD'].value_counts(normalize=True))

# 驗證新特徵跟 TARGET 的相關性方向
bureau_cols = [c for c in app_train_bureau.columns if c.startswith('BUREAU_')]
print(app_train_bureau[bureau_cols + ['TARGET']].corr()['TARGET'].sort_values())
# %%
from src.features import aggregate_previous_application, merge_previous_application

prev = pd.read_csv('../data/previous_application.csv')
prev = reduce_mem_usage(prev)
print('prev shape:', prev.shape)

prev_agg = aggregate_previous_application(prev)
print('聚合後 shape:', prev_agg.shape)

del prev
gc.collect()

# %%
app_train_full = merge_previous_application(app_train_bureau, prev_agg)
print('merge 後 shape:', app_train_full.shape)

prev_cols = [c for c in app_train_full.columns if c.startswith('PREV_')]
print(app_train_full[prev_cols + ['TARGET']].corr()['TARGET'].sort_values())
# %%
from src.features import (aggregate_installments, aggregate_pos_cash,
                           aggregate_credit_card, merge_generic)

inst = pd.read_csv('../data/installments_payments.csv')
inst = reduce_mem_usage(inst)
inst_agg = aggregate_installments(inst)
print('installments 聚合後 shape:', inst_agg.shape)
del inst
gc.collect()

app_train_full = merge_generic(app_train_full, inst_agg, 'INSTAL')
del inst_agg
gc.collect()

# %%
pos = pd.read_csv('../data/POS_CASH_balance.csv')
pos = reduce_mem_usage(pos)
pos_agg = aggregate_pos_cash(pos)
print('POS_CASH 聚合後 shape:', pos_agg.shape)
del pos
gc.collect()

app_train_full = merge_generic(app_train_full, pos_agg, 'POS')
del pos_agg
gc.collect()

# %%
cc = pd.read_csv('../data/credit_card_balance.csv')
cc = reduce_mem_usage(cc)
cc_agg = aggregate_credit_card(cc)
print('credit_card 聚合後 shape:', cc_agg.shape)
del cc
gc.collect()

app_train_full = merge_generic(app_train_full, cc_agg, 'CC')
del cc_agg
gc.collect()

# %%
print('最終合併後 shape:', app_train_full.shape)

# 存成 parquet，之後不用每次都重跑整套聚合流程
app_train_full.to_parquet('../data/app_train_full.parquet', index=False)
print('已儲存 parquet')