import pandas as pd
import numpy as np


def clean_application(df):
    """
    對 application_train / application_test 做基礎清理。
    設計原則：所有操作都不依賴 TARGET，train/test 可以共用同一份邏輯，避免資料洩漏。
    """
    df = df.copy()

    # ---------- 1. DAYS_EMPLOYED 異常值處理 ----------
    # 365243 是「無業/退休」的編碼錯誤，不是真正的天數
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(np.int8)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # ---------- 2. DAYS_BIRTH 轉成年齡（正數、好解讀）----------
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(np.float32)

    # DAYS_EMPLOYED 也順便轉成「已就業年資」（正數），缺失保留 NaN
    df['YEARS_EMPLOYED'] = (-df['DAYS_EMPLOYED'] / 365.25).astype(np.float32)

    # ---------- 3. OWN_CAR_AGE 結構性缺失 ----------
    # 缺失 = 沒有車，不是「不知道車齡」，填 0 是合理的哨兵值
    # FLAG_OWN_CAR 本身會保留在資料中，模型仍能區分「沒車的0」和「有車剛買的0」
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)

    # ---------- 4. AMT_INCOME_TOTAL 極端值截尾 ----------
    # 只處理極端的單筆離群值(1.17億)，用 99.9 百分位數截尾，比直接刪列穩健
    income_cap = df['AMT_INCOME_TOTAL'].quantile(0.999)
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=income_cap)

    # ---------- 5. 房屋建物欄位：AVG/MODE/MEDI 三胞胎 ----------
    housing_cols = [c for c in df.columns if c.endswith('_MODE') or c.endswith('_MEDI')]
    # 保留前先算「這一列房屋資訊總共缺了幾個」，這本身可能是有意義的特徵
    housing_avg_cols = [c for c in df.columns if c.endswith('_AVG')]
    df['HOUSING_INFO_MISSING_COUNT'] = df[housing_avg_cols].isnull().sum(axis=1).astype(np.int16)
    df = df.drop(columns=housing_cols)

    # ---------- 6. 類別型缺失獨立成 'Missing' ----------
    cat_cols = df.select_dtypes(include='category').columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].cat.add_categories('Missing').fillna('Missing')

    return df
def add_financial_ratios(df):
    """
    建構核心金融風控比率特徵。
    全部用 df 既有欄位相除，注意分母為 0 或 NaN 的狀況要處理，避免產生 inf。
    """
    df = df.copy()

    # ---------- 負債 / 還款壓力比率 ----------
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']  # 近似貸款期數
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # ---------- 家庭 / 個人負擔 ----------
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_PER_PERSON'] = df['AMT_ANNUITY'] / df['CNT_FAM_MEMBERS']

    # ---------- 年齡 / 就業年資相關 ----------
    # 就業年資佔年齡的比例，越高代表越早/越穩定進入職場
    df['EMPLOYED_TO_AGE_RATIO'] = df['YEARS_EMPLOYED'] / df['AGE_YEARS']

    # ---------- EXT_SOURCE 聚合統計量（這題最強的一組特徵）----------
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    # 缺了幾個外部評分，本身也是訊號(徵信資料越完整的人通常風險資訊越透明)
    df['EXT_SOURCE_MISSING_COUNT'] = df[ext_cols].isnull().sum(axis=1).astype(np.int8)

    # ---------- 處理相除後可能產生的 inf ----------
    ratio_cols = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM',
                  'CREDIT_GOODS_RATIO', 'INCOME_PER_PERSON', 'ANNUITY_PER_PERSON',
                  'EMPLOYED_TO_AGE_RATIO']
    for col in ratio_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    return df
def aggregate_bureau(bureau_df):
    """
    把 bureau.csv 從「一筆信貸一列」聚合成「一個客戶一列」。
    回傳的 DataFrame 以 SK_ID_CURR 為 key，欄位全部加上 BUREAU_ 前綴方便辨識來源。
    """
    bureau_df = bureau_df.copy()

    # 數值型欄位：算多種統計量
    num_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',                       # 總共開過幾筆信貸
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],          # 逾期天數嚴重度
        'DAYS_CREDIT': ['min', 'max', 'mean'],           # 信貸開始時間分佈(距今天數，負值)
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],   # 信貸到期時間
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],        # 信貸總額
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],          # 目前尚欠總額
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean'],       # 目前逾期金額
        'CNT_CREDIT_PROLONG': 'sum',                     # 展延次數總和
    })
    # 攤平 multi-index 欄名，例如 ('AMT_CREDIT_SUM','sum') -> 'BUREAU_AMT_CREDIT_SUM_SUM'
    num_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in num_agg.columns]

    # 類別型欄位：CREDIT_ACTIVE 的各狀態筆數 (one-hot 後加總)
    status_dummies = pd.get_dummies(bureau_df[['SK_ID_CURR', 'CREDIT_ACTIVE']],
                                     columns=['CREDIT_ACTIVE'], prefix='BUREAU_STATUS')
    status_agg = status_dummies.groupby('SK_ID_CURR').sum()

    bureau_agg = num_agg.join(status_agg, how='left')

    # 信用額度使用率：欠款 / 總額度，衡量負債壓力
    bureau_agg['BUREAU_CREDIT_UTILIZATION'] = (
        bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM']
    ).replace([np.inf, -np.inf], np.nan)

    bureau_agg = bureau_agg.reset_index()  # SK_ID_CURR 從 index 變回一般欄位，方便之後 merge
    return bureau_agg


def merge_bureau_features(app_df, bureau_agg):
    """把聚合好的 bureau 特徵 merge 回主表，用 left join 保留主表所有客戶。"""
    merged = app_df.merge(bureau_agg, on='SK_ID_CURR', how='left')

    # merge 之後產生的缺失值代表「這人在 bureau 完全沒有信貸紀錄」
    # 用一個旗標記錄，計數類欄位補 0，比率/金額類欄位保留 NaN(讓模型自己學缺失的意義)
    merged['BUREAU_HAS_RECORD'] = merged['BUREAU_SK_ID_BUREAU_COUNT'].notnull().astype(np.int8)
    merged['BUREAU_SK_ID_BUREAU_COUNT'] = merged['BUREAU_SK_ID_BUREAU_COUNT'].fillna(0)

    return merged
def aggregate_previous_application(prev_df):
    """
    聚合 previous_application.csv：一個客戶過去在 Home Credit 申請貸款的歷史。
    """
    prev_df = prev_df.copy()

    # 數值型統計量
    num_agg = prev_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',                    # 過去總共申請幾次
        'AMT_ANNUITY': ['mean', 'max'],
        'AMT_APPLICATION': ['mean', 'max', 'sum'],
        'AMT_CREDIT': ['mean', 'max', 'sum'],
        'DAYS_DECISION': ['min', 'max', 'mean'],   # 過去申請距今天數
        'CNT_PAYMENT': ['mean', 'max'],            # 過去分期期數
    })
    num_agg.columns = ['PREV_' + '_'.join(col).upper() for col in num_agg.columns]

    # 核准狀態：拒絕次數是這張表最關鍵的訊號
    status_dummies = pd.get_dummies(prev_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']],
                                     columns=['NAME_CONTRACT_STATUS'], prefix='PREV_STATUS')
    status_agg = status_dummies.groupby('SK_ID_CURR').sum()

    prev_agg = num_agg.join(status_agg, how='left')

    # 拒絕率：被拒次數 / 總申請次數，比絕對次數更有可比性
    prev_agg['PREV_REFUSAL_RATE'] = (
        prev_agg.get('PREV_STATUS_Refused', 0) / prev_agg['PREV_SK_ID_PREV_COUNT']
    ).replace([np.inf, -np.inf], np.nan)

    prev_agg = prev_agg.reset_index()
    return prev_agg


def merge_previous_application(app_df, prev_agg):
    merged = app_df.merge(prev_agg, on='SK_ID_CURR', how='left')
    merged['PREV_HAS_RECORD'] = merged['PREV_SK_ID_PREV_COUNT'].notnull().astype(np.int8)
    merged['PREV_SK_ID_PREV_COUNT'] = merged['PREV_SK_ID_PREV_COUNT'].fillna(0)
    return merged
def aggregate_installments(inst_df):
    """
    installments_payments.csv：每一期實際還款 vs 應繳金額。
    核心訊號：實繳金額不足(短繳)、實繳時間比應繳時間晚(逾期繳款)。
    """
    inst_df = inst_df.copy()

    # 短繳金額：應繳 - 實繳，正值代表繳不足
    inst_df['PAYMENT_DIFF'] = inst_df['AMT_INSTALMENT'] - inst_df['AMT_PAYMENT']
    # 逾期天數：實際繳款日 - 應繳款日，正值代表遲繳
    inst_df['DAYS_PAST_DUE'] = inst_df['DAYS_ENTRY_PAYMENT'] - inst_df['DAYS_INSTALMENT']
    inst_df['DAYS_PAST_DUE'] = inst_df['DAYS_PAST_DUE'].clip(lower=0)  # 只看遲繳，提早繳不算負值

    agg = inst_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'PAYMENT_DIFF': ['mean', 'max', 'sum'],
        'DAYS_PAST_DUE': ['mean', 'max'],
        'AMT_PAYMENT': ['mean', 'sum'],
    })
    agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in agg.columns]
    agg = agg.reset_index()
    return agg


def aggregate_pos_cash(pos_df):
    """POS_CASH_balance.csv：分期/現金貸款的月結狀態，核心訊號是 DPD(逾期天數)。"""
    pos_df = pos_df.copy()

    agg = pos_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'SK_DPD': ['mean', 'max'],           # 逾期天數
        'SK_DPD_DEF': ['mean', 'max'],       # 逾期天數(排除小額容忍)
        'CNT_INSTALMENT_FUTURE': 'mean',     # 剩餘未繳期數
    })
    agg.columns = ['POS_' + '_'.join(col).upper() for col in agg.columns]
    agg = agg.reset_index()
    return agg


def aggregate_credit_card(cc_df):
    """credit_card_balance.csv：信用卡月結單，核心訊號是額度使用率跟提領現金行為。"""
    cc_df = cc_df.copy()

    cc_df['CC_UTILIZATION'] = (
        cc_df['AMT_BALANCE'] / cc_df['AMT_CREDIT_LIMIT_ACTUAL']
    ).replace([np.inf, -np.inf], np.nan)

    agg = cc_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'AMT_BALANCE': ['mean', 'max'],
        'CC_UTILIZATION': ['mean', 'max'],
        'SK_DPD': ['mean', 'max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'sum'],  # ATM 提現，通常是財務吃緊訊號
    })
    agg.columns = ['CC_' + '_'.join(col).upper() for col in agg.columns]
    agg = agg.reset_index()
    return agg


def merge_generic(app_df, agg_df, prefix):
    """通用 merge 函式，自動處理 HAS_RECORD 旗標跟 count 欄位補 0。"""
    merged = app_df.merge(agg_df, on='SK_ID_CURR', how='left')
    count_col = f'{prefix}_SK_ID_PREV_COUNT'
    merged[f'{prefix}_HAS_RECORD'] = merged[count_col].notnull().astype(np.int8)
    merged[count_col] = merged[count_col].fillna(0)
    return merged