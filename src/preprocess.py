import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def process_bureau(bureau_path="data/bureau.csv"):
    try:
        bureau = pd.read_csv(bureau_path)
    except FileNotFoundError:
        return None

    orig_cols = bureau.columns.tolist()
    bureau_dummies = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'], dummy_na=True)

    numeric_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max'],
        'DAYS_ENDDATE_FACT': ['max'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'sum', 'mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'sum'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'AMT_ANNUITY': ['max', 'sum', 'mean']
    }

    cat_cols = [col for col in bureau_dummies.columns if col not in orig_cols and col != 'SK_ID_CURR']
    cat_aggregations = {col: ['mean', 'sum'] for col in cat_cols if col in bureau_dummies.columns}
    aggregations = {**numeric_aggregations, **cat_aggregations}

    valid_aggregations = {col: agg for col, agg in aggregations.items() if col in bureau_dummies.columns}
    bureau_agg = bureau_dummies.groupby('SK_ID_CURR').agg(valid_aggregations)
    bureau_agg.columns = pd.Index(["BUREAU_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    bureau_agg['BUREAU_LOAN_COUNT'] = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count()[
        'SK_ID_BUREAU']

    return bureau_agg


def process_previous_application(prev_path="data/previous_application.csv"):
    print("正在讀取並聚合 previous_application.csv...")
    try:
        prev = pd.read_csv(prev_path)
    except FileNotFoundError:
        print("【提示】找不到 previous_application.csv，略過。")
        return None

    orig_cols = prev.columns.tolist()

    target_cats = ['NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CLIENT_TYPE']
    existing_cats = [col for col in target_cats if col in prev.columns]

    prev_dummies = pd.get_dummies(prev, columns=existing_cats, dummy_na=True)

    numeric_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    cat_cols = [col for col in prev_dummies.columns if col not in orig_cols and col != 'SK_ID_CURR']
    cat_aggregations = {col: ['mean', 'sum'] for col in cat_cols if col in prev_dummies.columns}
    aggregations = {**numeric_aggregations, **cat_aggregations}
    valid_aggregations = {col: agg for col, agg in aggregations.items() if col in prev_dummies.columns}

    prev_agg = prev_dummies.groupby('SK_ID_CURR').agg(valid_aggregations)
    prev_agg.columns = pd.Index(["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count()['SK_ID_PREV']

    print(f"previous_application 聚合完成！維度: {prev_agg.shape}")
    return prev_agg


def process_installments(path="data/installments_payments.csv"):
    print("正在讀取並聚合 installments_payments.csv (還款紀錄金礦)...")
    try:
        ins = pd.read_csv(path)
    except FileNotFoundError:
        print("【提示】找不到 installments_payments.csv，略過。")
        return None

    # 衍生特徵：確保產出網頁要的 LATE_DAYS 欄位
    ins['LATE_DAYS'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['LATE_DAYS'] = ins['LATE_DAYS'].apply(lambda x: x if x > 0 else 0)

    ins['AMT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['AMT_DIFF'] = ins['AMT_DIFF'].apply(lambda x: x if x > 0 else 0)

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'LATE_DAYS': ['max', 'mean', 'sum'],  # <--- 這裡直接計算 LATE_DAYS 的 mean/max/sum
        'AMT_DIFF': ['max', 'mean', 'sum'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean']
    }

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)

    # 產出欄位會變成 INS_LATE_DAYS_MEAN 等，我們同時保留原本命名與額外對應一個 LATE_DAYS_MEAN 方便網頁抓取
    ins_agg.columns = pd.Index(["INS_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg = ins_agg.reset_index()

    # 額外賦值一個乾淨的欄位名稱給網頁互動介面使用
    ins_agg['LATE_DAYS_MEAN'] = ins_agg['INS_LATE_DAYS_MEAN']

    print(f"installments 聚合完成！維度: {ins_agg.shape}")
    return ins_agg


def process_pos_cash(path="data/POS_CASH_balance.csv"):
    print("正在讀取並聚合 POS_CASH_balance.csv...")
    try:
        pos = pd.read_csv(path)
    except FileNotFoundError:
        print("【提示】找不到 POS_CASH_balance.csv，略過。")
        return None

    pos_dummies = pd.get_dummies(pos, columns=['NAME_CONTRACT_STATUS'], dummy_na=True)
    orig_cols = pos.columns.tolist()
    cat_cols = [col for col in pos_dummies.columns if col not in orig_cols and col != 'SK_ID_CURR']

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean'],
        'CNT_INSTALMENT': ['max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['max', 'mean']
    }
    cat_aggregations = {col: ['mean', 'sum'] for col in cat_cols}
    aggregations = {**aggregations, **cat_aggregations}

    valid_agg = {col: agg for col, agg in aggregations.items() if col in pos_dummies.columns}
    pos_agg = pos_dummies.groupby('SK_ID_CURR').agg(valid_agg)
    pos_agg.columns = pd.Index(["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count()['SK_ID_PREV']

    print(f"POS_CASH 聚合完成！維度: {pos_agg.shape}")
    return pos_agg


def process_credit_card(path="data/credit_card_balance.csv"):
    print("正在讀取並聚合 credit_card_balance.csv...")
    try:
        cc = pd.read_csv(path)
    except FileNotFoundError:
        print("【提示】找不到 credit_card_balance.csv，略過。")
        return None

    cc_dummies = pd.get_dummies(cc, columns=['NAME_CONTRACT_STATUS'], dummy_na=True)
    orig_cols = cc.columns.tolist()
    cat_cols = [col for col in cc_dummies.columns if col not in orig_cols and col != 'SK_ID_CURR']

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'AMT_BALANCE': ['max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['max', 'mean', 'sum'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum']
    }
    cat_aggregations = {col: ['mean', 'sum'] for col in cat_cols}
    aggregations = {**aggregations, **cat_aggregations}

    valid_agg = {col: agg for col, agg in aggregations.items() if col in cc_dummies.columns}
    cc_agg = cc_dummies.groupby('SK_ID_CURR').agg(valid_agg)
    cc_agg.columns = pd.Index(["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count()['SK_ID_PREV']

    print(f"credit_card 聚合完成！維度: {cc_agg.shape}")
    return cc_agg


def load_and_preprocess_data(file_path):
    print("正在載入主檔數據...")
    df = pd.read_csv(file_path)
    print(f"原始主檔大小: {df.shape}")

    # --- 依序整併所有核心附表 ---
    for func, filename in [
        (process_bureau, "data/bureau.csv"),
        (process_previous_application, "data/previous_application.csv"),
        (process_installments, "data/installments_payments.csv"),
        (process_pos_cash, "data/POS_CASH_balance.csv"),
        (process_credit_card, "data/credit_card_balance.csv")
    ]:
        agg_df = func(filename)
        if agg_df is not None:
            df = df.merge(agg_df, on='SK_ID_CURR', how='left')

    # --- 建立金融風控核心特徵 ---
    print("正在建立金融風控衍生特徵...")
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['AGE_YEARS'] = (-df['DAYS_BIRTH']) / 365.25

    # --- 處理類別變數編碼 ---
    print("正在進行類別變數編碼...")
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].dropna().unique()) <= 2:
                df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, dummy_na=True)

    # --- 處理數值型特徵缺失值 ---
    print("正在處理數值型缺失值...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    import re
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]

    print(f"全表整併與預處理後總數據大小: {df.shape}")
    return df