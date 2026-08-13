import sys
import os
# 不管從哪個資料夾執行這支腳本，都能正確找到專案根目錄
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import pandas as pd
import numpy as np

from src.utils import reduce_mem_usage
from src.features import (
    clean_application, add_financial_ratios,
    aggregate_bureau, merge_bureau_features,
    aggregate_previous_application, merge_previous_application,
    aggregate_installments, aggregate_pos_cash, aggregate_credit_card,
    merge_generic,
)


def build_full_features(app_df, data_dir='../data'):
    """
    完整特徵工程 pipeline，train/test 都呼叫這個函式，確保處理邏輯一致。
    app_df: 已讀入且 reduce_mem_usage 過的 application_train 或 application_test。
    """
    df = clean_application(app_df)
    df = add_financial_ratios(df)

    # ---------- bureau ----------
    bureau = pd.read_csv(f'{data_dir}/bureau.csv')
    bureau = reduce_mem_usage(bureau, verbose=False)
    bureau_agg = aggregate_bureau(bureau)
    del bureau; gc.collect()
    df = merge_bureau_features(df, bureau_agg)
    del bureau_agg; gc.collect()

    # ---------- previous_application ----------
    prev = pd.read_csv(f'{data_dir}/previous_application.csv')
    prev = reduce_mem_usage(prev, verbose=False)
    prev_agg = aggregate_previous_application(prev)
    del prev; gc.collect()
    df = merge_previous_application(df, prev_agg)
    del prev_agg; gc.collect()

    # ---------- installments_payments ----------
    inst = pd.read_csv(f'{data_dir}/installments_payments.csv')
    inst = reduce_mem_usage(inst, verbose=False)
    inst_agg = aggregate_installments(inst)
    del inst; gc.collect()
    df = merge_generic(df, inst_agg, 'INSTAL')
    del inst_agg; gc.collect()

    # ---------- POS_CASH_balance ----------
    pos = pd.read_csv(f'{data_dir}/POS_CASH_balance.csv')
    pos = reduce_mem_usage(pos, verbose=False)
    pos_agg = aggregate_pos_cash(pos)
    del pos; gc.collect()
    df = merge_generic(df, pos_agg, 'POS')
    del pos_agg; gc.collect()

    # ---------- credit_card_balance ----------
    cc = pd.read_csv(f'{data_dir}/credit_card_balance.csv')
    cc = reduce_mem_usage(cc, verbose=False)
    cc_agg = aggregate_credit_card(cc)
    del cc; gc.collect()
    df = merge_generic(df, cc_agg, 'CC')
    del cc_agg; gc.collect()

    return df


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

    for name in ['application_train', 'application_test']:
        print(f'處理 {name} ...')
        raw = pd.read_csv(os.path.join(DATA_DIR, f'{name}.csv'))
        raw = reduce_mem_usage(raw)
        full = build_full_features(raw, data_dir=DATA_DIR)
        print(f'{name} 最終 shape:', full.shape)
        full.to_parquet(os.path.join(DATA_DIR, f'{name}_full.parquet'), index=False)
        del raw, full
        gc.collect()
    print('全部完成')