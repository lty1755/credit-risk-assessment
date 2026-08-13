import pandas as pd
import numpy as np

def reduce_mem_usage(df, verbose=True):
    """
    遍歷每一欄，把數值型欄位 downcast 成能容納資料範圍的最小型態，
    通常可以把記憶體用量降到原本的 40%~60%。
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        # 用 pandas 的型態判斷 API，比直接比較 dtype 穩健
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            # 全部是 NaN 的欄位，min/max 會是 NaN，跳過不處理
            if pd.isna(c_min) or pd.isna(c_max):
                continue

            # 明確轉成 Python float，避免不同 numpy 版本間比較型態衝突
            c_min = float(c_min)
            c_max = float(c_max)

            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:  # float
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('category')
        # 其他型態（如 bool、datetime）保持原樣，不動它

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f'記憶體用量：{start_mem:.2f} MB → {end_mem:.2f} MB '
              f'(降低 {100*(start_mem-end_mem)/start_mem:.1f}%)')
    return df