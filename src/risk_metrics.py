import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows 中文顯示
plt.rcParams['axes.unicode_minus'] = False

# %%
oof_df = pd.read_csv(os.path.join(DATA_DIR, 'oof_predictions.csv'))
y_true = oof_df['TARGET'].values
y_pred = oof_df['oof_pred'].values

# ---------- ROC-AUC & Gini ----------
auc = roc_auc_score(y_true, y_pred)
gini = 2 * auc - 1
print(f'AUC: {auc:.5f}')
print(f'Gini 係數: {gini:.5f}')

# %%
# ---------- KS 統計量 ----------
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# fpr = 累積「好客戶」被誤判比例的反面關係，實際上:
# tpr(=recall) 是累積抓到的壞客戶比例，fpr 是累積誤傷的好客戶比例
ks_values = tpr - fpr
ks_stat = np.max(ks_values)
ks_threshold = thresholds[np.argmax(ks_values)]

print(f'\nKS 統計量: {ks_stat:.4f} ({ks_stat*100:.2f})')
print(f'KS 最大值發生在機率門檻: {ks_threshold:.4f}')

# 畫 KS 曲線
plt.figure(figsize=(8, 5))
plt.plot(thresholds, tpr, label='累積壞客戶捕獲率 (TPR)', color='red')
plt.plot(thresholds, fpr, label='累積好客戶誤傷率 (FPR)', color='blue')
plt.axvline(ks_threshold, color='gray', linestyle='--', alpha=0.5)
plt.fill_between(thresholds, tpr, fpr, alpha=0.2, color='green',
                  where=(tpr >= fpr))
plt.xlabel('預測違約機率門檻')
plt.ylabel('累積比例')
plt.title(f'KS 曲線 (KS = {ks_stat:.4f})')
plt.legend()
plt.gca().invert_xaxis()  # threshold 由高到低比較符合業務直覺(高分->低分)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '..', 'ks_curve.png'), dpi=150)
plt.show()
print('KS 曲線已存檔: ks_curve.png')
# %%
# ---------- 最佳切點：成本矩陣法 ----------
# 假設情境（可依實際業務調整）：
#   錯放一個違約客戶(False Negative,漏抓壞客戶) 的損失 = 貸款金額的一定比例(本金+利息損失)
#   錯拒一個好客戶(False Positive,誤傷好客戶)     的損失 = 機會成本(少賺的利息)
# 這裡用一個常見假設:違約損失 是 誤拒機會成本的 10 倍 (業界常見量級,實際比例應依真實資料調整)
COST_FN = 10   # 放款給會違約的人，損失權重
COST_FP = 1    # 拒絕好客戶，損失權重(機會成本)

thresholds_to_test = np.linspace(0.01, 0.99, 99)
total_costs = []

for t in thresholds_to_test:
    y_pred_binary = (y_pred >= t).astype(int)
    fn = ((y_pred_binary == 0) & (y_true == 1)).sum()  # 該拒絕卻核准了(漏放)
    fp = ((y_pred_binary == 1) & (y_true == 0)).sum()  # 該核准卻拒絕了(誤傷)
    cost = fn * COST_FN + fp * COST_FP
    total_costs.append(cost)

total_costs = np.array(total_costs)
best_idx = np.argmin(total_costs)
best_threshold_cost = thresholds_to_test[best_idx]

print(f'成本矩陣最佳切點: {best_threshold_cost:.3f}')
print(f'該切點下總成本: {total_costs[best_idx]:.0f}')

# 對照組:如果完全不篩選(全部核准)的成本
baseline_cost = (y_true == 1).sum() * COST_FN
print(f'完全不篩選(全部核准)的成本: {baseline_cost:.0f}')
print(f'成本降低: {(1 - total_costs[best_idx]/baseline_cost)*100:.1f}%')

# 畫成本曲線
plt.figure(figsize=(8, 5))
plt.plot(thresholds_to_test, total_costs)
plt.axvline(best_threshold_cost, color='red', linestyle='--',
            label=f'最佳切點 = {best_threshold_cost:.3f}')
plt.xlabel('預測違約機率門檻')
plt.ylabel('總成本(加權)')
plt.title('成本矩陣 - 尋找最佳決策切點')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, '..', 'cost_curve.png'), dpi=150)
plt.show()
# %%
# ---------- 信用評分轉換 (PDO 方法) ----------
# 業界慣例參數:base_score=600 對應 base_odds=50(好壞比 50:1)，PDO=20(odds翻倍,加20分)
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

# odds = 好客戶機率 / 壞客戶機率 = (1-PD) / PD
# 為避免 PD=0 造成除以零，做極小值保護
pd_clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
odds = (1 - pd_clipped) / pd_clipped
credit_score = offset + factor * np.log(odds)

oof_df['credit_score'] = credit_score
print(oof_df['credit_score'].describe())

# %%
# ---------- 風險分級 Tier A-D：改用分位數切分，確保跟實際分數分佈對齊 ----------
quantiles = oof_df['credit_score'].quantile([0.25, 0.50, 0.75]).values
print('分數四分位數:', quantiles)

def assign_tier_by_quantile(score, q):
    if score >= q[2]:
        return 'A (低風險，前25%高分客戶)'
    elif score >= q[1]:
        return 'B (中低風險)'
    elif score >= q[0]:
        return 'C (中高風險，提高利率)'
    else:
        return 'D (高風險，拒絕放款)'

oof_df['risk_tier'] = oof_df['credit_score'].apply(lambda s: assign_tier_by_quantile(s, quantiles))

tier_summary = oof_df.groupby('risk_tier').agg(
    人數=('TARGET', 'count'),
    實際違約率=('TARGET', 'mean'),
).reset_index()
tier_summary['佔比'] = (tier_summary['人數'] / len(oof_df) * 100).round(2)

# 依風險排序顯示(不要依字母排序)
tier_order = ['A (低風險，前25%高分客戶)', 'B (中低風險)', 'C (中高風險，提高利率)', 'D (高風險，拒絕放款)']
tier_summary['risk_tier'] = pd.Categorical(tier_summary['risk_tier'], categories=tier_order, ordered=True)
tier_summary = tier_summary.sort_values('risk_tier')

print('\n風險分級摘要(分位數切分):')
print(tier_summary)

oof_df.to_csv(os.path.join(DATA_DIR, 'oof_predictions_scored.csv'), index=False)
print('\n已存含評分/分級的完整結果: oof_predictions_scored.csv')