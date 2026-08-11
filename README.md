Markdown# 🏦 Enterprise Credit Risk Scoring & MLOps Decision System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-green.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Webapp-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

本專案是一個端對端（End-to-End）的**企業級信用風險評分與智慧放款決策系統**，基於 Home Credit Default Risk 競賽數據集打造。專案不僅追求機器學習預測效能（AUC），更深度結合了**金融風控業務邏輯（商業成本矩陣）**與 **MLOps 模型維運監控（PSI）**。

---

## 🌟 核心亮點與工程架構

1. **多重附表關聯特徵工程 (Multi-Table Relational Feature Engineering)**
   - 整合主檔與 5 大核心附表：`bureau.csv`（聯合徵信）、`previous_application.csv`（歷史申貸）、`installments_payments.csv`（分期還款紀錄金礦）、`POS_CASH_balance.csv` 與 `credit_card_balance.csv`。
   - 透過高維聚合（Aggregations）萃取出超過 500 個金融行為特徵。
2. **嚴格的時間序列驗證 (Out-of-Time, OOT Validation)**
   - 捨棄常規隨機切分，依據申請流水號進行嚴格的 **OOT 時間序列切分**（前 80% 訓練、後 20% 驗證），真實模擬銀行面對未來未知客戶的泛化能力。
3. **商業成本矩陣最佳化 (Business Cost Matrix Optimization)**
   - 突破傳統統計學 0.5 截斷點，透過動態模擬尋找**最低總成本放款門檻**，精準平衡「呆帳本金損失 (Type II Error)」與「拒絕好客戶機會成本 (Type I Error)」。
4. **MLOps 模型穩定度監控 (PSI - Population Stability Index)**
   - 導入群體穩定度指標監控訓練集與驗證集的預測分佈，確保模型無資料飄移（Data Drift）現象。
5. **互動式決策與解釋儀表板 (Streamlit + SHAP)**
   - 提供直覺的網頁操作介面，結合 **SHAP 值** 即時視覺化推動單一客戶違約風險的關鍵歸因。

---

## 📂 專案目錄結構

```text
credit-risk-scoring-project/
│
├── data/                       # 數據資料夾 (主檔與各大附表)
├── src/                        # 核心程式碼模組
│   ├── __init__.py
│   ├── preprocess.py           # 5 大附表整併與特徵工程
│   └── train.py                # OOT 訓練、成本矩陣最佳化與 PSI 計算
│
├── app.py                      # Streamlit 互動式決策儀表板
├── notebook.ipynb              # 探索性分析與完整建模流程筆記本
├── requirements.txt            # 專案依賴套件清單
└── README.md                   # 專案說明文件
🚀 快速開始指南
1. 安裝依賴套件 pip install -r requirements.txt
2. 執行模型訓練與評估 python src/train.py
(執行後將自動輸出 OOT 驗證集 AUC、最佳放款門檻與 PSI 監控值)
3. 啟動 Streamlit 互動儀表板 streamlit run app.py
📊 專案成效指標評估維度成效表現說明預測效能 (ROC-AUC)~0.7864在嚴格的 OOT 時間序列驗證下保持高水準泛化能力。
商業決策門檻0.5200透過成本矩陣動態搜尋出的最佳審查截斷點。
模型穩定度 (PSI)0.0004遠低於 0.1 安全標準，具備極佳的 MLOps 上線體質。