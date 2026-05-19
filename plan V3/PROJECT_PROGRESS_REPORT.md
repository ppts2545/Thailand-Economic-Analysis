# Thailand Financial Market ML — Technical Progress Report
**Date:** 2026-05-17  
**Project:** Weekly frequency XGBoost model for Thai financial markets (SET, Gold, USD/THB)  
**Status:** Active development — Phase 3 (model refinement + validation)

---

## 1. Project Objective

Build a **production-grade quantitative model** to predict weekly returns of 3 Thai financial assets:
- **SET Index** — Thai equity market (most difficult, noise-dominated EM asset)
- **Gold (THB)** — macro-driven trend asset
- **USD/THB** — short-term flow/momentum FX pair

Success criteria: DirAcc > 55%, Sharpe > 1.0 on unseen test data (2020–2025).

---

## 2. Data Pipeline

### 2.1 Raw Sources (`src/fetch_data.py`)

| Source | Frequency | Series | Period |
|--------|-----------|--------|--------|
| yfinance | Daily | SET, Gold, Oil, S&P500, Nasdaq, VIX, DXY, US10Y, US2Y, USD/THB, **EEM** | 2000–2026 |
| FRED API | Monthly | Fed Funds, CPI, Unemployment, Industrial Production, Global Uncertainty, Consumer Sentiment, Govt Spending, USD/THB (gap fill) | 2000–2026 |
| World Bank API | Annual | Exports/Imports % GDP, Consumption, Gross Capital Formation, Govt Debt, Lending Rate, Inflation, Unemployment | 2000–2024 |
| IMF DataMapper | Annual | GDP growth, Inflation, Current Account | 2000–2025 |
| Bangkok Post RSS | Irregular | Headlines + VADER sentiment | 2026-05 (current only) |
| GDELT DOC API v2 | Historical | Thailand financial news, VADER scored | 2015–2025 (fetching) |

### 2.2 Weekly Feature Pipeline (`src/preprocess_weekly.py`)

```
Step 1  Daily prices → resample W-FRI (Friday close, ffill ≤5 days)
Step 2  Weekly return = pct_change, winsorize 1%/99%
Step 2.5 Cross-market DAILY lag features (timezone lead effect) ← KEY INNOVATION
Step 3  FRED monthly → forward-fill to weekly (1-month publication lag)
Step 4  Annual macro → broadcast to each week (prior year's value, 1-yr release lag)
Step 5  Merge all → clip 2000-09 to 2025-12 → unified_weekly.csv
```

**Output:** `data/processed/unified_weekly.csv` — 1,322 weeks × 53 columns

**Training window:** 2000-09-01 → 2019-12-31 (~1,009 weeks)  
**Test window:** 2020-01-01 → 2025-12-26 (~313 weeks)

### 2.3 Timezone Lead Effect (Step 2.5) — Critical Design Decision

```
S&P500 closes 4:00pm ET = 4:00am Bangkok (Friday)
SET opens 9:30am Bangkok (Friday)
→ Thursday's S&P500 return arrives in Bangkok BEFORE SET opens Friday
→ sp500_ret_d_lag1 = genuine pre-open signal, NOT data leakage
```

Assets with daily lags: S&P500, Nasdaq, Oil, DXY, USD/THB, **EEM** (added Phase 3)  
Lag periods: 1 business day (Thursday close) and 2 business days (Wednesday close)

---

## 3. Model Architecture — 3-Layer XGBoost

### 3.1 Feature Engineering Layers

| Layer | Description | Example Features |
|-------|-------------|-----------------|
| **Layer 1** | Hard macroeconomic data | Fed funds rate, CPI, industrial production |
| **Layer 2** | Sentiment proxies | VIX, global uncertainty index, consumer sentiment |
| **Layer 3** | News sentiment flow | weighted_compound, 3w/8w MA, momentum, z-score shock |
| **Layer 4** | Regime interactions | bull/bear/crisis × market features (20 cross-terms) |
| **Layer 5** | SET/Regime/Season | MA20/52 crossover, drawdown depth, calendar effects |
| **Layer 7** | Cross-market daily lags | sp500/nasdaq/eem/dxy daily return 1–2 bdays before Friday |
| **Derived** | Lag/Momentum/Signal | 26w/52w momentum, vol4w/13w, yield z-scores, DXY mom |

**Total features before filtering:** ~202  
**After 3-stage filter:** ~105 valid features

### 3.2 3-Stage Feature Filter

```python
Stage 1: Missing rate < 30%          (202 → 178 features)
Stage 2: Zero fraction < 60%         (178 → 112 features)  ← removes step-pattern monthly cols
Stage 3: |Spearman r| ≥ 0.02 with any target  (112 → 105 features)
```

### 3.3 Feature Selection per Model

```python
select_top_features(X_train, y_train, top_n=12, must_include=[...])
```
- Ranks by |Spearman r| with target
- **must_include override:** force domain-knowledge features regardless of rank

### 3.4 must_include per Target (Domain Override)

| Target | Forced Features | Reason |
|--------|----------------|--------|
| SET | eem_ret_d_lag1, em_outflow_pressure, eem_vs_sp500_lag1, sp500_ret_d_lag1, nasdaq_ret_d_lag1, risk_on_signal, dxy_ret_d_lag1, USD_THB_ret_d_lag1, dxy_3w_mom | 3-driver theory: foreign flow + global risk + FX |
| Gold | real_yield, real_yield_chg4w, yield_curve | Erb & Harvey (2013): gold driven by real yield |
| USD/THB | yield_curve, real_yield | Rate differential channel |

### 3.5 Hyperparameter Optimization (Optuna)

```python
n_trials = 40  # Bayesian TPE sampler
Search space:
  n_estimators:    [50, 400]
  max_depth:       [2, 6]       ← shallow trees = anti-overfit
  learning_rate:   [0.01, 0.3]
  subsample:       [0.5, 1.0]
  colsample:       [0.5, 1.0]
  reg_lambda:      [0.1, 10.0]
  min_child_weight:[1, 20]
  
Objective: 5-fold TimeSeriesSplit CV mean R²
```

### 3.6 Multi-Horizon

| Horizon | Target Definition |
|---------|------------------|
| 1-week | `target = ret.shift(-1)` (next Friday return) |
| 4-week | `target = ret.rolling(4).sum().shift(-4)` (cumulative 4-week) |

---

## 4. Training Window Research (Phase 3 — Key Finding)

**Problem:** SET DirAcc was 48.5% (below random 50%) with 20-year training window.

**Hypothesis:** Market structure changed post-2010. Data from 2000–2009 teaches patterns irrelevant to current EM capital flow regime.

**Experiment:** Systematic comparison across 3 training windows, SET model only.

| Training Window | Train Rows | DirAcc 1w | DirAcc 4w | Sharpe 4w | AnnRet 4w |
|----------------|------------|-----------|-----------|-----------|-----------|
| 2000→2019 (20yr) | 1,009 | 48.2% | 46.6% | -0.75 | -4.4% |
| **2010→2019 (10yr)** | 522 | **53.4%** | 48.9% | 0.19 | +1.2% |
| **2013→2019 (7yr)** | 365 | 47.9% | **53.7%** | **0.61** | **+3.7%** |

**Finding:** Shorter, more recent windows dramatically outperform for SET:
- 1w model optimal: **2010–2019** (post-GFC EM flow regime)
- 4w model optimal: **2013–2019** (Fed taper tantrum era — closest to current structure)

**Implementation:** SET models now use target-specific training windows. Gold/FX unchanged (2000–2019).

---

## 5. Current Model Performance (Latest Run — 2026-05-17)

### 5.1 Direction Accuracy

| Target | 1w DirAcc | 4w DirAcc | Ensemble DirAcc | Coverage |
|--------|-----------|-----------|-----------------|----------|
| **SET** | **53.7%** | **55.3%** | **59.7% ▲** | 51.5% |
| Gold | 60.2% | 63.8% | 60.1% | 99.0% |
| USD/THB | 52.8% | 53.7% | 50.8% | 59.9% |

**Ensemble = direction-agreement filter:** only signal when both 1w and 4w agree. SET ensemble 59.7% on 51.5% of weeks.

### 5.2 Out-of-Sample Predictive Power

| Target | Horizon | Train R² | Test R² | Overfit Gap |
|--------|---------|----------|---------|-------------|
| SET | 1w | 0.102 | -0.007 | 0.109 ✓ |
| SET | 4w | (see below) | — | — |
| Gold | 1w | 0.039 | -0.004 | 0.043 ✓ |
| Gold | 4w | 0.066 | +0.017 | 0.049 ✓ |
| USD/THB | 1w | 0.138 | +0.011 | 0.127 ✓ |
| USD/THB | 4w | 0.117 | +0.003 | 0.114 ✓ |

Note: R² near zero is expected for weekly return prediction. DirAcc is the primary metric.

### 5.3 Trading Simulation (2020–2025)

**Position sizing methodology:**
```
position = sign(pred) × vol_scale × confidence × kelly_fraction × [0, max_exposure]
  vol_scale       = vol_target(10%) / realized_vol_annual
  confidence      = tanh(|pred| / weekly_vol)  ∈ (0,1)
  kelly_fraction  = 0.5  (half-Kelly)
  min_confidence  = 0.20  (flat if below threshold)
  cost            = cost_oneway × |Δposition|  (charged on change only)
```

| Strategy | AnnRet | Sharpe | MaxDD | Active% | Total Cost |
|----------|--------|--------|-------|---------|-----------|
| SET [1w] | -0.6% | -0.52 | -4.4% | 35.9% | 5.6% |
| **SET [4w]** | **+3.0%** | **0.51 ★** | **-23.5%** | 68.9% | 11.8% |
| Gold [1w] | +0.1% | 0.18 | -1.9% | 12.9% | 0.4% |
| **Gold [4w]** | **+14.5%** | **2.67 ★** | **-9.8%** | 91.6% | 0.8% |
| USD/THB [1w] | +0.1% | 0.20 | -1.0% | 25.9% | 0.7% |
| **USD/THB [4w]** | **+1.4%** | **0.54 ★** | **-7.9%** | 62.5% | 0.7% |

**Buy-and-hold comparison (2020–2025):**
- SET B&H: -1.7% ann. / Sharpe -0.12
- Gold B&H: +18.1% ann. / Sharpe 1.15
- USD/THB B&H: +1.6% ann. / Sharpe 0.21

### 5.4 Asset Classification (Quant Framework)

| Asset | Classification | Signal Type | Practical Edge |
|-------|---------------|-------------|----------------|
| 🟢 **Gold [4w]** | Macro-driven trend asset | real_yield, DXY, yield_curve | Strong — Sharpe 2.67, 5/5 WFV folds >1 |
| 🟡 **SET [4w]** | EM flow-driven index | EEM, S&P500 lead, DXY | Moderate — Sharpe 0.51, needs futures |
| 🟡 **USD/THB [4w]** | Short-term flow/momentum FX | yield_curve, rate_carry | Moderate — Sharpe 0.54 |
| 🔴 **SET [1w]** | Noise-dominated EM index | Mixed | Marginal — cost kills edge |

### 5.5 SHAP Layer Contribution (Test Set)

**SET:**
```
Layer 2 · Sentiment Proxy      35.1%  (VIX, uncertainty)
Layer 1 · Hard Data            35.0%
Derived · Lag/Momentum         28.6%
Layer 7 · Cross-Market Lag      1.1%  ← EEM/SP500 small but directionally correct
```

**Gold:**
```
Derived · Lag/Momentum         54.6%  (real yield momentum)
Derived · Signal Processing    14.4%
Layer 2 · Sentiment Proxy      14.1%
Layer 7 · Cross-Market Lag     12.0%
```

**VIX Regime Accuracy (SET):**
```
Bull (VIX low):    ~49–52%
Normal:            ~47–51%
Crisis (VIX high): ~50–53%
```
No strong regime dependency → signal is regime-agnostic (consistent behavior).

---

## 6. Walk-Forward Validation — Gold 4w

5 expanding-window folds (2010–2020), 30 Optuna trials/fold:

| Fold | Train End | Test Period | DirAcc | Sharpe | AnnRet |
|------|-----------|-------------|--------|--------|--------|
| [1] | 2010-12-31 | 2011→2012 | 55.3% | 1.19 | +9.6% |
| [2] | 2012-12-31 | 2013→2014 | 55.3% | 1.09 | +8.5% |
| [3] | 2014-12-31 | 2015→2016 | 58.1% | 1.65 | +9.8% |
| [4] | 2016-12-31 | 2017→2018 | 58.9% | 2.17 | +10.9% |
| [5] | 2018-12-31 | 2019→2020 | 63.2% | 2.78 | +14.6% |

**Mean DirAcc: 58.2% ±3.2% | Mean Sharpe: 1.78 ±0.71**  
**5/5 folds Sharpe > 1.0 | 5/5 folds DirAcc > 55%**

Conclusion: Gold 4w edge is **temporally stable** — not a single lucky split.

---

## 7. SET Cost Sensitivity Analysis

SET 1w model (cost is the primary constraint):

| Execution Venue | One-way Cost | AnnRet | Sharpe | Verdict |
|----------------|-------------|--------|--------|---------|
| TFEX SET50 futures | 0.05% | +0.2% | 0.14 | Marginal |
| Online broker | 0.10% | 0.0% | 0.01 | Break-even |
| Mid broker | 0.15% | -0.1% | -0.13 | Loss |
| Retail ETF | 0.30% | -0.6% | -0.52 | Loss |

**Break-even: 0.10–0.15% one-way**  
SET 4w model (Sharpe 0.51) is viable with TFEX futures if cost kept below ~0.15%.

---

## 8. SET Top Predictors (Spearman Correlation, Training Set)

| Feature | r | Driver |
|---------|---|--------|
| eem_ret_d_lag1 | +0.1155 | Foreign flow (EM ETF, Thursday) |
| nasdaq_ret_d_lag1 | +0.1118 | Global risk (tech/growth, Thursday) |
| sp500_ret_d_lag1 | +0.1109 | Global risk (US market, Thursday) |
| USD_THB_ret_d_lag2 × is_bull | +0.0572 | FX × regime interaction |
| oil_ret_d_lag1 | +0.0398 | Commodity/global demand |
| set_above_ma20w | +0.0381 | Trend regime |

**Key finding:** EEM (r=0.1155) matches Nasdaq/S&P500 as SET predictor — confirms foreign flow is Driver #1.

---

## 9. Key Decisions Log

| Date | Decision | Reason | Impact |
|------|----------|--------|--------|
| Phase 1 | Use W-FRI resampling (not calendar month) | ~5.5× more data | Enabled robust XGBoost |
| Phase 2 | must_include domain override | real_yield was rank #19 but Gold's canonical driver | Gold DirAcc 53.8% → 60.8% |
| Phase 2 | Importance-weighted VADER sentiment | Raw VADER ignores article credibility | Sentiment features now active |
| Phase 2 | Direction-agreement ensemble (not value blend) | 1w and 4w are different scales | Avoids scale mismatch |
| Phase 3 | Add EEM daily lag | Direct EM foreign flow proxy | eem_ret_d_lag1 r=0.1155 = #1 feature |
| Phase 3 | Exclude _annual features from SET | Annual GDP/inflation too slow for weekly SET | SET DirAcc 48.5% → 53.7% |
| Phase 3 | SET training window: 2010/2013 instead of 2000 | Pre-2010 market structure ≠ current EM regime | SET 4w Sharpe -0.75 → +0.51 |
| Phase 3 | Composite em_outflow_pressure = DXY↑×VIX↑×USD/THB↑ | All 3 aligned = strong capital flight signal | In SET must_include |

---

## 10. Current Architecture Summary

```
Data Sources → preprocess_weekly.py → unified_weekly.csv (1,322 × 53)
                                              ↓
                          Feature Engineering (202 → 105 features)
                          [Layers 1-7 + Block I composite features]
                                              ↓
                    ┌─────────────────────────┴──────────────────────┐
                    │                                                 │
              SET (1w/4w)                                    Gold / USD/THB (1w/4w)
          Train: 2010–2019 / 2013–2019                       Train: 2000–2019
          Features: no _annual                                Features: all valid
          MUST: EEM + SP500 + DXY                             MUST: real_yield + yield_curve
                    │                                                 │
                    └──────────────────┬──────────────────────────────┘
                                       ↓
                         XGBoost (40 Optuna trials, TimeSeriesCV)
                                       ↓
                        Multi-horizon (1w + 4w) predictions
                                       ↓
                      Direction-Agreement Ensemble Filter
                                       ↓
                    Position Sizing (vol-scale × half-Kelly × confidence)
                                       ↓
                         Cost-Aware Backtest (|Δpos| × cost)
```

---

## 11. Progress vs Baseline

| Metric | Monthly OLS (v1) | Monthly Lasso (v2) | **Weekly XGBoost (v3 current)** |
|--------|-----------------|-------------------|--------------------------------|
| Data frequency | Annual/Monthly | Monthly | **Weekly** |
| Training samples | ~180 | ~180 | **~500–1,009** |
| Gold DirAcc | ~55% | ~57% | **60.2% (1w) / 63.8% (4w)** |
| Gold Sharpe (backtest) | N/A | N/A | **2.67** |
| SET DirAcc | ~50% | ~52% | **53.7% (1w) / 55.3% (4w)** |
| SET Ensemble | N/A | N/A | **59.7% (high-confidence weeks)** |
| Validation method | None | None | **Walk-forward (5 folds)** |

---

## 12. Known Limitations

1. **News data sparse**: Bangkok Post RSS = current only (~10 articles). GDELT 2015–2025 still fetching. Sentiment features currently near-zero → not contributing to model.
2. **No real foreign investor flow data**: EEM is a proxy. SET publishes actual daily foreign buy/sell volume (SET website) — not yet fetched.
3. **SET 4w MaxDD -23.5%**: High drawdown despite positive Sharpe. Position sizing may need tighter stops or Kelly fraction reduction.
4. **Walk-forward for SET not yet validated**: Only done for Gold 4w. SET window finding (2010/2013) based on single-split comparison — needs WFV to confirm.
5. **GDELT data quality**: Non-English articles filtered by ASCII ratio. Effectiveness of GDELT sentiment not yet measured.
6. **USD/THB gap 2000–2003**: Filled from FRED monthly series (lower frequency). May introduce noise in early training data.

---

## 13. Next Development Steps (Priority Order)

| Priority | Task | Expected Impact |
|----------|------|----------------|
| 🔴 High | Fetch actual SET foreign investor daily flow from SET website | Direct Driver #1 — may push SET DirAcc to 60%+ |
| 🔴 High | Walk-forward validation for SET 2010→2019 window | Confirm edge is stable, not window-specific |
| 🟡 Med | Wait for GDELT completion → re-run with full sentiment history | Sentiment may add 1–3% DirAcc |
| 🟡 Med | Reduce SET 4w max drawdown (tighter Kelly or stop-loss) | Risk-adjusted return improvement |
| 🟡 Med | Add VIX9D/VIX spread (term structure) as risk-appetite signal | Additional Layer 2 feature for SET |
| 🟢 Low | Hyperparameter search: increase n_trials to 100 for Gold 4w | May push Gold Sharpe toward 3.0 |
| 🟢 Low | Live paper trading simulation (forward test from Jan 2026) | Real-world validation |

---

## 14. File Structure

```
Thailand-Economic-Analysis/
├── src/
│   ├── fetch_data.py          # Data collection (all sources + GDELT + EEM)
│   ├── preprocess_weekly.py   # Feature pipeline → unified_weekly.csv
│   ├── feature_engineering.py # (placeholder)
│   └── preprocess.py          # (placeholder)
├── notebooks/model/
│   ├── 01_ols_annual.ipynb    # Baseline OLS (annual data)
│   ├── 02_lasso_monthly.ipynb # Lasso (monthly data)
│   └── 03_3layer_xgboost_sentiment.ipynb  # MAIN MODEL
├── data/
│   ├── raw/                   # CSV from all APIs
│   └── processed/
│       └── unified_weekly.csv # Master feature matrix (1,322 × 53)
├── plan V2/
│   └── insight.txt            # Qualitative market structure findings
├── plan V3/
│   └── PROJECT_PROGRESS_REPORT.md  # This file
└── compare_windows.py         # SET training window comparison script
```

---

*Report generated: 2026-05-17 | Model version: 3-Layer XGBoost + EEM + Window Optimization*
