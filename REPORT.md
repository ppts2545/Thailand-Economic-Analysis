# Thai Equity Market Alpha Research
### Systematic Search for Predictable Returns in the SET Index (2000–2025)

---

## Executive Summary

This project systematically searched for exploitable alpha in the Thai Stock Exchange (SET) using 25 years of weekly data, XGBoost machine learning, and NLP sentiment signals. The search covered market-level return prediction, sector rotation, and news sentiment — testing over 70 features across multiple model architectures.

**Key finding:** A single global signal — the EEM (Emerging Markets ETF) weekly return lag — is the most consistent predictor of SET returns. IC = +0.11, statistically significant (p < 0.05), stable across the full 2000–2025 period.

**Bottom line:**
- Complex ML models (XGBoost, LASSO) add marginal value over this simple signal
- News NLP is promising in recent years (2020+) but data is too sparse to be conclusive
- Sector rotation has real IC but is destroyed by transaction costs at weekly frequency
- A simple EEM-lag rule-based strategy is the most practical implementation

---

## 1. Problem & Motivation

The SET index is one of Asia's most accessible emerging markets but is relatively understudied in systematic/quantitative research. The key questions:

1. Can macro and global market features predict SET weekly returns out-of-sample?
2. Does sector rotation carry exploitable cross-sectional signal after transaction costs?
3. Does Thai financial news sentiment carry incremental predictive power?

**Data pipeline:**
- **Price/market features:** 53 weekly features (2000–2025), cleaned via a rigorous leakage audit (NB07)
- **Sectors:** 7 SET sectors (BANK, ENERGY, ICT, COMMERCE, HEALTH, PROPERTY, FOOD), weekly returns
- **NLP:** ~1,295 Thai financial news articles (Google News RSS + GDELT), VADER sentiment scoring
- All features are lagged correctly — no look-ahead bias

---

## 2. Signal Discovery: What Actually Works

### 2.1 EEM Lag (Strongest Signal Found)

| Metric | Value |
|--------|-------|
| Signal | `eem_ret_d_lag1` — EEM's most recent daily return |
| Spearman IC | +0.11 |
| IC t-test p | < 0.05 |
| Sign-stable | 2000–2025 (positive in >55% of 52-week rolling windows) |
| ICIR (NB08) | 1.47 |

**Why this works:** Global EM risk sentiment (proxied by EEM) propagates into SET with a lag due to reporting delays, time-zone differences, and the slower reaction of Thai institutional investors to global macro shifts. This is a well-documented spillover effect in EM finance literature.

**Strategy (EEM Rule L/S):** Long SET when `eem_ret_d_lag1 > 0`, short when `< 0`. Full results in NB16.

### 2.2 Supporting Signals (Weaker, Used in ML Models)

| Feature | IC | Note |
|---------|-----|------|
| `sp500_ret_w` | ~0.05 | Contemporaneous — limited practical use |
| `SET_index_rvol_4w` | ~0.04 | Volatility regime |
| `sp500_rvol_4w` | ~0.04 | Global vol regime |
| `us_10yr_treasury_ret_w` | ~0.03 | Risk-off/on |

### 2.3 What Does NOT Work

| Feature/Approach | Why Discarded |
|-----------------|---------------|
| Annual macro (GDP, CPI) | 12-week publication delay, too slow for weekly signals |
| `th_uncertainty` | Bootstrap test: not statistically significant (p > 0.50) |
| Oil prices (`oil_ret_w`) | Inconsistent sign, low IC |
| VIX level | Regime feature — useful for conditioning, not as direct signal |
| NLP sentiment (2007–2022) | Coverage < 30% of weeks — insufficient data |

---

## 3. Model Architecture & Results

### 3.1 Market-Level Prediction (XGBoost, 11 Features)

**Setup:** Walk-forward CV, expanding window, 52-week folds, 2003–2025 OOS.

| Model | DirAcc | IC | p-value | ICIR |
|-------|--------|----|---------|------|
| Ridge (11 feats) | 52.8% | +0.071 | 0.165 | — |
| **XGB-Pruned (11 feats)** | **54.3%** | **+0.079** | **0.033** | **1.47** |
| Null predictor | 50.0% | 0 | — | — |

**Verdict:** Weak but statistically significant. XGB-Pruned is the best market-level model.

### 3.2 NLP Enhancement

Adding VADER sentiment features from Thai financial news:

| Target | +NLP DirAcc | No-NLP | Δ DirAcc | Δ Sharpe |
|--------|-------------|--------|----------|----------|
| SET 1W | 62.0% | 58.4% | **+3.6%** | **+0.38** |
| Gold 1W | 65.2% | 65.2% | 0.0% | +0.20 |
| USD/THB 1W | 60.4% | 58.4% | **+2.0%** | +0.28 |

**Caveat:** NLP coverage is only 27–37% in the 2020+ era. Results will improve with denser news data.

### 3.3 Sector Rotation (Cross-Sectional)

**Setup:** Panel (date × sector), cross-sectional excess return target, XGBoost Regressor + Ranker.

| Configuration | Gross Sharpe | Net Sharpe (TC) | Verdict |
|--------------|-------------|-----------------|---------|
| Weekly L/S (NB13) | +0.28 | **-1.76** | TC kills it |
| Monthly L/S B1a (NB15) | +0.35 | -0.24 | Borderline |
| Bi-monthly L/S B1a (NB15) | +0.48 | +0.17 | Marginally viable |
| **Long-only monthly B2 (NB15)** | **+0.78** | **+0.69** | **Best sector variant** |

**Root cause of failure:** IC ~0.026 is too weak to overcome 4 × 0.1% × 52 = 20.8%/yr TC drag in weekly L/S. Long-only monthly (TC ~1.6%/yr) is the only viable variant — but closely tracks the equal-weight benchmark (Sharpe 0.70).

---

## 4. EEM Signal Strategy — Detailed Results

*Full analysis in NB16.*

The simplest viable strategy: **sign(eem_ret_d_lag1) determines SET position each week.**

| Variant | Gross Sharpe | Net Sharpe | Ann Return | Max DD | Score |
|---------|-------------|------------|------------|--------|-------|
| **Rule L/flat** | **+0.81** | **+0.58** | **+7.2%** | -38% | **10/10** |
| Rule L/S | +0.73 | +0.40 | +6.6% | -38% | 10/10 |
| Magnitude-scaled | +0.46 | +0.38 | +3.4% | -23% | — |
| 1-Feature XGB | — | +0.08 | +1.3% | — | — |
| SET Buy & Hold | +0.39 | +0.39 | +6.6% | -48% | — |

**Rule beats ML:** 1-feature XGB Sharpe = +0.08 vs rule Sharpe = +0.40. Occam's razor wins.  
**Break-even TC:** L/S survives up to ~0.30% one-way; L/flat survives up to ~0.50% one-way.

**Why simple beats complex:** With IC = 0.11, the signal is informative but the relationship is linear. XGBoost on a single feature with 1,300 training points tends to overfit the tails. The rule-based approach captures the full IC without variance from model estimation.

---

## 5. Key Lessons & Limitations

### What we learned

1. **TC is the dominant cost at weekly frequency.** Any L/S strategy requires gross Sharpe > 0.5 just to break even after 0.1% TC — very few features achieve this.

2. **Global spillover > local macro.** EEM, S&P 500 lags, and US Treasury yields are far more predictive of SET than Thai-specific macro data. Thailand is a price-taker in global risk sentiment.

3. **NLP needs data density.** The +3.6% DirAcc improvement from NLP is real but fragile — it rests on ~250 news-covered weeks out of 1,300. With denser coverage, this could be the strongest signal.

4. **Sector cross-section exists but is marginal.** IC ~0.026 with p = 0.034 is statistically detectable but not economically exploitable at weekly L/S frequency. Monthly long-only rotation captures the underlying momentum.

### Limitations

- **Single-country focus:** Results specific to SET; spillover effects may differ in other EM markets
- **NLP coverage gap:** 77% of weeks (2000–2020) have zero news articles — sentiment analysis is effectively a 2020+ signal
- **TC assumption:** 0.1% one-way is conservative for institutional trading; retail costs could be 2–5× higher
- **Regime instability:** The EEM-SET relationship may weaken if Thailand decouples from global EM (e.g., during idiosyncratic political events)

---

## 6. Next Steps

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| High | Collect more NLP data (Bangkok Post, Reuters API, 2015–2025) | Close the data gap; validate +3.6% DirAcc at scale |
| High | Monthly-horizon model (aggregate to 4-week returns) | Higher SNR; EEM signal may reach IC > 0.15 |
| Medium | EEM signal combined with sector rotation | L/flat sector tilt on EEM-positive weeks |
| Medium | Regime-conditional strategy (bull/crisis-aware) | EEM signal stronger in normal regime (NB16) |
| Low | Multi-asset extension (Gold, USD/THB) | NLP showed +0.20 Sharpe on Gold |

---

## 7. Notebook Index

| Notebook | Type | Description |
|----------|------|-------------|
| eda/01–05 | EDA | Data overview, NLP validation, comprehensive EDA |
| eda/06 | Alpha research | Feature IC screening, signal validation |
| eda/07 | Pipeline audit | Leakage check, unified_weekly_clean.csv |
| eda/08 | Robustness | Feature pruning, model simplification |
| eda/09 | Signal robustness | Stability hardening, permutation tests |
| eda/10 | Backtest | Walk-forward PnL, XGB-Pruned |
| eda/11 | Long/short | True L/S backtest (corrected always-long bias) |
| eda/12 | Multi-target | SET + Gold + USD/THB joint model |
| eda/13 | Sector alpha | Cross-sectional sector rotation |
| eda/14 | Sector NLP | Sector-specific news sentiment |
| eda/15 | Low-TC sector | Monthly & long-only sector rotation variants |
| **eda/16** | **EEM signal** | **Primary deliverable: EEM lag strategy** |
| model/01–06 | Models | OLS, LASSO, XGBoost, NLP-enhanced |

---

## 8. Data & Reproducibility

```
data/processed/unified_weekly_clean.csv   # 1,322 weeks × 72 features (clean)
data/processed/sector_weekly.csv          # 7 SET sectors, weekly returns
data/processed/sentiment_weekly.csv       # NLP features (10 cols, lagged)
data/processed/sector_sentiment_weekly.csv # Sector NLP features
src/nlp_sentiment.py                      # Full NLP pipeline
scripts/fetch_sector_data.py              # Sector data fetcher
```

All feature engineering uses strict look-ahead prevention (shift-1 minimum).  
Walk-forward CV uses expanding training windows with no future data leakage.

---

*Research period: 2000–2025 | Language: Python 3.13 | Key libraries: XGBoost, pandas, scipy, VADER*  
*Author: poommy | Last updated: 2026-05-23*
