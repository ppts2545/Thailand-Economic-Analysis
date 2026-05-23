# Thai Equity Market Alpha Research
### Systematic Search for Predictable Returns in the SET Index (2000–2025)

---

## Executive Summary

This project systematically searched for exploitable alpha in the Thai Stock Exchange (SET) using 25 years of weekly data, XGBoost machine learning, and NLP sentiment signals. The search covered market-level return prediction, sector rotation, and news sentiment — testing over 70 features across multiple model architectures.

**Key finding:** A single global signal — the EEM (Emerging Markets ETF) weekly return lag — is the most consistent predictor of SET returns. IC = +0.11, statistically significant (p < 0.05), stable across the full 2000–2025 period.

**Final system (NB20):** EEM L/flat + Gold diversifier + Risk Parity + Volatility Targeting + Drawdown Control achieves **net Sharpe +0.78, MaxDD −19%, Calmar 0.54** vs SET B&H (Sharpe +0.19, MaxDD −48%).

**Bottom line:**
- Complex ML models (XGBoost, LASSO) add marginal value over this simple signal
- News NLP is promising in recent years (2020+) but data is too sparse to be conclusive
- Sector rotation has real IC but is destroyed by transaction costs at weekly frequency
- Portfolio construction (risk parity + vol targeting) is the most impactful enhancement — reduces MaxDD from −28% to −19%
- Break-even TC for the full system: ~0.30% one-way

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
| **A: EEM L/flat + sector tilt** *(NB17)* | **+1.13** | **+0.62** | **+9.7%** | -40% | **7/8** |
| Rule L/flat (NB16) | +0.81 | +0.58 | +7.2% | -38% | 10/10 |
| Rule L/S (NB16) | +0.73 | +0.40 | +6.6% | -38% | 10/10 |
| 1-Feature XGB | — | +0.08 | +1.3% | — | — |
| SET Buy & Hold | +0.39 | +0.39 | +6.6% | -48% | — |

**Combination adds value:** EEM L/flat + sector momentum tilt improves Sharpe from +0.58 to +0.62 and return from +7.2% to +9.7% with manageable TC (~7%/yr).  
**Rule beats ML:** 1-feature XGB Sharpe = +0.08 vs rule Sharpe = +0.40. Occam's razor wins.  
**Break-even TC:** L/flat survives up to ~0.50% one-way.

**Why simple beats complex:** With IC = 0.11, the signal is linear and robust. XGBoost on a single feature with 1,300 training points overfits the tails. The rule-based approach captures the full IC without estimation variance.

---

## 4b. Full System Construction — NB20

*Full analysis in NB20.*

Four-layer systematic portfolio combining all proven components:

| Layer | Mechanism | Key Parameter |
|-------|-----------|---------------|
| L1 — Signal | EEM L/flat on SET; Gold always long | eem_ret_d_lag1 > 0 |
| L2 — Risk Parity | 1/vol weights (52-wk rolling), monthly rebalance | Gold gets ~2× weight vs SET |
| L3 — Vol Targeting | Scale to 10% annual target vol (cap 2×) | 12-wk rolling vol |
| L4 — DD Control | Halve exposure when DD < −15%; restore at −10% | Circuit-breaker |

### Performance vs Benchmarks (2004–2025, net of 0.1% TC)

| Strategy | Net Sharpe | Ann Return | Max DD | Calmar |
|----------|-----------|------------|--------|--------|
| SET Buy & Hold | +0.186 | +5.0% | −47.7% | 0.11 |
| EEM L/flat (NB16) | +0.564 | +8.2% | −42.2% | 0.19 |
| EEM L/flat + Gold 50/50 (NB18) | +0.814 | +10.3% | −27.9% | 0.37 |
| **Full System (NB20)** | **+0.778** | **+10.4%** | **−19.1%** | **0.54** |

### Layer Isolation (what each layer adds)

| System | Sharpe | Max DD |
|--------|--------|--------|
| L1 (Signal + Gold) | +0.820 | −27.9% |
| L1 + L2 (Risk Parity) | +0.835 | −28.2% |
| L1 + L2 + L3 (Vol Targeting) | **+0.874** | **−20.1%** |
| L1 + L2 + L3 + L4 (DD Control) | +0.778 | −19.1% |

**Key finding:** Vol targeting (L3) is the single most impactful layer — reduces MaxDD from −28% to −20% while improving Sharpe. Drawdown control (L4) costs ~0.1 Sharpe (TC drag from position changes) but shaves a further 1pp off MaxDD.

**Break-even TC:** Full system survives up to ~0.30% one-way.  
**Calmar improvement:** 0.54 vs 0.37 (EEM+Gold) — 46% better risk-adjusted return per unit drawdown.  
**Regime stability:** Positive Sharpe in all sub-periods (2004–2009: 0.91, 2010–2014: 0.77, 2015–2019: 0.57, 2020–2025: 0.82).

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
| Medium | Add Gold directional signal (yield_curve_slope) to NB20 system | L3 vol-targeted Gold + signal could lift Sharpe > 0.90 |
| Medium | Live paper-trading pilot | Validate EEM signal stability post-2025 |
| Low | Stock-level monthly momentum (NB19 follow-up) | Monthly reduces TC from 41% → ~10%/yr |

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
| eda/17 | EEM + sector tilt | EEM L/flat combined with sector momentum |
| eda/18 | Gold & FX signals | IC grid, multi-asset portfolio (SET+Gold+FX) |
| eda/19 | Stock cross-section | 54-stock momentum IC, weekly L/S (TC kills it) |
| **eda/20** | **System construction** | **Full system: risk parity + vol target + DD control** |
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
*Author: poommy | Last updated: 2026-05-23 | NB20 added: Full system construction*
