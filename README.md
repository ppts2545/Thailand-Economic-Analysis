# Thai Equity Market Alpha Research

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-ML-FF6600" />
  <img src="https://img.shields.io/badge/NLP-VADER_Sentiment-4CAF50" />
  <img src="https://img.shields.io/badge/Data-2000--2025-1976D2" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

Systematic search for predictable returns in the SET index using 25 years of weekly data, XGBoost, and NLP sentiment signals from Thai financial news.

---

## Key Finding

> **EEM lag signal**: EEM's most recent weekly return predicts SET returns the following week.
> IC = +0.11 · p = 0.0002 · stable in 78.7% of 52-week rolling windows (2003–2025)

A rule-based strategy — *long SET when EEM was positive last week, flat otherwise* — achieves **Sharpe +0.58** after 0.1% transaction costs, beating buy-and-hold (Sharpe +0.39) with smaller drawdown (−38% vs −48%).

## Results at a Glance

| Strategy | Sharpe (net TC) | Ann Return | Max DD |
|---|---|---|---|
| **EEM Rule L/flat** *(recommended)* | **+0.58** | +7.2% | −38% |
| EEM Rule L/S | +0.40 | +6.6% | −38% |
| EEM + Sector Tilt | *see NB17* | — | — |
| Sector Long-Only Monthly | +0.69 | +15.1% | −64% |
| SET Buy & Hold | +0.39 | +6.6% | −48% |

→ Full methodology and findings: [REPORT.md](REPORT.md)

---

## What Was Tested

```
Market-level prediction  →  XGBoost on 53 features (IC p=0.033, ICIR=1.47)
NLP sentiment            →  +3.6% DirAcc on SET, +0.38 Sharpe uplift
Sector rotation L/S      →  Real IC (p=0.034) but TC-negative at weekly frequency
Low-TC sector variants   →  Long-only monthly viable (Sharpe +0.69)
EEM lag signal           →  10/10 scorecard — strongest signal in the project
EEM + sector tilt        →  Global signal + sector momentum combined (NB17)
```

## Notebooks

| # | Notebook | Key Result |
|---|---|---|
| eda/06 | [Stable Alpha Research](notebooks/eda/06_stable_alpha_research.ipynb) | EEM lag IC=0.11 confirmed |
| eda/07 | [Leakage Audit](notebooks/eda/07_leakage_audit_pipeline.ipynb) | Clean dataset (1,322×72) |
| eda/08 | [Alpha Validation](notebooks/eda/08_alpha_validation.ipynb) | XGB-Pruned IC p=0.018 |
| eda/09 | [Signal Robustness](notebooks/eda/09_signal_robustness.ipynb) | Permutation tests, 11 features |
| eda/10–12 | [Backtests](notebooks/eda/10_backtest.ipynb) | Walk-forward PnL, L/S, multi-target |
| eda/13 | [Sector Alpha](notebooks/eda/13_sector_alpha.ipynb) | IC significant, TC kills L/S |
| eda/15 | [Low-TC Sector](notebooks/eda/15_low_tc_sector.ipynb) | B2 long-only Sharpe +0.69 |
| **eda/16** | [**EEM Signal**](notebooks/eda/16_eem_signal.ipynb) | **Score 10/10, Sharpe +0.58** |
| eda/17 | [EEM + Sector Tilt](notebooks/eda/17_eem_sector_tilt.ipynb) | Combined strategy |

---

## Project Goal

Thailand is often analyzed in isolation, which misses the bigger picture. This project takes a **quantitative, data-driven approach** — combining global macro signals, ML models, and NLP sentiment to answer:

> **Are there exploitable, statistically significant patterns in SET returns that survive transaction costs?**

### Countries Under Analysis

| Flag | Country | Region | Role in Analysis |
|------|---------|--------|-----------------|
| <img src="https://flagcdn.com/w40/th.png" width="28"/> | **Thailand** | Southeast Asia | Primary subject |
| <img src="https://flagcdn.com/w40/us.png" width="28"/> | United States | North America | Global benchmark, capital flow driver |
| <img src="https://flagcdn.com/w40/cn.png" width="28"/> | China | East Asia | Thailand's largest trade partner |
| <img src="https://flagcdn.com/w40/jp.png" width="28"/> | Japan | East Asia | Regional anchor, FDI source |
| <img src="https://flagcdn.com/w40/de.png" width="28"/> | Germany | Europe | Export-led growth model (European comparison) |
| <img src="https://flagcdn.com/w40/sg.png" width="28"/> | Singapore | Southeast Asia | Regional high-income peer |

> **Planned expansion:** South Korea, India, UK, France, Vietnam, and Brazil — to broaden coverage across Europe, South Asia, and emerging markets.

---

## What We Track

This analysis is built around **9 structural economic factors** that together explain unemployment dynamics and overall economic health:

```
Factor 1  — Consumption          (Household spending % of GDP)
Factor 2  — Interest Rate        (Lending rate, broad money M2)
Factor 3  — Inflation            (CPI, year-over-year %)
Factor 4  — Unemployment         (Total, youth, by sector)
Factor 5  — Business Investment  (Gross capital formation, FDI inflows)
Factor 6  — Trade / Exports      (Exports & imports % of GDP, USD/THB)
Factor 7  — Geopolitical Risk    (VIX, gold, global uncertainty index)
Factor 8  — Technology / Innovation (Nasdaq, AI/automation trends)
Factor 9  — Government Policy    (Govt expenditure & debt % of GDP)
```

All indicators are tracked from **2003 to present** — long enough to capture the 2008 Global Financial Crisis, COVID-19, and post-pandemic recovery across all countries.

---

## Data Pipeline

```
World Bank API   ──►  Labor market indicators (unemployment, sector employment)
                       Economic context (GDP growth, inflation, trade openness)

IMF DataMapper   ──►  GDP growth, inflation, unemployment, current account
                       (multi-country, harmonized annual data)

FRED (St. Louis) ──►  High-frequency monthly data: Fed Funds rate, US CPI,
                       Thailand exchange rate, Bangkok property prices,
                       global economic policy uncertainty

yfinance         ──►  Market signals: SET index, USD/THB, VIX, Gold,
                       S&P 500, Crude Oil, US 10-yr Treasury, Nasdaq

Google Trends    ──►  Job search behavior in Thailand (หางาน, สมัครงาน)
                       Global signals: AI, automation, layoffs

Bangkok Post RSS ──►  English economic headlines → VADER sentiment scoring
```

All raw data is saved to `data/raw/` as CSV files for reproducibility.

---

## Project Structure

```
Thailand-Economic-Analysis/
├── src/
│   ├── fetch_data.py          # All data fetching (World Bank, IMF, FRED, yfinance, etc.)
│   ├── preprocess.py          # Data cleaning and normalization
│   └── feature_engineering.py # Feature construction for modeling
├── data/
│   └── raw/                   # Raw CSVs saved from fetch_data.py
├── notebooks/                 # Exploratory analysis and visualizations
├── .env                       # API keys (FRED_API_KEY) — not committed
├── requirement.txt
└── README.md
```

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd Thailand-Economic-Analysis
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirement.txt

# Set your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
echo "FRED_API_KEY=your_key_here" > .env

# Fetch all data
python src/fetch_data.py
```

---

## Research Questions

1. **Labor market resilience** — Why does Thailand maintain low headline unemployment even during recessions, and how does this compare to Germany or Japan?
2. **Export dependency** — Thailand's trade/GDP ratio (~120%) dwarfs most peers. Is this a strength or vulnerability?
3. **Digital transition** — How do Google Trends for AI/automation in Thailand compare to Singapore and the US? Is the workforce adapting?
4. **Capital flows & currency** — How does the USD/THB respond to Fed rate changes versus how EUR/USD responds?
5. **Geopolitical exposure** — Which countries are most affected by VIX spikes and trade war news?

---

## Data Sources

| Source | Coverage | Access |
|--------|----------|--------|
| [World Bank Open Data](https://data.worldbank.org/) | 200+ countries, annual | Free, no key |
| [IMF DataMapper](https://www.imf.org/external/datamapper/) | 190+ countries, annual | Free, no key |
| [FRED (St. Louis Fed)](https://fred.stlouisfed.org/) | US + global, monthly | Free API key |
| [Yahoo Finance (yfinance)](https://finance.yahoo.com/) | Markets, daily | Free |
| [Google Trends](https://trends.google.com/) | Search interest, weekly | Free (rate-limited) |
| [Bangkok Post RSS](https://www.bangkokpost.com/) | Thai business headlines | Free |
