"""
fetch_raw_data.py -- Single source of truth for all raw data

รัน script นี้เพื่ออัปเดตข้อมูลทั้งหมดใน data/raw/
ใช้ใน pipeline: fetch_raw_data.py → NB02 → NB07 → NB08+ (analysis)

Sources:
  Market prices  : Yahoo Finance (via yfinance) -- free, no key
  FRED macro     : St. Louis Fed FRED API       -- free, no key required
  World Bank     : World Bank Open Data API     -- free, no key
  NLP news       : Google News RSS (via fetch_sector_nlp.py)

Usage:
  python3 scripts/fetch_raw_data.py            # update all
  python3 scripts/fetch_raw_data.py --market   # market only
  python3 scripts/fetch_raw_data.py --fred     # FRED only
  python3 scripts/fetch_raw_data.py --macro    # World Bank macro only
"""

import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

START = '2000-01-01'
END   = datetime.today().strftime('%Y-%m-%d')

# ── 1. MARKET DATA (Yahoo Finance) ───────────────────────────────────────────
# Source: https://finance.yahoo.com
# Free, daily OHLCV, adjusted for splits/dividends

MARKET_TICKERS = {
    'SET_index':        '^SET.BK',    # SET index (Bangkok SET)
    'sp500':            '^GSPC',      # S&P 500
    'nasdaq':           '^IXIC',      # NASDAQ Composite
    'gold':             'GC=F',       # Gold futures (COMEX)
    'oil':              'CL=F',       # WTI Crude Oil futures
    'vix':              '^VIX',       # CBOE Volatility Index
    'dxy':              'DX-Y.NYB',   # US Dollar Index
    'USD_THB':          'THB=X',      # USD/THB exchange rate
    'us_10yr_treasury': '^TNX',       # US 10-Year Treasury yield
    'us_2yr_treasury':  '^IRX',       # US 2-Year Treasury yield (13-wk proxy)
    'eem':              'EEM',        # iShares MSCI Emerging Markets ETF
}

def fetch_market_data():
    print('\n── Market Data (Yahoo Finance) ──────────────────────────')
    for name, ticker in MARKET_TICKERS.items():
        out_path = RAW_DIR / f'{name}_market_signals.csv'
        try:
            raw = yf.download(ticker, start=START, end=END,
                              progress=False, auto_adjust=True)
            if raw.empty:
                print(f'  {name:30s} FAILED (empty)')
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                close = raw['Close'].iloc[:, 0]
            else:
                close = raw['Close']

            df = close.dropna().reset_index()
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date']).dt.date
            df.to_csv(out_path, index=False)
            print(f'  {name:30s} OK  {len(df):5d} rows  '
                  f'{df["date"].min()} → {df["date"].max()}')
        except Exception as e:
            print(f'  {name:30s} ERROR: {e}')
        time.sleep(0.5)

# ── 2. FRED DATA (St. Louis Fed) ─────────────────────────────────────────────
# Source: https://fred.stlouisfed.org
# Free API, no key required for basic series (uses pandas_datareader)
# Series IDs: https://fred.stlouisfed.org/series/<ID>

FRED_SERIES = {
    'fred_us_fed_funds_rate':    'FEDFUNDS',     # Federal Funds Rate (monthly)
    'fred_us_cpi_monthly':       'CPIAUCSL',     # US CPI All Items (monthly)
    'fred_us_unemployment':      'UNRATE',       # US Unemployment Rate (monthly)
    'fred_us_consumer_sentiment':'UMCSENT',      # U Michigan Consumer Sentiment
    'fred_us_industrial_prod':   'INDPRO',       # US Industrial Production Index
    'fred_us_govt_spending':     'FGEXPND',      # US Federal Govt Expenditures
    'fred_usd_thb_monthly':      'DEXTHUS',      # USD/THB exchange rate (monthly)
    'fred_th_us_imports':        'BOPGIMP',      # US current account imports (proxy)
    'fred_th_exchange_rate_real':'RBUSBIS',      # BIS real effective exchange rate (broad)
    'fred_th_uncertainty':       'USEPUINDXD',   # US Economic Policy Uncertainty (proxy)
    'fred_global_uncertainty':   'GEPUCURRENT',  # Global Economic Policy Uncertainty
    'fred_th_property_prices':   'CSUSHPINSA',   # Case-Shiller (US proxy, Thailand N/A on FRED)
}

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Fetch a single FRED series via public CSV endpoint (no key required)."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    # FRED CSV columns: observation_date, <SERIES_ID>
    df.columns = ['date', 'value']
    df['date']  = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['series'] = series_id
    return df.dropna(subset=['date', 'value'])

def fetch_fred_data():
    print('\n── FRED Data (St. Louis Fed) ────────────────────────────')
    for name, series_id in FRED_SERIES.items():
        out_path = RAW_DIR / f'{name}.csv'
        try:
            df = fetch_fred_series(series_id)
            df.to_csv(out_path, index=False)
            print(f'  {name:35s} OK  {len(df):5d} rows  '
                  f'{df["date"].min().date()} → {df["date"].max().date()}')
        except Exception as e:
            print(f'  {name:35s} ERROR: {e}')
        time.sleep(1.0)  # polite to FRED

# ── 3. WORLD BANK MACRO DATA (annual) ────────────────────────────────────────
# Source: https://data.worldbank.org
# Free API, no key required
# Indicator codes: https://data.worldbank.org/indicator/<CODE>

WORLDBANK_INDICATORS = {
    'imf_gdp_growth_TH':          'NY.GDP.MKTP.KD.ZG',  # GDP growth (annual %)
    'inflation_TH':               'FP.CPI.TOTL.ZG',     # Inflation CPI (annual %)
    'thailand_unemployment_rate': 'SL.UEM.TOTL.ZS',     # Unemployment (% labor force)
    'consumption_pct_gdp_TH':     'NE.CON.TOTL.ZS',     # Household consumption (% GDP)
    'exports_pct_gdp_TH':         'NE.EXP.GNFS.ZS',     # Exports of goods & services (% GDP)
    'imports_pct_gdp_TH':         'NE.IMP.GNFS.ZS',     # Imports of goods & services (% GDP)
    'gross_capital_formation_TH': 'NE.GDI.TOTL.ZS',     # Gross capital formation (% GDP)
    'govt_expenditure_pct_gdp_TH':'NE.CON.GOVT.ZS',     # Govt consumption expenditure (% GDP)
    'govt_debt_pct_gdp_TH':       'GC.DOD.TOTL.GD.ZS',  # Central govt debt (% GDP)
    'lending_rate_TH':            'FR.INR.LEND',         # Lending interest rate (%)
    'thailand_trade':             'TG.VAL.TOTL.GD.ZS',  # Trade (% GDP)
}

def fetch_worldbank_indicator(indicator: str, country: str = 'TH') -> pd.DataFrame:
    """Fetch a World Bank indicator via public API."""
    url = (f'https://api.worldbank.org/v2/country/{country}/indicator/{indicator}'
           f'?format=json&per_page=100&mrv=30')
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if len(data) < 2 or not data[1]:
        return pd.DataFrame()
    rows = []
    for item in data[1]:
        if item.get('value') is not None:
            rows.append({'year': int(item['date']), 'value': float(item['value'])})
    df = pd.DataFrame(rows).sort_values('year')
    df['country'] = country
    return df

def fetch_macro_data():
    print('\n── World Bank Macro Data ────────────────────────────────')
    for name, indicator in WORLDBANK_INDICATORS.items():
        out_path = RAW_DIR / f'{name}.csv'
        try:
            df = fetch_worldbank_indicator(indicator)
            if df.empty:
                print(f'  {name:40s} EMPTY')
                continue
            df.to_csv(out_path, index=False)
            print(f'  {name:40s} OK  {len(df):3d} years  '
                  f'{df["year"].min()}–{df["year"].max()}')
        except Exception as e:
            print(f'  {name:40s} ERROR: {e}')
        time.sleep(0.5)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch all raw data for Thailand alpha project')
    parser.add_argument('--market', action='store_true', help='Market data only')
    parser.add_argument('--fred',   action='store_true', help='FRED data only')
    parser.add_argument('--macro',  action='store_true', help='World Bank macro only')
    args = parser.parse_args()

    run_all = not (args.market or args.fred or args.macro)

    print('=' * 60)
    print('fetch_raw_data.py — Thailand Economic Analysis')
    print(f'Output: {RAW_DIR}')
    print(f'Period: {START} → {END}')
    print('=' * 60)

    if run_all or args.market:
        fetch_market_data()

    if run_all or args.fred:
        fetch_fred_data()

    if run_all or args.macro:
        fetch_macro_data()

    print('\n── Done ─────────────────────────────────────────────────')
    print('Next step: run NB02 (data_cleaning) to rebuild processed files')
    print('  → data/processed/unified_monthly.csv')
    print('  → data/processed/unified_weekly_clean.csv')
