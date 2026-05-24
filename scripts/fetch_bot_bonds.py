"""
fetch_bot_bonds.py — Thai government bond yields from Bank of Thailand

Source  : BOT Open Data API  (gateway.api.bot.or.th)
Register: https://portal.api.bot.or.th  (free account required)
Auth    : Authorization: Bearer <token>  stored in .env as BOT_API_KEY

Series (monthly, all T-Bill & Government Bond Yield):
  FMRTINTM00030 = 1Y   FMRTINTM00031 = 2Y   FMRTINTM00032 = 3Y
  FMRTINTM00034 = 5Y   FMRTINTM00036 = 7Y   FMRTINTM00039 = 10Y

Output : data/raw/bot_bond_yields.csv
Columns: date (YYYY-MM-01), yield_1y, yield_2y, yield_3y, yield_5y, yield_7y, yield_10y

Usage:
  python3 scripts/fetch_bot_bonds.py           # full history (uses BOT_API_KEY from .env)
  python3 scripts/fetch_bot_bonds.py --key <k> # explicit key
  python3 scripts/fetch_bot_bonds.py --test    # quick test (3 months)
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RAW_DIR / 'bot_bond_yields.csv'

GW_BASE = 'https://gateway.api.bot.or.th'

# T-Bill & Government Bond Yield series codes
YIELD_SERIES = {
    'yield_1y':  'FMRTINTM00030',
    'yield_2y':  'FMRTINTM00031',
    'yield_3y':  'FMRTINTM00032',
    'yield_5y':  'FMRTINTM00034',
    'yield_7y':  'FMRTINTM00036',
    'yield_10y': 'FMRTINTM00039',
}


def _headers(api_key: str) -> dict:
    return {'accept': 'application/json', 'Authorization': f'Bearer {api_key}'}


def fetch_series(series_code: str, start: str, end: str, api_key: str) -> list[dict]:
    """Fetch one series from /observations/. Returns list of {period_start, value}."""
    r = requests.get(
        f'{GW_BASE}/observations/',
        headers=_headers(api_key),
        params={'series_code': series_code, 'start_period': start, 'end_period': end},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    series_list = data.get('result', {}).get('series', [])
    if not series_list:
        return []
    return series_list[0].get('observations', [])


def fetch_series_chunked(series_code: str, start: str, end: str,
                          api_key: str, chunk_years: int = 5) -> list[dict]:
    """Fetch a series in year-by-year chunks to bypass the 120-record API limit."""
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    all_obs  = []
    current  = start_dt

    while current <= end_dt:
        chunk_end = min(current + pd.DateOffset(years=chunk_years) - pd.DateOffset(days=1), end_dt)
        obs = fetch_series(series_code,
                           current.strftime('%Y-%m-%d'),
                           chunk_end.strftime('%Y-%m-%d'),
                           api_key)
        all_obs.extend(obs)
        current = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.2)

    # Deduplicate by period_start
    seen = set()
    unique = []
    for o in all_obs:
        if o['period_start'] not in seen:
            seen.add(o['period_start'])
            unique.append(o)
    return unique


def fetch_bot_bonds(start: str = '2000-01-01', end: str = None,
                    api_key: str = '') -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    if not api_key:
        raise ValueError(
            'BOT_API_KEY not set.\n'
            '  Register at: https://portal.api.bot.or.th\n'
            '  Then add BOT_API_KEY=<token> to your .env file'
        )

    print('\n── Thai Bond Yields (Bank of Thailand API) ──────────────')
    print(f'  Period: {start} → {end}')

    all_dfs = []
    for col, code in YIELD_SERIES.items():
        try:
            obs = fetch_series_chunked(code, start, end, api_key)
            if not obs:
                print(f'  {col}: no data')
                continue
            df_s = pd.DataFrame(obs)
            df_s['date'] = pd.to_datetime(df_s['period_start'] + '-01')
            df_s[col]    = pd.to_numeric(df_s['value'], errors='coerce')
            all_dfs.append(df_s[['date', col]])
            last_val = df_s[col].dropna().iloc[-1]
            print(f'  {col}: {len(df_s):4d} months  '
                  f'{df_s["date"].min().date()} → {df_s["date"].max().date()}  '
                  f'latest={last_val:.3f}%')
        except Exception as e:
            print(f'  {col}: ERROR — {e}')

    if not all_dfs:
        return pd.DataFrame()

    # Merge all tenors on date
    df = all_dfs[0]
    for df_s in all_dfs[1:]:
        df = df.merge(df_s, on='date', how='outer')

    # Derived signal: yield slope (10Y - 1Y)
    if 'yield_10y' in df.columns and 'yield_1y' in df.columns:
        df['th_slope'] = df['yield_10y'] - df['yield_1y']

    return df.sort_values('date').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Fetch Thai government bond yields')
    parser.add_argument('--key',  type=str, default='', help='BOT API key (or set BOT_API_KEY in .env)')
    parser.add_argument('--test', action='store_true',  help='Quick test: last 3 months only')
    args = parser.parse_args()

    api_key = args.key or os.environ.get('BOT_API_KEY', '')

    print('=' * 60)
    print('fetch_bot_bonds.py — Thailand Economic Analysis')
    print(f'Output: {OUT_PATH}')
    if api_key:
        print(f'API key: {api_key[:12]}… (set)')
    else:
        print('ERROR: BOT_API_KEY not set')
        print('  → Add BOT_API_KEY=<token> to .env')
        return
    print('=' * 60)

    if args.test:
        start = '2024-01-01'
        end   = '2024-03-31'
    else:
        start = '2000-01-01'
        end   = datetime.today().strftime('%Y-%m-%d')

        # Incremental: only fetch new months
        if OUT_PATH.exists():
            existing = pd.read_csv(OUT_PATH, parse_dates=['date'])
            if not existing.empty:
                last = existing['date'].max()
                start = (last + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
                print(f'  Incremental update from {start}')

    df_new = fetch_bot_bonds(start=start, end=end, api_key=api_key)
    if df_new.empty:
        print('No data retrieved.')
        return

    if args.test:
        print('\nSample data:')
        print(df_new.to_string(index=False))
        return

    # Merge with existing
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, parse_dates=['date'])
        df = pd.concat([existing, df_new], ignore_index=True).drop_duplicates('date')
    else:
        df = df_new

    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f'\n✓ Saved {len(df)} rows → {OUT_PATH}')
    print(f'  Date range: {df["date"].min().date()} → {df["date"].max().date()}')
    print(f'  Yield slope (latest): {df["th_slope"].dropna().iloc[-1]:+.2f}pp')


if __name__ == '__main__':
    main()
