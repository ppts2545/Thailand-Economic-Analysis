"""
fetch_bot_bonds.py — Thai government bond yields from Bank of Thailand

Primary source : BOT Open API  (apiportal.bot.or.th)  — free, no key needed
Fallback source: BOT Excel download (bot.or.th direct file)

Output : data/raw/bot_bond_yields.csv
Columns: date, yield_1y, yield_2y, yield_3y, yield_5y, yield_7y, yield_10y

Usage:
  python3 scripts/fetch_bot_bonds.py            # full history
  python3 scripts/fetch_bot_bonds.py --year 2024  # single year
  python3 scripts/fetch_bot_bonds.py --test       # probe endpoints only
"""

import argparse
import io
import time
import warnings
warnings.filterwarnings('ignore')

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date

ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / 'bot_bond_yields.csv'

# ── BOT Open API ──────────────────────────────────────────────────────────────
# Endpoint: https://apiportal.bot.or.th/bot/public/Stat-YieldCurve/v2/YIELD_CURVE
# Params  : start_period=YYYY-MM-DD, end_period=YYYY-MM-DD
# No API key required for read-only public stats endpoints
BOT_API_URL = 'https://apiportal.bot.or.th/bot/public/Stat-YieldCurve/v2/YIELD_CURVE'

# ── BOT Excel fallback ────────────────────────────────────────────────────────
# Historical bond yield Excel published by BOT Financial Statistics division
BOT_EXCEL_URL = (
    'https://www.bot.or.th/content/dam/bot/financial-statistics/'
    'financial-markets/data/FM_BONDYIELD_EN.xlsx'
)

TENORS = [1, 2, 3, 5, 7, 10]  # years


def fetch_via_bot_api(start: str, end: str) -> pd.DataFrame:
    """
    Fetch yield curve from BOT Open API.

    Returns DataFrame with columns: date, yield_1y ... yield_10y
    Raises on failure.
    """
    params = {'start_period': start, 'end_period': end}
    headers = {'accept': 'application/json'}
    resp = requests.get(BOT_API_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    # Response structure: {"result": {"data": [{"period": "YYYY-MM-DD", "1": val, ...}]}}
    if 'result' not in data or 'data' not in data['result']:
        raise ValueError(f'Unexpected BOT API response structure: {list(data.keys())}')

    records = []
    for row in data['result']['data']:
        rec = {'date': pd.to_datetime(row.get('period') or row.get('date'))}
        for t in TENORS:
            val = row.get(str(t)) or row.get(f'{t}Y') or row.get(f'yield_{t}y')
            rec[f'yield_{t}y'] = float(val) if val is not None else np.nan
        records.append(rec)

    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


def fetch_via_excel() -> pd.DataFrame:
    """
    Fallback: download BOT yield Excel, parse all tenor columns.
    The Excel has columns like: Date, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y
    """
    resp = requests.get(BOT_EXCEL_URL, timeout=60)
    resp.raise_for_status()

    xl = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=0)
    xl.columns = [str(c).strip() for c in xl.columns]

    # Find date column (first column)
    date_col = xl.columns[0]
    xl = xl.rename(columns={date_col: 'date'})
    xl['date'] = pd.to_datetime(xl['date'], errors='coerce')
    xl = xl.dropna(subset=['date'])

    # Map tenor columns
    rename = {}
    for t in TENORS:
        for candidate in [str(t), f'{t}Y', f'{t}y', f'{t} Y', f'{t} yr']:
            if candidate in xl.columns:
                rename[candidate] = f'yield_{t}y'
                break

    xl = xl.rename(columns=rename)
    keep = ['date'] + [f'yield_{t}y' for t in TENORS if f'yield_{t}y' in xl.columns]
    return xl[keep].sort_values('date').reset_index(drop=True)


def fetch_bot_bonds(start_year: int = 2000, end_year: int = None,
                    test_only: bool = False) -> pd.DataFrame:
    """Main fetch function: tries API first, falls back to Excel."""
    if end_year is None:
        end_year = datetime.today().year

    print('\n── Thai Bond Yields (Bank of Thailand) ─────────────────')

    # ── Try BOT API year-by-year (API limits date range) ─────────────────────
    all_frames = []
    api_ok = False
    if not test_only:
        print('  Trying BOT Open API …')
        try:
            for yr in range(start_year, end_year + 1):
                start = f'{yr}-01-01'
                end   = f'{yr}-12-31'
                df_yr = fetch_via_bot_api(start, end)
                if not df_yr.empty:
                    all_frames.append(df_yr)
                    print(f'    {yr}: {len(df_yr):4d} rows', end='\r')
                time.sleep(0.3)
            api_ok = True
            print(f'\n  BOT API OK — {sum(len(f) for f in all_frames)} rows total')
        except Exception as e:
            print(f'\n  BOT API failed: {e}')

    if not api_ok and not test_only:
        print('  Trying BOT Excel download …')
        try:
            df = fetch_via_excel()
            print(f'  BOT Excel OK — {len(df)} rows  '
                  f'{df["date"].min().date()} → {df["date"].max().date()}')
            return df
        except Exception as e:
            print(f'  BOT Excel failed: {e}')
            raise RuntimeError('Both BOT API and Excel fallback failed. '
                               'Check internet connection and source URLs.') from e

    if api_ok:
        df = pd.concat(all_frames, ignore_index=True).drop_duplicates('date')
        return df.sort_values('date').reset_index(drop=True)

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Fetch Thai government bond yields')
    parser.add_argument('--year',  type=int, help='Fetch single year only')
    parser.add_argument('--test',  action='store_true', help='Probe endpoints only')
    args = parser.parse_args()

    print('=' * 60)
    print('fetch_bot_bonds.py — Thailand Economic Analysis')
    print(f'Output: {OUT_PATH}')
    print('=' * 60)

    if args.test:
        print('\nTesting BOT API …')
        try:
            sample = fetch_via_bot_api('2024-01-01', '2024-01-31')
            print(f'  BOT API: OK  ({len(sample)} rows)')
            print(sample.head(3).to_string(index=False))
        except Exception as e:
            print(f'  BOT API: FAIL — {e}')
        print('\nTesting BOT Excel …')
        try:
            sample = fetch_via_excel()
            print(f'  BOT Excel: OK  ({len(sample)} rows)')
            print(sample.tail(3).to_string(index=False))
        except Exception as e:
            print(f'  BOT Excel: FAIL — {e}')
        return

    start_year = args.year if args.year else 2000
    end_year   = args.year if args.year else datetime.today().year

    df = fetch_bot_bonds(start_year=start_year, end_year=end_year)
    if df.empty:
        print('No data retrieved.')
        return

    # Append to existing if partial update
    if OUT_PATH.exists() and args.year:
        existing = pd.read_csv(OUT_PATH, parse_dates=['date'])
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates('date')
        df = df.sort_values('date').reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f'\n✓ Saved {len(df)} rows → {OUT_PATH}')
    print(f'  Date range: {df["date"].min().date()} → {df["date"].max().date()}')
    for t in TENORS:
        col = f'yield_{t}y'
        if col in df.columns:
            last = df[col].dropna().iloc[-1] if not df[col].dropna().empty else np.nan
            print(f'  {t:2d}Y yield (latest): {last:.3f}%')


if __name__ == '__main__':
    main()
