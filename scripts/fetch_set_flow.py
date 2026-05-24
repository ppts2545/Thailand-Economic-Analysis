"""
fetch_set_flow.py — SET daily fund flow by investor type

Source: SET website investor-type trading statistics
URL   : https://www.set.or.th/api/set/market/stats-by-investortype/SET

Investor types:
  Foreign         (F) — non-Thai investors
  Institutional   (I) — Thai funds, insurance, mutual funds
  Proprietary     (P) — broker proprietary desks
  Retail          (R) — individual Thai investors

Output : data/raw/set_fund_flow.csv
Columns: date, foreign_net, institutional_net, proprietary_net, retail_net
         (all in THB millions, positive = net buy)

Usage:
  python3 scripts/fetch_set_flow.py              # full history from 2010
  python3 scripts/fetch_set_flow.py --days 90    # last 90 days
  python3 scripts/fetch_set_flow.py --test       # probe endpoint only
"""

import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, date

ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / 'set_fund_flow.csv'

# SET API for investor-type fund flow
SET_API_URL = 'https://www.set.or.th/api/set/market/stats-by-investortype/SET'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.set.or.th/en/market/market-statistics/trading-by-investor-type',
}

# Investor type keys in SET API response
INVESTOR_MAP = {
    'F': 'foreign_net',
    'I': 'institutional_net',
    'P': 'proprietary_net',
    'R': 'retail_net',
}


def fetch_one_date(dt: date) -> dict | None:
    """Fetch investor-type net flow for a single trading date."""
    date_str = dt.strftime('%Y-%m-%d')
    params = {'date': date_str, 'lang': 'en'}
    try:
        resp = requests.get(SET_API_URL, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            return None  # non-trading day
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    # Expected structure: list of {investorType, netBuy, buy, sell}
    # or nested under a key
    rows = data if isinstance(data, list) else data.get('data', data.get('result', []))
    if not rows:
        return None

    rec = {'date': pd.to_datetime(date_str)}
    for item in rows:
        itype = str(item.get('investorType', item.get('investor_type', ''))).upper()
        if itype in INVESTOR_MAP:
            net = item.get('netBuy', item.get('net_buy', item.get('net', None)))
            if net is not None:
                rec[INVESTOR_MAP[itype]] = float(str(net).replace(',', ''))

    # Only return if at least foreign flow was captured
    if 'foreign_net' not in rec:
        return None
    # Ensure all columns present
    for col in INVESTOR_MAP.values():
        rec.setdefault(col, np.nan)
    return rec


def fetch_range(start: str, end: str) -> pd.DataFrame:
    """Fetch fund flow for a date range (inclusive)."""
    start_dt = pd.to_datetime(start).date()
    end_dt   = pd.to_datetime(end).date()

    records = []
    current = start_dt
    total_days = (end_dt - start_dt).days + 1

    print(f'  Fetching {start} → {end} ({total_days} calendar days) …')
    n_ok = 0
    while current <= end_dt:
        if current.weekday() < 5:  # skip weekends
            rec = fetch_one_date(current)
            if rec:
                records.append(rec)
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f'    {n_ok} trading days fetched …', end='\r')
        current += timedelta(days=1)
        time.sleep(0.15)  # polite rate

    print(f'\n  Got {n_ok} trading days')
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


def fetch_set_flow(start: str = '2010-01-01', end: str = None) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print('\n── SET Fund Flow by Investor Type ───────────────────────')
    df = fetch_range(start, end)
    if df.empty:
        print('  WARNING: no data retrieved')
    else:
        print(f'  Date range: {df["date"].min().date()} → {df["date"].max().date()}')
        last = df.iloc[-1]
        print(f'  Latest ({last["date"].date()}): '
              f'Foreign={last["foreign_net"]/1e3:+.1f}Bbn  '
              f'Inst={last["institutional_net"]/1e3:+.1f}Bbn  '
              f'Retail={last["retail_net"]/1e3:+.1f}Bbn')
    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch SET fund flow by investor type')
    parser.add_argument('--days', type=int, help='Fetch last N calendar days only')
    parser.add_argument('--test', action='store_true', help='Probe endpoint only')
    args = parser.parse_args()

    print('=' * 60)
    print('fetch_set_flow.py — Thailand Economic Analysis')
    print(f'Output: {OUT_PATH}')
    print('=' * 60)

    if args.test:
        print('\nTesting SET API …')
        # Try last available trading day
        from datetime import date as date_cls
        today = date_cls.today()
        for offset in range(7):
            dt = today - timedelta(days=offset)
            if dt.weekday() < 5:
                rec = fetch_one_date(dt)
                if rec:
                    print(f'  SET API: OK  — {dt}')
                    for k, v in rec.items():
                        if k != 'date':
                            print(f'    {k}: {v:+.0f} THB M')
                    return
        print('  SET API: no data for recent dates — check endpoint or market hours')
        return

    if args.days:
        start = (datetime.today() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        end   = datetime.today().strftime('%Y-%m-%d')
    else:
        start = '2010-01-01'
        end   = datetime.today().strftime('%Y-%m-%d')

    # Incremental update: skip dates already in CSV
    if OUT_PATH.exists() and not args.days:
        existing = pd.read_csv(OUT_PATH, parse_dates=['date'])
        if not existing.empty:
            last_date = existing['date'].max()
            start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f'  Incremental update from {start}')

    df_new = fetch_set_flow(start=start, end=end)
    if df_new.empty:
        print('No new data.')
        return

    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, parse_dates=['date'])
        df = pd.concat([existing, df_new], ignore_index=True).drop_duplicates('date')
    else:
        df = df_new

    df = df.sort_values('date').reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f'\n✓ Saved {len(df)} rows → {OUT_PATH}')


if __name__ == '__main__':
    main()
