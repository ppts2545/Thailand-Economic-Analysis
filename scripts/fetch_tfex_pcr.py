"""
fetch_tfex_pcr.py — TFEX SET50 futures put/call ratio

Source: TFEX (Thailand Futures Exchange) daily statistics
URL   : https://www.tfex.co.th/api/en/market-statistics/derivative-daily-statistics

Computes:
  PCR_OI  = total put open interest / total call open interest
  PCR_VOL = total put volume / total call volume

Output : data/raw/tfex_pcr.csv
Columns: date, put_oi, call_oi, pcr_oi, put_vol, call_vol, pcr_vol,
         total_oi, total_vol

Usage:
  python3 scripts/fetch_tfex_pcr.py             # full history from 2010
  python3 scripts/fetch_tfex_pcr.py --days 90   # last 90 days
  python3 scripts/fetch_tfex_pcr.py --test       # probe endpoint only
"""

import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT    = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / 'tfex_pcr.csv'

# TFEX daily statistics API
TFEX_API_URL = 'https://www.tfex.co.th/api/en/market-statistics/derivative-daily-statistics'

# Alternative: TFEX monthly statistics summary
TFEX_MONTHLY_URL = 'https://www.tfex.co.th/api/en/market-statistics/open-interest-summary'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.tfex.co.th/en/market-statistics/products/daily-statistics.html',
}

# SET50 futures product codes on TFEX
# S50 = SET50 futures (main contract)
SET50_PRODUCTS = ['S50', 'SET50']


def _safe_float(val) -> float:
    if val is None:
        return np.nan
    try:
        return float(str(val).replace(',', '').replace('-', '0') or 0)
    except (ValueError, TypeError):
        return np.nan


def fetch_tfex_daily(date_str: str) -> dict | None:
    """
    Fetch TFEX daily stats for a single date.
    Returns dict with put/call OI and volume, or None if no data.
    """
    params = {'date': date_str, 'lang': 'en'}
    try:
        resp = requests.get(TFEX_API_URL, params=params, headers=HEADERS, timeout=20)
        if resp.status_code in (404, 204):
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    # Response: list of products with {productCode, putOI, callOI, putVolume, callVolume}
    # or similar structure
    rows = data if isinstance(data, list) else (
        data.get('data') or data.get('result') or []
    )
    if not rows:
        return None

    put_oi = call_oi = put_vol = call_vol = 0.0
    found = False

    for row in rows:
        code = str(row.get('productCode', row.get('symbol', row.get('product', '')))).upper()
        # Include all SET50 options & futures, or aggregate all derivatives for market PCR
        option_type = str(row.get('optionType', row.get('type', ''))).upper()

        if 'PUT' in option_type or option_type == 'P':
            put_oi  += _safe_float(row.get('openInterest', row.get('oi', 0)))
            put_vol += _safe_float(row.get('volume', row.get('vol', 0)))
            found = True
        elif 'CALL' in option_type or option_type == 'C':
            call_oi  += _safe_float(row.get('openInterest', row.get('oi', 0)))
            call_vol += _safe_float(row.get('volume', row.get('vol', 0)))
            found = True

    if not found or (put_oi == 0 and call_oi == 0):
        return None

    pcr_oi  = put_oi  / call_oi  if call_oi  > 0 else np.nan
    pcr_vol = put_vol / call_vol if call_vol > 0 else np.nan

    return {
        'date':      pd.to_datetime(date_str),
        'put_oi':    put_oi,
        'call_oi':   call_oi,
        'pcr_oi':    round(pcr_oi,  4) if not np.isnan(pcr_oi)  else np.nan,
        'put_vol':   put_vol,
        'call_vol':  call_vol,
        'pcr_vol':   round(pcr_vol, 4) if not np.isnan(pcr_vol) else np.nan,
        'total_oi':  put_oi + call_oi,
        'total_vol': put_vol + call_vol,
    }


def fetch_range(start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start).date()
    end_dt   = pd.to_datetime(end).date()

    records = []
    current = start_dt
    n_ok = 0
    print(f'  Fetching {start} → {end} …')

    while current <= end_dt:
        if current.weekday() < 5:
            rec = fetch_tfex_daily(current.strftime('%Y-%m-%d'))
            if rec:
                records.append(rec)
                n_ok += 1
                if n_ok % 50 == 0:
                    print(f'    {n_ok} days done …', end='\r')
        current += timedelta(days=1)
        time.sleep(0.2)

    print(f'\n  Got {n_ok} trading days with PCR data')
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values('date').reset_index(drop=True)


def fetch_tfex_pcr(start: str = '2010-01-01', end: str = None) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print('\n── TFEX PUT/CALL Ratio ──────────────────────────────────')
    df = fetch_range(start, end)
    if df.empty:
        print('  WARNING: no data retrieved')
    else:
        print(f'  Date range: {df["date"].min().date()} → {df["date"].max().date()}')
        last = df.iloc[-1]
        print(f'  Latest ({last["date"].date()}): '
              f'PCR_OI={last["pcr_oi"]:.3f}  PCR_VOL={last["pcr_vol"]:.3f}')
        print(f'  PCR_OI  — mean: {df["pcr_oi"].mean():.3f}  '
              f'std: {df["pcr_oi"].std():.3f}  '
              f'range: [{df["pcr_oi"].min():.3f}, {df["pcr_oi"].max():.3f}]')
    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch TFEX SET50 put/call ratio')
    parser.add_argument('--days', type=int, help='Fetch last N calendar days')
    parser.add_argument('--test', action='store_true', help='Probe endpoint only')
    args = parser.parse_args()

    print('=' * 60)
    print('fetch_tfex_pcr.py — Thailand Economic Analysis')
    print(f'Output: {OUT_PATH}')
    print('=' * 60)

    if args.test:
        print('\nTesting TFEX API …')
        from datetime import date as date_cls
        today = date_cls.today()
        for offset in range(7):
            dt = today - timedelta(days=offset)
            if dt.weekday() < 5:
                rec = fetch_tfex_daily(dt.strftime('%Y-%m-%d'))
                if rec:
                    print(f'  TFEX API: OK  — {dt}')
                    print(f'    PCR_OI={rec["pcr_oi"]:.3f}  '
                          f'PCR_VOL={rec["pcr_vol"]:.3f}  '
                          f'total_OI={rec["total_oi"]:,.0f}')
                    return
        print('  TFEX API: no data for recent dates')
        print('  Note: TFEX options data may require different endpoint.')
        print(f'  Try: {TFEX_API_URL}')
        return

    if args.days:
        start = (datetime.today() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        end   = datetime.today().strftime('%Y-%m-%d')
    else:
        start = '2010-01-01'
        end   = datetime.today().strftime('%Y-%m-%d')

    # Incremental update
    if OUT_PATH.exists() and not args.days:
        existing = pd.read_csv(OUT_PATH, parse_dates=['date'])
        if not existing.empty:
            last_date = existing['date'].max()
            start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f'  Incremental update from {start}')

    df_new = fetch_tfex_pcr(start=start, end=end)
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
