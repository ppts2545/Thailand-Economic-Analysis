"""
fetch_set_stocks.py -- download weekly price data for ~60 liquid SET stocks

Output: data/processed/set_stocks_weekly.csv
  - Rows: Friday weekly close dates
  - Columns: ticker symbols (e.g. PTT, KBANK, ...)
  - Values: week-over-week return (not price, to avoid stationarity issues)

Tickers selected: top SET stocks by market cap / liquidity across all sectors.
Uses Yahoo Finance suffix .BK (Bangkok Stock Exchange).
"""

import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / 'data' / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Top ~60 liquid SET stocks across sectors ──────────────────────────────────
# Format: short_name → YF_TICKER
STOCKS = {
    # BANK
    'KBANK':  'KBANK.BK',
    'SCB':    'SCB.BK',
    'BBL':    'BBL.BK',
    'KTB':    'KTB.BK',
    'BAY':    'BAY.BK',
    'TTB':    'TTB.BK',
    'TISCO':  'TISCO.BK',
    'KKP':    'KKP.BK',
    # ENERGY
    'PTT':    'PTT.BK',
    'PTTEP':  'PTTEP.BK',
    'GULF':   'GULF.BK',
    'GPSC':   'GPSC.BK',
    'EGCO':   'EGCO.BK',
    'RATCH':  'RATCH.BK',
    'BGRIM':  'BGRIM.BK',
    'TOP':    'TOP.BK',
    # ICT / TECH
    'ADVANC': 'ADVANC.BK',
    'INTUCH': 'INTUCH.BK',
    'TRUE':   'TRUE.BK',
    'DELTA':  'DELTA.BK',
    'HANA':   'HANA.BK',
    # COMMERCE / RETAIL
    'CPALL':  'CPALL.BK',
    'MAKRO':  'MAKRO.BK',
    'BJC':    'BJC.BK',
    'HMPRO':  'HMPRO.BK',
    'CRC':    'CRC.BK',
    'COM7':   'COM7.BK',
    'GLOBAL': 'GLOBAL.BK',
    # HEALTH
    'BDMS':   'BDMS.BK',
    'BH':     'BH.BK',
    'BCH':    'BCH.BK',
    'CHG':    'CHG.BK',
    'PR9':    'PR9.BK',
    # PROPERTY
    'CPN':    'CPN.BK',
    'LH':     'LH.BK',
    'AP':     'AP.BK',
    'SPALI':  'SPALI.BK',
    'SC':     'SC.BK',
    'ORI':    'ORI.BK',
    # FOOD / AGRI
    'CPF':    'CPF.BK',
    'TU':     'TU.BK',
    'OSP':    'OSP.BK',
    'MINT':   'MINT.BK',
    'CBG':    'CBG.BK',
    # INDUSTRIAL / MATERIALS
    'IVL':    'IVL.BK',
    'SCC':    'SCC.BK',
    'SCCC':   'SCCC.BK',
    'PTTGC':  'PTTGC.BK',
    # TRANSPORT / INFRA
    'AOT':    'AOT.BK',
    'BEM':    'BEM.BK',
    'BTS':    'BTS.BK',
    'AAV':    'AAV.BK',
    # FINANCE / CREDIT
    'MTC':    'MTC.BK',
    'SAWAD':  'SAWAD.BK',
    'TIDLOR': 'TIDLOR.BK',
    'AEONTS': 'AEONTS.BK',
    # MEDIA / TOURISM
    'MAJOR':  'MAJOR.BK',
    'ERW':    'ERW.BK',
    'CENTEL': 'CENTEL.BK',
}

START = '2000-01-01'
END   = '2026-01-01'

def fetch_stock(name, ticker, retries=2):
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, start=START, end=END,
                              interval='1wk', progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 52:
                return None
            # Use Close price
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw['Close'].iloc[:, 0]
            else:
                close = raw['Close']
            # Resample to weekly Friday close
            close = close.resample('W-FRI').last().dropna()
            ret = close.pct_change().dropna()
            ret.name = name
            return ret
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    return None

print(f'Fetching {len(STOCKS)} SET stocks...')
results = {}
failed  = []

for i, (name, ticker) in enumerate(STOCKS.items()):
    ret = fetch_stock(name, ticker)
    if ret is not None and len(ret) >= 52:
        results[name] = ret
        print(f'  [{i+1:02d}/{len(STOCKS)}] {name:8s} OK  ({len(ret)} weeks)')
    else:
        failed.append(name)
        print(f'  [{i+1:02d}/{len(STOCKS)}] {name:8s} FAILED')
    time.sleep(0.3)   # polite rate limiting

print(f'\nSuccessful: {len(results)} / {len(STOCKS)}')
if failed:
    print(f'Failed: {failed}')

# ── Combine into wide DataFrame ──
df = pd.DataFrame(results)
df.index = pd.to_datetime(df.index).normalize()
df = df.sort_index()

# Drop stocks with < 5 years of data
min_obs = 52 * 5
df = df.loc[:, df.notna().sum() >= min_obs]
print(f'\nAfter filtering (<5yr): {df.shape[1]} stocks, {df.shape[0]} weeks')
print(f'Date range: {df.index[0].date()} -> {df.index[-1].date()}')

# Save
out_path = OUT_DIR / 'set_stocks_weekly.csv'
df.to_csv(out_path)
print(f'\nSaved: {out_path}  ({df.shape})')

# Coverage stats
cov = df.notna().mean().sort_values(ascending=False)
print(f'\nCoverage (fraction of weeks with data):')
print(cov.describe().round(3))
print(f'\nStocks with >80% coverage: {(cov > 0.80).sum()}')
