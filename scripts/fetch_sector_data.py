"""
fetch_sector_data.py -- Download SET sector constituents from yfinance
and aggregate into weekly sector returns.

Output: data/processed/sector_weekly.csv
  columns: date (index), BANK, ENERGY, ICT, COMMERCE, HEALTH, PROPERTY, FOOD
  each column = equal-weight weekly return of top constituents in that sector
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'data' / 'processed' / 'sector_weekly.csv'

# ── Sector constituents (major SET-listed stocks, suffix .BK) ──────────────────
# Selected for: long history (>2005), high market cap, sector representation
SECTORS = {
    'BANK': [
        'BBL.BK',    # Bangkok Bank (oldest listed bank)
        'KBANK.BK',  # Kasikorn Bank
        'SCB.BK',    # SCB (Siam Commercial Bank)
        'KTB.BK',    # Krungthai Bank
        'BAY.BK',    # Bank of Ayudhya
        'TISCO.BK',  # TISCO Bank
        'TMB.BK',    # TMBThanachart Bank
    ],
    'ENERGY': [
        'PTT.BK',    # PTT (largest Thai energy co, IPO 2001)
        'PTTEP.BK',  # PTT Exploration & Production
        'TOP.BK',    # Thai Oil
        'IRPC.BK',   # IRPC
        'RATCH.BK',  # Ratchaburi Electric
    ],
    'ICT': [
        'ADVANC.BK', # AIS / Advanced Info Service
        'INTUCH.BK', # Intouch Holdings (ADVANC parent)
        'TRUE.BK',   # True Corporation
        'DTAC.BK',   # DTAC (delisted ~2023, will have partial history)
        'GULF.BK',   # Gulf Energy (ICT infra)
    ],
    'COMMERCE': [
        'CPALL.BK',  # CP ALL (7-Eleven Thailand)
        'HMPRO.BK',  # HomePro
        'MAKRO.BK',  # Siam Makro
        'BJC.BK',    # Berli Jucker
        'CRC.BK',    # Central Retail Corp
    ],
    'HEALTH': [
        'BDMS.BK',   # Bangkok Dusit Medical Services
        'BGH.BK',    # Bangkok General Hospital
        'BH.BK',     # Bumrungrad Hospital
        'CHG.BK',    # Chularat Hospital Group
        'RJH.BK',    # Rajavithi Hospital
    ],
    'PROPERTY': [
        'LH.BK',     # Land and Houses
        'SPALI.BK',  # Supalai
        'AP.BK',     # AP Thailand
        'QH.BK',     # Quality Houses
        'SIRI.BK',   # Sansiri
    ],
    'FOOD': [
        'CPF.BK',    # Charoen Pokphand Foods
        'TU.BK',     # Thai Union Group
        'MINT.BK',   # Minor International
        'SHR.BK',    # Shang Xia Resort (minor)
        'OSP.BK',    # Osotspa
    ],
}

START_DATE = '2000-01-01'
END_DATE   = '2025-12-31'

def fetch_sector_returns(sectors, start, end):
    """Download adjusted close prices for all stocks and compute weekly sector returns."""
    all_tickers = [t for tickers in sectors.values() for t in tickers]

    print(f'Downloading {len(all_tickers)} stocks from {start} to {end}...')
    raw = yf.download(
        all_tickers,
        start=start, end=end,
        interval='1wk',
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']]

    print(f'\nDownloaded: {prices.shape[0]} weeks x {prices.shape[1]} stocks')

    # Weekly returns
    rets = prices.pct_change().iloc[1:]  # drop first NaN row

    # Aggregate to sector level (equal-weight, require ≥2 valid stocks)
    sector_rets = {}
    for sector, tickers in sectors.items():
        valid = [t for t in tickers if t in rets.columns]
        if not valid:
            print(f'  {sector}: no data available')
            continue
        sector_r = rets[valid].dropna(how='all')  # keep rows with any valid stock
        # Equal-weight: mean across available stocks each week
        sector_rets[sector] = sector_r.mean(axis=1)
        n_valid = sector_r.notna().sum(axis=1)
        coverage = (n_valid >= 2).mean()
        print(f'  {sector:12s}: {len(valid)} tickers, {len(sector_r)} weeks, '
              f'{coverage:.0%} weeks with ≥2 stocks')

    df = pd.DataFrame(sector_rets)
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)

    # Align to consistent weekly frequency (Friday close)
    df = df.resample('W-FRI').last()

    return df


def main():
    sector_df = fetch_sector_returns(SECTORS, START_DATE, END_DATE)

    print(f'\nFinal sector dataset:')
    print(f'  Shape: {sector_df.shape}')
    print(f'  Date range: {sector_df.index[0].date()} -> {sector_df.index[-1].date()}')
    print(f'\nCoverage (non-NaN) per sector:')
    for col in sector_df.columns:
        print(f'  {col:12s}: {sector_df[col].notna().mean():.1%}')

    print(f'\nSample (last 5 rows):')
    print(sector_df.tail().round(4))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sector_df.to_csv(OUT)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
