"""
fetch_sector_nlp.py -- Fetch sector-specific Thai news and compute weekly sentiment.

Pipeline:
  1. Query Google News RSS with sector-specific keywords (7 sectors)
  2. VADER sentiment on each headline/description
  3. Aggregate to weekly (W-FRI) per sector: mean sentiment, news volume, positive ratio
  4. Output: data/processed/sector_sentiment_weekly.csv

Columns:
  date (index), BANK_sent, ENERGY_sent, ..., FOOD_sent   (mean VADER compound)
  BANK_vol,  ENERGY_vol,  ..., FOOD_vol                  (article count)
  BANK_pos,  ENERGY_pos,  ..., FOOD_pos                  (positive article ratio)

Coverage: Google News RSS returns up to 100 articles per query (mostly 2020-2026).
  Pre-coverage periods are NaN -- handled natively by XGBoost in NB14.
"""

import warnings
warnings.filterwarnings('ignore')

import time
import urllib.parse
import numpy as np
import pandas as pd
import feedparser
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'data' / 'processed' / 'sector_sentiment_weekly.csv'

# ── Sector-specific queries ────────────────────────────────────────────────────
# Professional approach: company names (most specific) + sector theme terms
SECTOR_QUERIES = {
    'BANK': [
        'KBANK OR BBL Thailand bank earnings',
        'SCB "Siam Commercial Bank" Thailand financial',
        '"Bank of Thailand" interest rate monetary policy',
        '"Kasikorn Bank" OR "Bangkok Bank" results',
        'KTB "Krungthai Bank" Thailand credit',
    ],
    'ENERGY': [
        'PTT Thailand energy oil results',
        'PTTEP "PTT Exploration" petroleum Thailand',
        '"Thai Oil" TOP refinery margins',
        'Thailand crude oil refinery energy sector',
        'IRPC OR RATCH Thailand power energy',
    ],
    'ICT': [
        'AIS "Advanced Info Service" Thailand telecom',
        '"True Corporation" OR TRUE Thailand 5G spectrum',
        'Thailand telecom revenue subscribers digital',
        'ADVANC INTUCH Thailand mobile data',
        'Thailand ICT technology internet infrastructure',
    ],
    'COMMERCE': [
        '"CP ALL" "7-Eleven" Thailand convenience retail',
        '"Central Retail" CRC Thailand shopping',
        'HomePro HMPRO Thailand home improvement',
        'Thailand retail sales consumer spending',
        '"Siam Makro" OR BJC Thailand wholesale',
    ],
    'HEALTH': [
        'BDMS "Bangkok Dusit Medical" hospital earnings',
        '"Bumrungrad Hospital" BH Thailand medical tourism',
        'Thailand private hospital healthcare revenue',
        'CHG RJH Thailand hospital group results',
        'Thailand medical tourism healthcare sector',
    ],
    'PROPERTY': [
        '"Land and Houses" LH Thailand real estate',
        'Supalai SPALI Thailand property condo',
        '"AP Thailand" OR Sansiri property developer',
        'Thailand real estate housing market',
        'Bangkok condo market property prices Thailand',
    ],
    'FOOD': [
        'CPF "Charoen Pokphand Foods" agribusiness results',
        '"Thai Union" TU seafood export earnings',
        'MINT "Minor International" restaurant hospitality',
        'Thailand food agriculture export prices',
        'OSP Osotspa Thailand consumer food beverage',
    ],
}

SLEEP_SEC  = 1.5   # polite delay between requests
MAX_WEEKS  = 260   # look back ~5 years max


def fetch_rss(query: str) -> list[dict]:
    """Fetch Google News RSS for one query, return list of {published, compound}."""
    url = (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode({
            "q":    query,
            "hl":   "en-US",
            "gl":   "US",
            "ceid": "US:en",
        })
    )
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print(f"    RSS error: {e}")
        return []

    sia = SentimentIntensityAnalyzer()
    articles = []
    for entry in feed.entries:
        # Parse published date
        pub_str = entry.get('published', '')
        try:
            pub = parsedate_to_datetime(pub_str)
            pub = pub.replace(tzinfo=None)  # strip tz for pandas
        except Exception:
            continue

        # Sentiment on title + snippet
        text = entry.get('title', '') + '. ' + entry.get('summary', '')
        score = sia.polarity_scores(text)['compound']
        articles.append({'published': pub, 'compound': score})

    return articles


def fetch_all_sectors(queries: dict[str, list[str]]) -> pd.DataFrame:
    """Fetch news for all sectors, return long-format DataFrame."""
    rows = []
    for sector, sector_queries in queries.items():
        print(f"\n  {sector}: fetching {len(sector_queries)} queries...")
        for q in sector_queries:
            arts = fetch_rss(q)
            for a in arts:
                rows.append({'sector': sector, **a})
            print(f"    [{q[:50]}...] -> {len(arts)} articles")
            time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['published'] = pd.to_datetime(df['published'])
    df = df.drop_duplicates(subset=['sector', 'published'])
    df = df.sort_values(['sector', 'published'])
    print(f"\nTotal articles fetched: {len(df)}")
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-format articles to weekly sector sentiment features."""
    if df.empty:
        return pd.DataFrame()

    # Snap each article to W-FRI ending week
    df = df.copy()
    df['week'] = df['published'].dt.to_period('W-FRI').dt.end_time.dt.normalize()

    sectors = sorted(df['sector'].unique())
    all_weeks = pd.date_range(
        df['week'].min(), df['week'].max(), freq='W-FRI'
    )

    weekly_parts = []
    for sec in sectors:
        sub = df[df['sector'] == sec].copy()
        g = sub.groupby('week')['compound'].agg(
            sent='mean',
            vol='count',
        ).reindex(all_weeks)
        g['pos'] = (
            sub[sub['compound'] > 0.05]
            .groupby('week')['compound']
            .count()
            .reindex(all_weeks)
            .div(g['vol'].replace(0, np.nan))
        )
        g.columns = [f'{sec}_{c}' for c in g.columns]
        weekly_parts.append(g)

    result = pd.concat(weekly_parts, axis=1)
    result.index.name = 'date'
    result.index = pd.to_datetime(result.index)
    return result


def main():
    print("Fetching sector-specific news from Google News RSS...")
    articles = fetch_all_sectors(SECTOR_QUERIES)

    if articles.empty:
        print("No articles fetched. Exiting.")
        return

    print("\nAggregating to weekly sector sentiment...")
    weekly = aggregate_weekly(articles)

    print(f"\nSector sentiment dataset:")
    print(f"  Shape: {weekly.shape}")
    print(f"  Date range: {weekly.index[0].date()} -> {weekly.index[-1].date()}")

    print(f"\nCoverage (non-NaN weeks) per sector:")
    for sec in ['BANK', 'ENERGY', 'ICT', 'COMMERCE', 'HEALTH', 'PROPERTY', 'FOOD']:
        col = f'{sec}_sent'
        if col in weekly.columns:
            n_nonnull = weekly[col].notna().sum()
            pct = n_nonnull / len(weekly)
            print(f"  {sec:12s}: {n_nonnull} weeks ({pct:.0%})")

    print(f"\nSample (last 5 rows):")
    sent_cols = [c for c in weekly.columns if c.endswith('_sent')]
    print(weekly[sent_cols].tail().round(3))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUT)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
