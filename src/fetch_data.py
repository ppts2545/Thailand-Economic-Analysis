import os
import time
import random
import feedparser
import requests as req_lib
import polars as pl
import yfinance as yf
from fredapi import Fred
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

# Extended start: Gold/Oil available from Aug 2000 — use 2000-01-01 so weekly
# resampling lands cleanly; USD/THB gap (2000-2003) filled via FRED EXTHUS monthly.
DATA_START = '2000-01-01'

# Countries to compare: Thailand vs major economies
COUNTRIES = {
    'TH': 'Thailand',
    'US': 'United States',
    'CN': 'China',
    'JP': 'Japan',
    'DE': 'Germany',
    'SG': 'Singapore',
}

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


# --- SECTION 1: LABOR MARKET (World Bank API) ---
# Annual data: unemployment, youth unemployment, labor force participation

LABOR_INDICATORS = {
    'unemployment_rate':        'SL.UEM.TOTL.ZS',   # % of total labor force
    'youth_unemployment_rate':  'SL.UEM.1524.ZS',   # % of youth labor force (15-24)
    'labor_force_participation': 'SL.TLF.CACT.ZS',  # % of working-age population
    'employment_ratio':         'SL.EMP.TOTL.SP.ZS', # Employment-to-population ratio
    # Sector employment — tracks shift from agriculture → industry → services
    'employment_agriculture':   'SL.AGR.EMPL.ZS',   # % employed in agriculture
    'employment_industry':      'SL.IND.EMPL.ZS',   # % employed in industry
    'employment_services':      'SL.SRV.EMPL.ZS',   # % employed in services
}

def fetch_worldbank(indicator_code, country='TH'):
    """
    Fetch a World Bank indicator for one country.
    Returns a polars DataFrame: year, value, country.
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country}"
        f"/indicator/{indicator_code}?format=json&per_page=50&mrv=22"
    )
    resp = req_lib.get(url, headers=BROWSER_HEADERS, timeout=10)
    resp.raise_for_status()
    records = resp.json()[1]
    return pl.DataFrame([
        {'year': int(r['date']), 'value': r['value'], 'country': country}
        for r in records if r['value'] is not None
    ]).sort('year')

def fetch_labor_market(countries: dict = COUNTRIES):
    """
    Fetch all labor market indicators for all countries.
    Returns a dict of {indicator_name: polars DataFrame (all countries stacked)}.
    """
    result = {}
    for name, code in LABOR_INDICATORS.items():
        frames = []
        for iso in countries:
            try:
                frames.append(fetch_worldbank(code, country=iso))
            except Exception as e:
                print(f"[Skip] {iso} / {name}: {e}")
        if frames:
            result[name] = pl.concat(frames)
    return result


# --- SECTION 2: ECONOMIC CONTEXT (World Bank API) ---
# GDP growth, inflation, trade openness — for understanding why unemployment moves

ECONOMIC_INDICATORS = {
    # --- 1. Consumption ---
    'consumption_pct_gdp':    'NE.CON.PRVT.ZS',    # Household final consumption (% of GDP)

    # --- 2. Interest Rate / Monetary Policy ---
    'lending_rate':           'FR.INR.LEND',        # Commercial lending rate (%)
    'broad_money_gdp':        'FM.LBL.BMNY.GD.ZS',  # Broad money M2 (% of GDP)

    # --- 3. Inflation ---
    'inflation':              'FP.CPI.TOTL.ZG',     # CPI inflation (%)

    # --- 4. Unemployment → in LABOR_INDICATORS already ---

    # --- 5. Business Investment ---
    'gross_capital_formation': 'NE.GDI.TOTL.ZS',   # Gross capital formation (% of GDP)
    'fdi_inflows':             'BX.KLT.DINV.WD.GD.ZS', # FDI net inflows (% of GDP)

    # --- 6. Exports / Trade ---
    'exports_pct_gdp':        'NE.EXP.GNFS.ZS',    # Exports of goods & services (% of GDP)
    'imports_pct_gdp':        'NE.IMP.GNFS.ZS',    # Imports of goods & services (% of GDP)
    'trade_pct_gdp':          'NE.TRD.GNFS.ZS',    # Total trade openness (% of GDP)

    # --- 7. Geopolitical / Confidence → VIX + gold in MARKET_TICKERS ---

    # --- 8. Technology / Innovation ---
    'gdp_growth':             'NY.GDP.MKTP.KD.ZG',  # GDP growth (captures tech cycles)

    # --- 9. Government Policy ---
    'govt_expenditure_pct_gdp': 'GC.XPN.TOTL.GD.ZS', # Govt expenditure (% of GDP)
    'govt_debt_pct_gdp':        'GC.DOD.TOTL.GD.ZS',  # Central govt debt (% of GDP)
}

def fetch_economic_context(countries: dict = COUNTRIES):
    """
    Fetch economic context indicators for all countries.
    Same structure as fetch_labor_market().
    """
    result = {}
    for name, code in ECONOMIC_INDICATORS.items():
        frames = []
        for iso in countries:
            try:
                frames.append(fetch_worldbank(code, country=iso))
            except Exception as e:
                print(f"[Skip] {iso} / {name}: {e}")
        if frames:
            result[name] = pl.concat(frames)
    return result


# --- SECTION 3: JOB SEARCH TRENDS (Google Trends) ---
# Search interest as a leading/coincident indicator for labor market sentiment.
# High "job search" queries often correlate with rising unemployment or career pivots.

JOB_KEYWORDS = {
    'TH': ['หางาน', 'สมัครงาน', 'ว่างงาน'],
    'global': ['unemployment', 'jobs', 'layoffs'],
}

# Google Trends keyword groups mapped to the 9 economic factors
TREND_KEYWORD_GROUPS = {
    'consumption_TH':   ['ซื้อบ้าน', 'ซื้อรถ', 'ผ่อนบ้าน'],          # factor 1
    'confidence_TH':    ['ลงทุน', 'หุ้น', 'เศรษฐกิจไทย'],             # factor 5,9
    'technology_global':['artificial intelligence', 'automation', 'AI jobs'],  # factor 8
    'geopolitical':     ['war', 'sanctions', 'trade war'],              # factor 7
}

def fetch_job_search_trends(keywords: list, geo='TH', timeframe='today 12-m', max_retries=5):
    """
    Fetch Google Trends interest for job-related keywords.
    Returns a pandas DataFrame (pytrends output), or None on failure.
    """
    pytrends = TrendReq(hl='en-US', tz=360, requests_args={'headers': BROWSER_HEADERS})

    for attempt in range(max_retries):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()
            if not data.empty:
                if 'isPartial' in data.columns:
                    data = data.drop(columns=['isPartial'])
                return data
        except Exception as e:
            if "429" in str(e):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Rate limited] Retry {attempt + 1}/{max_retries} in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[Error] {e}")
                break

    return None


# --- SECTION 4: MARKET SIGNALS (yfinance) ---
# Financial markets often price in economic shifts before official data is published.

MARKET_TICKERS = {
    # Thailand
    'SET_index':        '^SET.BK',  # Thailand stock market (factor 1,5,9)
    'USD_THB':          'THB=X',    # Baht strength (factor 6)
    # Global risk / geopolitical (factor 7)
    'vix':              '^VIX',     # Fear gauge — spikes during war/crisis
    'gold':             'GC=F',     # Safe haven — rises when confidence falls
    # Global growth (factor 6)
    'sp500':            '^GSPC',    # US market (global demand proxy)
    'oil':              'CL=F',     # Crude oil (global growth / supply chain)
    # Interest rate (factor 2)
    'us_10yr_treasury': '^TNX',     # US 10-yr yield — global rate benchmark
    'us_2yr_treasury':  '^IRX',     # US 13-week T-Bill (short end) — yield curve
    # Technology / innovation (factor 8)
    'nasdaq':           '^IXIC',    # Nasdaq — tech sector performance
    # USD strength — driver of Gold (inverse) and USD/THB (direct)
    'dxy':              'DX-Y.NYB', # US Dollar Index (basket of 6 currencies)
}

def fetch_market_signals(tickers: dict = MARKET_TICKERS, start: str = DATA_START):
    """
    Fetch OHLCV history for each market ticker from `start` date to today.
    Returns a dict of {name: polars DataFrame with date, open, high, low, close, volume}.
    """
    result = {}
    for name, symbol in tickers.items():
        try:
            df_pd = yf.download(symbol, start=start, progress=False, auto_adjust=True)
            if not df_pd.empty:
                # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
                if df_pd.columns.nlevels > 1:
                    df_pd.columns = [c[0].lower() for c in df_pd.columns]
                else:
                    df_pd.columns = [c.lower() for c in df_pd.columns]
                df_pd = df_pd.reset_index()
                df_pd.rename(columns={'date': 'date', 'index': 'date'}, inplace=True)
                df_pd['date'] = df_pd['date'].dt.strftime('%Y-%m-%d')

                cols = {'date': df_pd['date'].tolist(),
                        'close': df_pd['close'].astype(float).tolist()}
                for col in ['open', 'high', 'low', 'volume']:
                    if col in df_pd.columns:
                        cols[col] = df_pd[col].astype(float).tolist()
                result[name] = pl.DataFrame(cols)
        except Exception as e:
            print(f"[Skip] {name}: {e}")
    return result


# --- SECTION 5: NEWS SENTIMENT (Bangkok Post RSS) ---
# Quant-grade weighted sentiment model:
#   score = VADER_compound × importance × (1 + power_bonus)
# where:
#   importance = max keyword weight found in headline (default 1.0)
#   power_bonus = count of high-impact keywords / 3 (capped at 1.0)
# Credibility is uniform (Bangkok Post RSS = authoritative source = 1.0).
# Recency decay is applied at aggregation time (in the notebook), not here.

BANGKOK_POST_RSS = {
    'business': 'https://www.bangkokpost.com/rss/data/business.xml',
    'economy':  'https://www.bangkokpost.com/rss/data/economy.xml',
}

# Keyword importance map:  word/phrase → importance multiplier
# Sources: Tetlock (2007), Loughran & McDonald (2011), empirical hedge fund practice
IMPORTANCE_MAP = {
    # Central bank & rates — highest market impact
    'federal reserve': 2.0, 'fed ': 2.0, 'rate hike': 2.0, 'rate cut': 2.0,
    'interest rate': 1.8, 'monetary policy': 1.8, 'bank of thailand': 1.8,
    'quantitative easing': 1.8, 'taper': 1.7, 'pivot': 1.7,
    # Macro shock events
    'recession': 2.0, 'default': 2.0, 'crisis': 1.8, 'emergency': 1.8,
    'collapse': 2.0, 'crash': 2.0, 'panic': 1.7, 'systemic': 1.8,
    # Directional price words (Loughran & McDonald power words)
    'surge': 1.4, 'soar': 1.4, 'plunge': 1.5, 'tumble': 1.5,
    'selloff': 1.5, 'rally': 1.3, 'record high': 1.4, 'record low': 1.4,
    'all-time': 1.3,
    # Trade & geopolitics
    'trade war': 1.8, 'tariff': 1.5, 'sanction': 1.8, 'war': 1.7,
    'inflation': 1.5, 'deflation': 1.5, 'stagflation': 1.8,
    # Thailand-specific macro
    'baht': 1.4, 'set index': 1.3, 'gdp': 1.3, 'export': 1.2, 'current account': 1.3,
    # Diminished credibility words → reduce weight
    'rumor': 0.5, 'unconfirmed': 0.5, 'speculation': 0.6, 'reportedly': 0.7,
}


def _article_importance(text: str) -> tuple[float, int]:
    """Return (max_importance, power_word_count) for a piece of text."""
    t = text.lower()
    max_imp   = 1.0
    pw_count  = 0
    for word, imp in IMPORTANCE_MAP.items():
        if word in t:
            if imp > 1.0:
                pw_count += 1
            max_imp = max(max_imp, imp)
    return max_imp, pw_count


def fetch_news_sentiment(feeds: dict = BANGKOK_POST_RSS):
    """
    Fetch latest headlines from Bangkok Post RSS and score with weighted VADER sentiment.

    Columns returned:
      title, published, compound       — raw VADER score
      importance                        — keyword importance multiplier (1.0–2.0)
      power_word_count                  — number of high-impact keywords found
      weighted_compound                 — compound × importance × (1 + min(pw_count/3, 1))
      sentiment                         — positive / negative / neutral (from weighted score)
      feed                              — source feed name
    """
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for feed_name, url in feeds.items():
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            raw_compound = analyzer.polarity_scores(entry.title)['compound']
            importance, pw_count = _article_importance(entry.title)
            power_bonus  = min(pw_count / 3.0, 1.0)          # cap bonus at 1× extra
            weighted     = raw_compound * importance * (1.0 + power_bonus)
            weighted     = max(-1.0, min(1.0, weighted))       # clamp to [-1, 1]
            rows.append({
                'title':            entry.title,
                'published':        entry.get('published', ''),
                'compound':         round(raw_compound, 4),
                'importance':       round(importance, 2),
                'power_word_count': pw_count,
                'weighted_compound': round(weighted, 4),
                'sentiment': ('positive' if weighted > 0.05
                              else ('negative' if weighted < -0.05 else 'neutral')),
                'feed': feed_name,
            })
    return pl.DataFrame(rows)


# --- SECTION 5b: GDELT DOC API v2 — Historical Thailand Financial News ---
# Free, no key needed. Coverage: ~2015-present (earlier data sparse).
# DOC API returns article metadata: url, title, seendate, domain, language, tone.
# We apply the same weighted VADER pipeline as Bangkok Post.

GDELT_DOC_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'

GDELT_QUERIES = [
    'Thailand economy GDP growth',
    'Thailand SET index stock market',
    'Thailand baht currency exchange rate',
    'Bank of Thailand interest rate monetary policy',
    'Thailand inflation exports trade',
    'Thailand financial crisis recession',
    'Federal Reserve rate hike cut Thailand',
    'gold price oil price Thailand',
]


def fetch_gdelt_news(
    start_year: int = 2015,
    end_year:   int = 2025,
    max_per_chunk: int = 250,
    delay: float = 1.5,
) -> pl.DataFrame:
    """
    Fetch Thailand financial news from GDELT DOC API v2.

    Iterates quarterly date chunks × GDELT_QUERIES, applies weighted VADER scoring,
    deduplicates by URL, and returns a DataFrame with the same schema as
    fetch_news_sentiment() plus a 'source' column = 'gdelt'.

    Columns: title, published, compound, importance, power_word_count,
             weighted_compound, sentiment, feed, source
    """
    analyzer = SentimentIntensityAnalyzer()
    seen_urls: set = set()
    rows: list[dict] = []

    # Build quarterly date windows
    quarters = []
    for year in range(start_year, end_year + 1):
        for q_start_month, q_end_month in [(1,3), (4,6), (7,9), (10,12)]:
            import datetime
            qs = datetime.date(year, q_start_month, 1)
            # Last day of end month
            if q_end_month == 12:
                qe = datetime.date(year, 12, 31)
            else:
                qe = datetime.date(year, q_end_month + 1, 1) - datetime.timedelta(days=1)
            if qe > datetime.date.today():
                qe = datetime.date.today()
            if qs >= qe:
                break
            quarters.append((qs, qe))

    total = len(quarters) * len(GDELT_QUERIES)
    done  = 0
    for qs, qe in quarters:
        start_dt = qs.strftime('%Y%m%d') + '000000'
        end_dt   = qe.strftime('%Y%m%d') + '235959'

        for query in GDELT_QUERIES:
            done += 1
            params = {
                'query':         query,
                'mode':          'artlist',
                'format':        'json',
                'startdatetime': start_dt,
                'enddatetime':   end_dt,
                'maxrecords':    max_per_chunk,
                'sourcelang':    'english',
            }
            try:
                resp = req_lib.get(GDELT_DOC_URL, params=params,
                                   headers=BROWSER_HEADERS, timeout=20)
                if resp.status_code != 200:
                    time.sleep(delay)
                    continue
                data = resp.json()
                articles = data.get('articles', []) or []
                for art in articles:
                    url   = art.get('url', '')
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    title = art.get('title', '') or ''
                    # Skip non-English titles (< 60% printable ASCII)
                    if title and sum(ord(c) < 128 for c in title) / len(title) < 0.60:
                        continue
                    seendate = art.get('seendate', '')  # e.g. '20230415T120000Z'
                    # Parse to ISO date string
                    try:
                        pub_dt = seendate[:8]  # YYYYMMDD
                        published = f"{pub_dt[:4]}-{pub_dt[4:6]}-{pub_dt[6:8]}"
                    except Exception:
                        published = ''

                    raw_compound = analyzer.polarity_scores(title)['compound']
                    importance, pw_count = _article_importance(title)
                    power_bonus = min(pw_count / 3.0, 1.0)
                    weighted    = raw_compound * importance * (1.0 + power_bonus)
                    weighted    = max(-1.0, min(1.0, weighted))

                    rows.append({
                        'title':             title,
                        'published':         published,
                        'compound':          round(raw_compound, 4),
                        'importance':        round(importance, 2),
                        'power_word_count':  pw_count,
                        'weighted_compound': round(weighted, 4),
                        'sentiment': ('positive' if weighted >  0.05
                                      else ('negative' if weighted < -0.05
                                            else 'neutral')),
                        'feed':   query[:40],
                        'source': 'gdelt',
                    })
            except Exception as e:
                print(f"  [GDELT warn] {qs}→{qe} | {query[:30]}: {e}")

            if done % 20 == 0:
                print(f"  GDELT progress: {done}/{total} chunks | "
                      f"{len(rows)} articles ({len(seen_urls)} unique)")
            time.sleep(delay + random.uniform(0, 0.5))

    print(f"  GDELT done: {len(rows)} scored articles from {len(seen_urls)} unique URLs")
    if not rows:
        return pl.DataFrame(schema={
            'title': pl.Utf8, 'published': pl.Utf8, 'compound': pl.Float64,
            'importance': pl.Float64, 'power_word_count': pl.Int64,
            'weighted_compound': pl.Float64, 'sentiment': pl.Utf8,
            'feed': pl.Utf8, 'source': pl.Utf8,
        })
    return pl.DataFrame(rows).sort('published')


# --- SECTION 6: IMF API — Monetary Policy Indicators ---
# Replaces BOT API (website unstable). IMF is free, no key needed, covers all countries.
# Dataset: IFS (International Financial Statistics)
# Docs: https://datahelp.imf.org/knowledgebase/articles/667681

# --- BOT API (commented out — website down as of 2026-05) ---
# BOT_BASE_URL = 'https://apigw1.bot.or.th/bot/public'
# Portal: https://www.bot.or.th — restore this section when BOT API is back.

# Switched from SDMX endpoint (down) to IMF DataMapper API (stable, no key needed)
IMF_BASE_URL = 'https://www.imf.org/external/datamapper/api/v1'

IMF_INDICATORS = {
    'gdp_growth':  'NGDP_RPCH',  # Real GDP growth (%)
    'inflation':   'PCPIPCH',    # Inflation, CPI (%)
    'unemployment':'LUR',         # Unemployment rate (%)
    'current_acct':'BCA_NGDPD',  # Current account (% of GDP)
}

# DataMapper uses 3-letter ISO codes
IMF_COUNTRY_CODES = {
    'TH': 'THA', 'US': 'USA', 'CN': 'CHN', 'JP': 'JPN', 'DE': 'DEU', 'SG': 'SGP',
}

def fetch_imf(indicator_key: str, country='TH'):
    """
    Fetch an IMF DataMapper indicator for one country.
    Returns a polars DataFrame: year, value, country, indicator.
    """
    iso3 = IMF_COUNTRY_CODES.get(country, country)
    indicator_code = IMF_INDICATORS[indicator_key]
    url = f"{IMF_BASE_URL}/{indicator_code}/{iso3}"
    resp = req_lib.get(url, headers=BROWSER_HEADERS, timeout=10)
    resp.raise_for_status()

    values = resp.json().get('values', {}).get(indicator_code, {}).get(iso3, {})
    return pl.DataFrame([
        {'year': int(yr), 'value': val, 'country': country, 'indicator': indicator_key}
        for yr, val in values.items() if val is not None and int(yr) >= 2003
    ]).sort('year')


# --- SECTION 7: UN COMTRADE — Import / Export by Sector ---
# Free API, no key required for basic queries (500 req/hour limit).
# Thailand reporter code = 764
# flowCode: M = imports, X = exports

# UN Comtrade removed — now requires paid subscription.
# Trade data (exports/imports % of GDP) moved to ECONOMIC_INDICATORS via World Bank.


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def run_and_save(label, filename, fn, *args, **kwargs):
    print(f"\n=== {label} ===")
    try:
        df = fn(*args, **kwargs)
        df.write_csv(os.path.join(DATA_DIR, filename))
        print(df.tail(5))
    except Exception as e:
        print(f"[SKIP] {label}: {e}")

if __name__ == "__main__":
    run_and_save('Thailand Unemployment Rate', 'thailand_unemployment_rate.csv',
                 fetch_worldbank, 'SL.UEM.TOTL.ZS', country='TH')

    print("\n=== Market Signals (2003→present) ===")
    try:
        markets = fetch_market_signals()
        for name, df in markets.items():
            df.write_csv(os.path.join(DATA_DIR, f'{name}_market_signals.csv'))
            print(f"{name}: {df.tail(1)}")
    except Exception as e:
        print(f"[SKIP] Market Signals: {e}")

    run_and_save('News Sentiment (Bangkok Post)', 'news_sentiment.csv',
                 fetch_news_sentiment)

    print("\n=== GDELT Historical News (2015→2025) ===")
    try:
        gdelt_df = fetch_gdelt_news(start_year=2015, end_year=2025)
        gdelt_df.write_csv(os.path.join(DATA_DIR, 'news_gdelt.csv'))
        print(f"Saved {len(gdelt_df)} rows → news_gdelt.csv")
        print(gdelt_df.head(5))
    except Exception as e:
        print(f"[SKIP] GDELT: {e}")

    run_and_save('IMF GDP Growth (Thailand)', 'imf_gdp_growth_TH.csv',
                 fetch_imf, 'gdp_growth', country='TH')

    # --- 9 Economic Factors: new World Bank indicators ---
    WB_SAVES = {
        'consumption_pct_gdp':     'NE.CON.PRVT.ZS',
        'gross_capital_formation':  'NE.GDI.TOTL.ZS',
        'govt_expenditure_pct_gdp': 'GC.XPN.TOTL.GD.ZS',
        'govt_debt_pct_gdp':        'GC.DOD.TOTL.GD.ZS',
        'exports_pct_gdp':          'NE.EXP.GNFS.ZS',
        'imports_pct_gdp':          'NE.IMP.GNFS.ZS',
        'lending_rate':             'FR.INR.LEND',
        'inflation':                'FP.CPI.TOTL.ZG',
    }
    for fname, code in WB_SAVES.items():
        run_and_save(fname, f'{fname}_TH.csv', fetch_worldbank, code, country='TH')

    # --- FRED: Monthly data for Thailand (9 factors, high frequency) ---
    print("\n=== FRED Monthly Indicators (Thailand) ===")
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        print("[SKIP] FRED_API_KEY not set in .env")
    else:
        fred = Fred(api_key=fred_key)

        FRED_SERIES = {
            # --- Thailand-specific (monthly/quarterly) ---
            'th_exchange_rate_real': 'RBTHBIS',      # Real effective exchange rate (factor 6)
            'th_us_imports':         'IMP5490',       # US imports from Thailand (factor 6)
            'th_property_prices':    'QTHR628BIS',    # Bangkok property prices (factor 1)
            'th_uncertainty':        'WUITHA',        # World uncertainty index TH (factor 7)
            'usd_thb_monthly':       'EXTHUS',        # USD/THB monthly (fills 2000-2003 gap)

            # --- US/Global monthly (affects Thailand directly) ---
            # Factor 2: Interest rate
            'us_fed_funds_rate':     'FEDFUNDS',      # Fed rate → affects THB & capital flows
            # Factor 3: Inflation
            'us_cpi_monthly':        'CPIAUCSL',      # US CPI → global inflation signal
            # Factor 4: Unemployment
            'us_unemployment':       'UNRATE',        # US unemployment → global demand
            # Factor 5: Business investment
            'us_industrial_prod':    'INDPRO',        # US industrial production
            # Factor 7: Geopolitical / uncertainty
            'global_uncertainty':    'GEPUCURRENT',   # Global economic policy uncertainty
            # Factor 8: Technology
            'us_consumer_sentiment': 'UMCSENT',       # Consumer sentiment → spending intent
            # Factor 9: Government policy
            'us_govt_spending':      'FGEXPND',       # US federal expenditure (quarterly)
        }

        for name, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=DATA_START)
                df = pl.DataFrame({
                    'date':  [str(d.date()) for d in s.index],
                    'value': s.values.tolist(),
                    'series': name,
                }).drop_nulls()
                df.write_csv(os.path.join(DATA_DIR, f'fred_{name}.csv'))
                print(f"  OK {name}: {len(df)} rows → fred_{name}.csv")
            except Exception as e:
                print(f"  [Skip] {name}: {e}")
