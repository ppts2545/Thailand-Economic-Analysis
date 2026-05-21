"""
NLP Sentiment Pipeline for Thailand Economic Analysis
=====================================================
Replaces FRED uncertainty proxies with genuine NLP-derived sentiment.

Pipeline:
  1. GDELTFetcher   — historical Thailand financial news (2015–present)
  2. SentimentScorer — VADER (fast) + FinBERT (accurate, optional)
  3. SentimentAggregator — weekly feature engineering
  4. LeadLagTester   — Spearman cross-correlation at k = -4..+4

Usage (fetch + aggregate):
    python src/nlp_sentiment.py --fetch --years 2015 2025
    python src/nlp_sentiment.py --aggregate
    python src/nlp_sentiment.py --leadlag
"""

import argparse
import datetime
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
GDELT_RAW   = RAW_DIR  / "news_gdelt.csv"
SENT_WEEKLY = PROC_DIR / "sentiment_weekly.csv"

# ── GDELT configuration ────────────────────────────────────────────────────
GDELT_URL = "http://api.gdeltproject.org/api/v2/doc/doc"  # HTTP avoids SSL timeout on historical queries

# 4 focused queries targeting ENGLISH sources covering Thailand finance
# domain: prefix forces specific outlets; fallback queries are broader
GDELT_QUERIES = [
    # Bangkok Post — English, Thailand-specific, authoritative
    "domain:bangkokpost.com economy OR GDP OR SET OR baht OR rate OR inflation",
    # Reuters/Bloomberg/FT — global English press, high-signal
    "domain:reuters.com Thailand economy OR baht OR SET OR rate",
    "domain:bloomberg.com Thailand OR baht OR gold OR Federal Reserve",
    # Broad English fallback (includes CNBC, WSJ, FT etc.)
    "Thailand economy baht SET inflation Federal Reserve gold",
]

# Keyword importance weights (Loughran & McDonald 2011 + domain)
IMPORTANCE_MAP = {
    "federal reserve": 2.0, "fed ": 2.0, "rate hike": 2.0, "rate cut": 2.0,
    "interest rate": 1.8, "monetary policy": 1.8, "bank of thailand": 1.8,
    "quantitative easing": 1.8, "taper": 1.7, "pivot": 1.7,
    "recession": 2.0, "default": 2.0, "crisis": 1.8, "emergency": 1.8,
    "collapse": 2.0, "crash": 2.0, "panic": 1.7, "systemic": 1.8,
    "surge": 1.4, "soar": 1.4, "plunge": 1.5, "tumble": 1.5,
    "selloff": 1.5, "rally": 1.3, "record high": 1.4, "record low": 1.4,
    "trade war": 1.8, "tariff": 1.5, "sanction": 1.8, "war": 1.7,
    "inflation": 1.5, "deflation": 1.5, "stagflation": 1.8,
    "baht": 1.4, "set index": 1.3, "gdp": 1.3, "export": 1.2,
    "rumor": 0.5, "unconfirmed": 0.5, "speculation": 0.6, "reportedly": 0.7,
}


def _importance(text: str) -> tuple[float, int]:
    t = text.lower()
    max_imp, pw = 1.0, 0
    for word, imp in IMPORTANCE_MAP.items():
        if word in t:
            if imp > 1.0:
                pw += 1
            max_imp = max(max_imp, imp)
    return max_imp, pw


# ══════════════════════════════════════════════════════════════════════════════
# 1. GDELT FETCHER
# ══════════════════════════════════════════════════════════════════════════════

class GDELTFetcher:
    """
    Fetches Thailand financial news from GDELT DOC API v2.

    Key fixes vs original fetch_data.py:
    - 6s minimum delay (API enforces 5s; we add buffer)
    - 45s timeout (SSL handshake slow for historical queries)
    - Handles empty body (API returns HTTP 200 + empty body when rate-limited)
    - Saves progress after each quarter (resumable)
    - Exponential backoff on errors
    - Monthly chunks for 2020+, quarterly for 2015–2019
    """

    DELAY    = 6.0   # seconds between requests (API requires ≥ 5s)
    TIMEOUT  = 45    # seconds; old queries need longer SSL handshake
    MAX_ARTS = 250   # GDELT hard limit per request

    def __init__(self, out_path: Path = GDELT_RAW):
        self.out_path = out_path
        self.vader    = SentimentIntensityAnalyzer()
        self.seen_urls: set = set()

        # Load existing data to resume from where we left off
        self._existing = pd.DataFrame()
        if out_path.exists() and out_path.stat().st_size > 100:
            try:
                self._existing = pd.read_csv(out_path)
                if "url" in self._existing.columns:
                    self.seen_urls = set(self._existing["url"].dropna())
                print(f"[Resume] Loaded {len(self._existing)} existing rows, "
                      f"{len(self.seen_urls)} unique URLs")
            except Exception as e:
                print(f"[Warn] Could not load existing data: {e}")

    # ── Date windows ────────────────────────────────────────────────────────
    @staticmethod
    def _build_windows(start_year: int, end_year: int) -> list[tuple]:
        """
        Monthly chunks throughout (shorter windows = more reliable GDELT responses).
        Pre-2019 data is sparse on GDELT DOC API; caller should start from 2019.
        """
        today = datetime.date.today()
        windows = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                qs = datetime.date(year, month, 1)
                if month == 12:
                    qe = datetime.date(year, 12, 31)
                else:
                    qe = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
                if qs > today:
                    return windows
                qe = min(qe, today)
                windows.append((qs, qe))
        return windows

    # ── Single API call with retry ───────────────────────────────────────────
    def _fetch_one(self, query: str, start_dt: str, end_dt: str,
                   attempt: int = 0) -> list[dict]:
        params = {
            "query":         query,
            "mode":          "artlist",
            "format":        "json",
            "startdatetime": start_dt,
            "enddatetime":   end_dt,
            "maxrecords":    self.MAX_ARTS,
            "sourcelang":    "English",  # capital-E: GDELT language code
        }
        try:
            r = requests.get(GDELT_URL, params=params,
                             timeout=self.TIMEOUT,
                             headers={"User-Agent": "Mozilla/5.0"})
            body = r.text.strip() if r.text else ""

            # Rate-limit message
            if "limit requests" in body.lower() or "contact kalev" in body.lower():
                wait = self.DELAY * (2 ** min(attempt, 3))
                print(f"  [RateLimit] Backing off {wait:.0f}s …")
                time.sleep(wait)
                return self._fetch_one(query, start_dt, end_dt, attempt + 1)

            if not body or r.status_code != 200:
                return []

            data = json.loads(body)
            return data.get("articles", []) or []

        except requests.exceptions.Timeout:
            if attempt < 2:
                wait = self.DELAY * (2 ** attempt)
                print(f"  [Timeout] Retry {attempt+1} in {wait:.0f}s …")
                time.sleep(wait)
                return self._fetch_one(query, start_dt, end_dt, attempt + 1)
            return []
        except Exception as e:
            print(f"  [Error] {e}")
            return []

    # ── Score one article ────────────────────────────────────────────────────
    def _score(self, art: dict, query: str) -> dict | None:
        url   = art.get("url", "")
        title = (art.get("title", "") or "").strip()
        if not title or url in self.seen_urls:
            return None
        # Skip non-English (< 60% printable ASCII)
        if title and sum(ord(c) < 128 for c in title) / len(title) < 0.60:
            return None

        self.seen_urls.add(url)
        seendate = art.get("seendate", "")
        try:
            pub = f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}"
        except Exception:
            pub = ""

        raw  = self.vader.polarity_scores(title)["compound"]
        imp, pw = _importance(title)
        bonus   = min(pw / 3.0, 1.0)
        weighted = max(-1.0, min(1.0, raw * imp * (1.0 + bonus)))

        return {
            "url":               url,
            "title":             title,
            "published":         pub,
            "compound":          round(raw, 4),
            "importance":        round(imp, 2),
            "power_word_count":  pw,
            "weighted_compound": round(weighted, 4),
            "sentiment": ("positive" if weighted > 0.05
                          else ("negative" if weighted < -0.05 else "neutral")),
            "query":             query[:50],
        }

    # ── Main fetch ───────────────────────────────────────────────────────────
    def fetch(self, start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
        """
        Fetch and score GDELT articles for the given year range.
        Saves progress after each quarter; safe to restart.
        """
        windows = self._build_windows(start_year, end_year)
        total   = len(windows) * len(GDELT_QUERIES)
        done    = 0
        rows: list[dict] = []

        print(f"[GDELT] {len(windows)} windows × {len(GDELT_QUERIES)} queries "
              f"= {total} requests (~{total * self.DELAY / 60:.0f} min)")

        for qs, qe in windows:
            start_dt = qs.strftime("%Y%m%d") + "000000"
            end_dt   = qe.strftime("%Y%m%d") + "235959"
            window_rows: list[dict] = []

            for query in GDELT_QUERIES:
                done += 1
                arts = self._fetch_one(query, start_dt, end_dt)
                for art in arts:
                    row = self._score(art, query)
                    if row:
                        window_rows.append(row)

                pct = done / total * 100
                print(f"  [{pct:4.0f}%] {qs}→{qe} | +{len(window_rows)} new "
                      f"| total={len(self._existing)+len(rows)+len(window_rows)}",
                      end="\r")
                time.sleep(self.DELAY + random.uniform(0, 0.5))

            rows.extend(window_rows)

            # Save progress after every window
            if rows or not self._existing.empty:
                combined = pd.concat(
                    [self._existing, pd.DataFrame(rows)],
                    ignore_index=True
                ).drop_duplicates(subset=["url"])
                combined.to_csv(self.out_path, index=False)

        print(f"\n[GDELT] Done. Total articles: "
              f"{len(self._existing) + len(rows)} unique")
        final = pd.concat(
            [self._existing, pd.DataFrame(rows)], ignore_index=True
        ).drop_duplicates(subset=["url"])
        final.to_csv(self.out_path, index=False)
        return final


# ══════════════════════════════════════════════════════════════════════════════
# 2. FINBERT SCORER  (optional — requires GPU for practical speed)
# ══════════════════════════════════════════════════════════════════════════════

class FinBERTScorer:
    """
    Scores article titles with ProsusAI/finbert (3-class: positive/negative/neutral).
    Falls back gracefully if transformers not available or no CUDA.

    Only recommended for a SAMPLE of articles or a GPU environment.
    CPU inference: ~0.5s per title → 100k articles = 14 hours.
    GPU (T4):      ~0.02s per title → 100k articles = 33 minutes.
    """

    MODEL_ID = "ProsusAI/finbert"

    def __init__(self):
        self.available = False
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                device=device,
                truncation=True,
                max_length=128,
            )
            self.available = True
            dev_str = "GPU" if device == 0 else "CPU"
            print(f"[FinBERT] Loaded on {dev_str} ({self.MODEL_ID})")
        except Exception as e:
            print(f"[FinBERT] Not available: {e}. Will use VADER only.")

    def score_batch(self, titles: list[str], batch_size: int = 32) -> list[float]:
        """
        Returns a list of signed compound scores:
          positive → +confidence
          negative → -confidence
          neutral  → 0
        Matches VADER compound range [-1, +1].
        """
        if not self.available:
            return [0.0] * len(titles)

        results = []
        for i in range(0, len(titles), batch_size):
            batch = titles[i : i + batch_size]
            out   = self._pipe(batch)
            for item in out:
                label = item["label"].lower()
                score = item["score"]
                if label == "positive":
                    results.append(+score)
                elif label == "negative":
                    results.append(-score)
                else:
                    results.append(0.0)
        return results

    def rescore_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add finbert_compound column to raw GDELT DataFrame."""
        if not self.available:
            df["finbert_compound"] = df["compound"]
            return df
        print(f"[FinBERT] Scoring {len(df)} titles …")
        df = df.copy()
        df["finbert_compound"] = self.score_batch(df["title"].tolist())
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. WEEKLY AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAggregator:
    """
    Converts raw article-level sentiment scores into weekly features
    aligned to the Monday–Friday trading week.

    All features are lagged 1 week (shift=1) so they are available
    at Monday open without look-ahead bias.

    Weekly features created:
    ┌─────────────────────────────┬──────────────────────────────────────────┐
    │ sent_mean_1w                │ Mean weighted compound this week          │
    │ sent_mean_vader_1w          │ Mean raw VADER compound this week         │
    │ sent_vol_4w                 │ Rolling 4w std of sent_mean (fear proxy)  │
    │ sent_mom4w                  │ 4-week MA of sentiment (trend)            │
    │ sent_surprise               │ sent_mean_1w − sent_mom4w (deviation)     │
    │ news_vol_w                  │ Article count this week                   │
    │ sent_positive_ratio         │ Fraction of positive articles             │
    │ sent_negative_ratio         │ Fraction of negative articles             │
    │ high_impact_ratio           │ Fraction with importance > 1.4            │
    │ sent_spike                  │ 1 if |sent_mean_1w z-score| > 2           │
    │ finbert_mean_1w             │ Mean FinBERT score (if available)         │
    └─────────────────────────────┴──────────────────────────────────────────┘
    """

    def aggregate(self, raw: pd.DataFrame,
                  use_finbert: bool = False) -> pd.DataFrame:
        """
        raw: DataFrame with columns [published, weighted_compound, compound,
             sentiment, importance, finbert_compound (optional)]

        Returns weekly DataFrame indexed by Friday (end of trading week),
        all features shifted 1 week to prevent look-ahead bias.
        """
        df = raw.copy()
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        df = df.dropna(subset=["published"])
        df = df[df["published"].dt.year >= 2014]

        # Snap to week-ending Friday
        df["week"] = df["published"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

        # ── Weekly aggregations ────────────────────────────────────────────
        def _agg(g: pd.DataFrame) -> pd.Series:
            wc  = g["weighted_compound"]
            c   = g["compound"]
            imp = g["importance"]
            pos = (g["sentiment"] == "positive").sum()
            neg = (g["sentiment"] == "negative").sum()
            n   = len(g)
            out = {
                "sent_mean_1w":        wc.mean(),
                "sent_mean_vader_1w":  c.mean(),
                "news_vol_w":          n,
                "sent_positive_ratio": pos / n if n else 0.5,
                "sent_negative_ratio": neg / n if n else 0.5,
                "high_impact_ratio":   (imp > 1.4).sum() / n if n else 0,
            }
            if use_finbert and "finbert_compound" in g.columns:
                out["finbert_mean_1w"] = g["finbert_compound"].mean()
            return pd.Series(out)

        weekly = df.groupby("week").apply(_agg).sort_index()
        weekly.index = pd.DatetimeIndex(weekly.index)

        # Reindex to full weekly calendar (fill gaps with neutral values)
        full_idx = pd.date_range(
            start=weekly.index.min(), end=weekly.index.max(), freq="W-FRI"
        )
        weekly = weekly.reindex(full_idx)
        weekly["news_vol_w"]          = weekly["news_vol_w"].fillna(0)
        weekly["sent_positive_ratio"] = weekly["sent_positive_ratio"].fillna(0.5)
        weekly["sent_negative_ratio"] = weekly["sent_negative_ratio"].fillna(0.5)
        weekly["high_impact_ratio"]   = weekly["high_impact_ratio"].fillna(0)
        for col in ["sent_mean_1w", "sent_mean_vader_1w"]:
            if col in weekly.columns:
                weekly[col] = weekly[col].fillna(0)

        # ── Derived features ───────────────────────────────────────────────
        s = weekly["sent_mean_1w"]
        weekly["sent_vol_4w"]   = s.rolling(4, min_periods=2).std()
        weekly["sent_mom4w"]    = s.rolling(4, min_periods=2).mean()
        weekly["sent_surprise"] = s - weekly["sent_mom4w"]

        # Spike: |z-score| > 2 (52-week rolling)
        roll_mean = s.rolling(52, min_periods=8).mean()
        roll_std  = s.rolling(52, min_periods=8).std().replace(0, np.nan)
        weekly["sent_spike"] = ((s - roll_mean).abs() / roll_std > 2).astype(float)

        # ── Shift 1 week for look-ahead protection ─────────────────────────
        for col in weekly.columns:
            weekly[col] = weekly[col].shift(1)

        weekly.index.name = "date"
        return weekly.dropna(how="all")

    def save(self, weekly: pd.DataFrame, path: Path = SENT_WEEKLY) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(path)
        print(f"[Aggregator] Saved {len(weekly)} weekly rows → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3b. BANGKOK POST CDX FETCHER (reliable backup — no API key, no rate limit issues)
# ══════════════════════════════════════════════════════════════════════════════

CDX_URL = "https://web.archive.org/cdx/search/cdx"

# Bangkok Post sections covering Thai financial news
# Note: finance/* and investment/* are not reliably archived by Wayback Machine
BP_SECTIONS = [
    "bangkokpost.com/business/*",
    "bangkokpost.com/economy/*",
]

# Financial stop-words to strip from URL slugs before VADER
_SLUG_STOP = {
    "the","and","for","with","from","that","this","are","has","have",
    "been","will","was","were","its","into","over","after","amid","but",
    "not","all","new","than","more","also","what","says","said","told",
    "set","net","per","unit","year","month","week","day","baht","post",
    "bangkok","thailand","thai","news","report","data","amid","sees",
}


class BangkokPostCDXFetcher:
    """
    Fetches historical Bangkok Post article URLs via Wayback Machine CDX API.

    Strategy:
    - CDX API returns archived URLs grouped by month (no rate limit documented)
    - URL slugs (e.g. 'gdp-growth-disappoints-exports-fall') carry headline words
    - VADER on slug words gives directional sentiment signal
    - Each CDX call returns up to 10,000 records (we cap at 1,000 per month)

    Coverage: 2015–present, no API key needed.
    Speed:    ~1 min for 10 years (CDX calls are fast; no per-article fetches).
    Quality:  Slug-based ≈ 70% of full-headline accuracy (key nouns/verbs present).
    """

    CDX_DELAY = 0.5   # polite delay between CDX API calls
    MAX_RECS  = 1000  # per month per section

    def __init__(self, out_path: Path = GDELT_RAW):
        self.out_path  = out_path
        self.vader     = SentimentIntensityAnalyzer()
        self.seen_urls: set = set()

        # Load existing data to resume without re-fetching
        if out_path.exists() and out_path.stat().st_size > 100:
            try:
                existing = pd.read_csv(out_path)
                if "url" in existing.columns:
                    self.seen_urls = set(existing["url"].dropna())
                print(f"[BP-CDX] Resuming — {len(self.seen_urls)} URLs already collected")
            except Exception:
                pass

    # ── Slug → clean text ────────────────────────────────────────────────────
    @staticmethod
    def _slug_to_text(url: str) -> str:
        """Extract and clean the URL slug from a Bangkok Post article URL."""
        path = url.split("bangkokpost.com")[-1]
        parts = [p for p in path.split("/") if p and not p.isdigit()]
        slug  = parts[-1] if parts else ""
        words = [w for w in slug.replace("-", " ").split()
                 if w not in _SLUG_STOP and len(w) > 2]
        return " ".join(words)

    # ── Single CDX call ──────────────────────────────────────────────────────
    def _fetch_cdx(self, section: str, from_yyyymm: str, to_yyyymm: str) -> list[dict]:
        # No collapse=urlkey: that caused CDX to return same URLs across all months.
        # Instead we filter by timestamp client-side to get only that month's snapshots.
        params = {
            "url":       section,
            "output":    "json",
            "limit":     self.MAX_RECS,
            "from":      from_yyyymm,
            "to":        to_yyyymm,
            "fl":        "timestamp,original",
            "filter":    "statuscode:200",
            "matchType": "prefix",
        }
        try:
            r = requests.get(CDX_URL, params=params, timeout=30)
            rows = r.json()
            return rows[1:] if rows else []  # skip header row
        except Exception as e:
            print(f"  [CDX warn] {section[:30]} {from_yyyymm}: {e}")
            return []

    def _score_row(self, ts: str, url: str, section: str) -> dict | None:
        """Score a single CDX record. Returns None if URL already seen or invalid."""
        if not url or "bangkokpost.com" not in url:
            return None
        if url in self.seen_urls:
            return None
        # Skip category pages (slug must have ≥ 5 path segments)
        path_parts = [p for p in url.split("/") if p]
        if len(path_parts) < 5:
            return None
        text = self._slug_to_text(url)
        if not text or len(text.split()) < 2:
            return None
        self.seen_urls.add(url)
        try:
            pub = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        except Exception:
            return None
        raw     = self.vader.polarity_scores(text)["compound"]
        imp, pw = _importance(text)
        bonus   = min(pw / 3.0, 1.0)
        weighted = max(-1.0, min(1.0, raw * imp * (1.0 + bonus)))
        return {
            "url":               url,
            "title":             text,
            "published":         pub,
            "compound":          round(raw, 4),
            "importance":        round(imp, 2),
            "power_word_count":  pw,
            "weighted_compound": round(weighted, 4),
            "sentiment": ("positive" if weighted > 0.05
                          else ("negative" if weighted < -0.05 else "neutral")),
            "query":             section.split("/")[0],
            "source":            "bangkokpost_cdx",
        }

    # ── Main fetch ───────────────────────────────────────────────────────────
    def fetch(self, start_year: int = 2019, end_year: int = 2025) -> pd.DataFrame:
        """
        Fetch slug-based sentiment for Bangkok Post articles via Wayback CDX.
        - Filters CDX results by snapshot timestamp to avoid returning stale URLs.
        - Deduplicates by URL across all months (seen_urls initialized from existing data).
        - Saves incrementally after each month (resumable).
        """
        today = datetime.date.today()
        new_rows: list[dict] = []

        total_months = (end_year - start_year + 1) * 12
        done = 0

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if datetime.date(year, month, 1) > today:
                    break
                # CDX timestamp range for this month
                month_prefix = f"{year}{month:02d}"
                from_ym = f"{month_prefix}01000000"
                to_ym   = f"{month_prefix}31235959"
                done += 1
                month_new = 0

                for section in BP_SECTIONS:
                    records = self._fetch_cdx(section, from_ym, to_ym)
                    for rec in records:
                        if len(rec) < 2:
                            continue
                        ts, url = rec[0], rec[1]
                        # Only accept snapshots actually from this month
                        if not ts.startswith(month_prefix):
                            continue
                        row = self._score_row(ts, url, section)
                        if row:
                            new_rows.append(row)
                            month_new += 1
                    time.sleep(self.CDX_DELAY)

                pct = done / total_months * 100
                print(f"  [{pct:4.0f}%] {year}-{month:02d} | "
                      f"+{month_new} new | total={len(new_rows)}", end="\r")

                # Save incrementally after every month
                if new_rows and done % 3 == 0:
                    self._save_incremental(new_rows)

        print(f"\n[BP-CDX] Done: {len(new_rows)} new articles")
        return self._save_incremental(new_rows)

    def _save_incremental(self, new_rows: list[dict]) -> pd.DataFrame:
        """Merge new_rows with existing file and save."""
        if not new_rows:
            return pd.read_csv(self.out_path) if self.out_path.exists() else pd.DataFrame()
        new_df = pd.DataFrame(new_rows)
        if self.out_path.exists() and self.out_path.stat().st_size > 100:
            try:
                existing = pd.read_csv(self.out_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["url"])
            except Exception:
                combined = new_df
        else:
            combined = new_df
        combined = combined.sort_values("published")
        combined.to_csv(self.out_path, index=False)
        return combined


# ══════════════════════════════════════════════════════════════════════════════
# 3c. BANGKOK POST SITEMAP FETCHER (recent full-title articles — no slugs)
# ══════════════════════════════════════════════════════════════════════════════

BP_SITEMAP_URLS = [
    "https://www.bangkokpost.com/sitemap/sitemap_business.xml",
    "https://www.bangkokpost.com/sitemap/sitemap_economy.xml",
]

# ── Google News RSS fetcher ─────────────────────────────────────────────────
GNEWS_QUERIES = [
    "Thailand economy baht SET index",
    "Thai baht GDP inflation Bank of Thailand",
    "Thailand gold Federal Reserve rate",
    "Thailand exports imports trade tourism",
    "Thailand recession crisis financial",
    "SET index rally selloff Thai stocks",
    "USD THB exchange rate devaluation",
    "Thailand interest rate monetary policy",
    "Thailand economic growth forecast",
    "Thailand budget deficit surplus spending",
]


class BangkokPostSitemapFetcher:
    """
    Fetches recent Bangkok Post articles from their XML sitemaps.
    Returns full article titles (not truncated URL slugs).
    Coverage: ~100 most recent articles per section (live feed).
    Use this to supplement CDX data with higher-quality recent titles.
    """

    def __init__(self, out_path: Path = GDELT_RAW):
        self.out_path  = out_path
        self.vader     = SentimentIntensityAnalyzer()
        self.seen_urls: set = set()
        if out_path.exists() and out_path.stat().st_size > 100:
            try:
                existing = pd.read_csv(out_path)
                if "url" in existing.columns:
                    self.seen_urls = set(existing["url"].dropna())
            except Exception:
                pass

    def fetch(self) -> pd.DataFrame:
        import xml.etree.ElementTree as ET
        new_rows: list[dict] = []

        for sitemap_url in BP_SITEMAP_URLS:
            try:
                r = requests.get(sitemap_url, timeout=20,
                                 headers={"User-Agent": "Mozilla/5.0"})
                root = ET.fromstring(r.text)
                ns   = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
                        "news": "http://www.google.com/schemas/sitemap-news/0.9"}
                for url_el in root.findall(".//sm:url", ns):
                    loc   = (url_el.findtext("sm:loc", "", ns) or "").strip()
                    title = (url_el.findtext(".//news:title", "", ns) or "").strip()
                    pub   = (url_el.findtext(".//news:publication_date", "", ns)
                             or url_el.findtext("sm:lastmod", "", ns) or "").strip()

                    if not loc or loc in self.seen_urls:
                        continue
                    if not title:
                        continue
                    # Skip non-English titles (< 60% ASCII)
                    if sum(ord(c) < 128 for c in title) / len(title) < 0.60:
                        continue
                    self.seen_urls.add(loc)

                    try:
                        pub_date = pub[:10]  # YYYY-MM-DD
                    except Exception:
                        pub_date = ""

                    raw     = self.vader.polarity_scores(title)["compound"]
                    imp, pw = _importance(title)
                    bonus   = min(pw / 3.0, 1.0)
                    weighted = max(-1.0, min(1.0, raw * imp * (1.0 + bonus)))
                    new_rows.append({
                        "url":               loc,
                        "title":             title,
                        "published":         pub_date,
                        "compound":          round(raw, 4),
                        "importance":        round(imp, 2),
                        "power_word_count":  pw,
                        "weighted_compound": round(weighted, 4),
                        "sentiment": ("positive" if weighted > 0.05
                                      else ("negative" if weighted < -0.05 else "neutral")),
                        "query":             "bangkokpost_sitemap",
                        "source":            "bangkokpost_sitemap",
                    })
                print(f"[Sitemap] {sitemap_url.split('/')[-1]}: {len(new_rows)} articles")
                time.sleep(1.0)
            except Exception as e:
                print(f"[Sitemap warn] {sitemap_url}: {e}")

        if not new_rows:
            print("[Sitemap] No new articles found")
            return pd.DataFrame()

        new_df = pd.DataFrame(new_rows)
        if self.out_path.exists() and self.out_path.stat().st_size > 100:
            try:
                existing = pd.read_csv(self.out_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["url"])
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined = combined.sort_values("published")
        combined.to_csv(self.out_path, index=False)
        print(f"[Sitemap] Saved {len(new_rows)} new articles → {self.out_path.name} "
              f"(total {len(combined)})")
        return combined


# ══════════════════════════════════════════════════════════════════════════════
# 3d. GOOGLE NEWS RSS FETCHER (no rate limit, free, recent + some historical)
# ══════════════════════════════════════════════════════════════════════════════

class GoogleNewsFetcher:
    """
    Fetches Thailand financial news from Google News RSS (no API key needed).
    Typically returns ~50-100 articles per query, mostly recent, some historical.
    Full article titles, reliable publication dates.

    Use as fallback when GDELT/CDX APIs are throttled.
    """

    GNEWS_BASE = "https://news.google.com/rss/search"
    DELAY      = 0.5

    def __init__(self, out_path: Path = GDELT_RAW):
        self.out_path  = out_path
        self.vader     = SentimentIntensityAnalyzer()
        self.seen_urls: set = set()
        if out_path.exists() and out_path.stat().st_size > 100:
            try:
                existing = pd.read_csv(out_path)
                if "url" in existing.columns:
                    self.seen_urls = set(existing["url"].dropna())
            except Exception:
                pass

    def _fetch_rss(self, query: str) -> list[dict]:
        from email.utils import parsedate_to_datetime
        import xml.etree.ElementTree as ET
        url = self.GNEWS_BASE + f"?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(r.text)
            rows = []
            for item in root.findall(".//item"):
                title = (item.findtext("title", "") or "").strip()
                pub   = (item.findtext("pubDate", "") or "").strip()
                link  = (item.findtext("link",    "") or "").strip()
                if not title or not link or link in self.seen_urls:
                    continue
                # Skip non-English titles (< 60% ASCII)
                if sum(ord(c) < 128 for c in title) / max(len(title), 1) < 0.60:
                    continue
                try:
                    dt       = parsedate_to_datetime(pub)
                    pub_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    continue
                self.seen_urls.add(link)
                raw     = self.vader.polarity_scores(title)["compound"]
                imp, pw = _importance(title)
                bonus   = min(pw / 3.0, 1.0)
                weighted = max(-1.0, min(1.0, raw * imp * (1.0 + bonus)))
                rows.append({
                    "url":               link,
                    "title":             title,
                    "published":         pub_date,
                    "compound":          round(raw, 4),
                    "importance":        round(imp, 2),
                    "power_word_count":  pw,
                    "weighted_compound": round(weighted, 4),
                    "sentiment": ("positive" if weighted > 0.05
                                  else ("negative" if weighted < -0.05 else "neutral")),
                    "query":             query[:50],
                    "source":            "google_news",
                })
            return rows
        except Exception as e:
            print(f"  [GNews warn] {query[:40]}: {e}")
            return []

    def fetch(self, queries: list[str] | None = None) -> pd.DataFrame:
        if queries is None:
            queries = GNEWS_QUERIES
        new_rows: list[dict] = []
        for i, q in enumerate(queries):
            rows = self._fetch_rss(q)
            new_rows.extend(rows)
            print(f"  [{i+1}/{len(queries)}] '{q[:45]}' → {len(rows)} new")
            time.sleep(self.DELAY)

        if not new_rows:
            print("[GNews] No new articles")
            return pd.read_csv(self.out_path) if self.out_path.exists() else pd.DataFrame()

        new_df = pd.DataFrame(new_rows)
        if self.out_path.exists() and self.out_path.stat().st_size > 100:
            try:
                existing = pd.read_csv(self.out_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["url"])
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined = combined.sort_values("published")
        combined.to_csv(self.out_path, index=False)
        print(f"[GNews] Saved {len(new_rows)} new articles → total {len(combined)}")
        return combined


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEAD-LAG TESTER
# ══════════════════════════════════════════════════════════════════════════════

class LeadLagTester:
    """
    Tests whether each sentiment feature leads, coincides with, or lags
    the target returns using Spearman cross-correlation at k = -4..+4 weeks.

    Threshold for 'useful leading signal': |r| ≥ 0.04 at k ≥ +1
    """

    TARGETS_1W = ["SET_index_ret_w", "gold_ret_w", "USD_THB_ret_w"]
    LAGS       = list(range(-4, 5))
    MIN_R      = 0.04   # gate threshold

    def run(self, sent_weekly: pd.DataFrame,
            market_weekly: pd.DataFrame) -> pd.DataFrame:
        """
        sent_weekly: output of SentimentAggregator.aggregate()
        market_weekly: unified_weekly.csv (must share date index)

        Returns DataFrame of shape (n_features × n_targets × n_lags)
        with columns: feature, target, lag, spearman_r, p_value
        """
        # Align
        common = sent_weekly.index.intersection(market_weekly.index)
        s = sent_weekly.loc[common]
        m = market_weekly.loc[common]

        records = []
        sent_cols = [c for c in s.columns if c not in self.TARGETS_1W]

        for feat in sent_cols:
            for target in self.TARGETS_1W:
                if target not in m.columns:
                    continue
                for k in self.LAGS:
                    # k > 0: sentiment shifted forward → leads returns
                    sent_shifted = s[feat].shift(k)
                    tmp = pd.concat([m[target], sent_shifted], axis=1).dropna()
                    if len(tmp) < 30:
                        continue
                    r, p = spearmanr(tmp.iloc[:, 0], tmp.iloc[:, 1])
                    records.append({
                        "feature": feat, "target": target,
                        "lag": k, "spearman_r": r, "p_value": p,
                    })

        return pd.DataFrame(records)

    def print_summary(self, ll_df: pd.DataFrame) -> dict:
        """
        Prints lead-lag table and returns gate_passed dict
        {feature: True/False} based on |r| >= MIN_R at k >= +1.
        """
        print("\n" + "=" * 72)
        print("LEAD-LAG ANALYSIS  (k>0 = sentiment LEADS returns)")
        print(f"Gate threshold: |r| ≥ {self.MIN_R} at k ≥ +1")
        print("=" * 72)

        gate_passed: dict[str, bool] = {}

        for feat in ll_df["feature"].unique():
            sub = ll_df[ll_df["feature"] == feat]
            max_lead_r = sub[sub["lag"] >= 1]["spearman_r"].abs().max()
            gate_passed[feat] = bool(max_lead_r >= self.MIN_R)
            tag = "★ LEADS" if gate_passed[feat] else "  lags "
            print(f"\n  {tag}  [{feat}]")
            header = "  " + "  ".join(f"k={k:+d}" for k in self.LAGS)
            print(f"  {'Target':25s}{header}")
            for tgt in self.TARGETS_1W:
                row_df = sub[sub["target"] == tgt].sort_values("lag")
                if row_df.empty:
                    continue
                vals = "  ".join(f"{r:+.3f}" for r in row_df["spearman_r"])
                best_k = int(row_df.loc[row_df["spearman_r"].abs().idxmax(), "lag"])
                print(f"  {tgt:25s}  {vals}  ← peak k={best_k:+d}")

        n_pass = sum(gate_passed.values())
        print(f"\n  Gate passed: {n_pass}/{len(gate_passed)} features have "
              f"leading signal |r| ≥ {self.MIN_R}")

        return gate_passed


# ══════════════════════════════════════════════════════════════════════════════
# 5. ABLATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    sent_weekly: pd.DataFrame,
    market_weekly: pd.DataFrame,
    leading_features: list[str],
    n_trials: int = 20,
) -> pd.DataFrame:
    """
    Compares 3 models:
      A) Macro + Market + NLP  (full)
      B) Macro + Market        (no NLP)
      C) Macro only            (baseline)

    Returns DataFrame with DirAcc, R², Sharpe for each model × target.
    """
    import xgboost as xgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    TARGETS = ["SET_index_ret_w", "gold_ret_w", "USD_THB_ret_w"]
    TSCV    = TimeSeriesSplit(n_splits=5)
    WEEKS_PER_YEAR = 52

    # Macro-only features (L1)
    MACRO_COLS = [
        "us_industrial_prod", "us_cpi_monthly", "us_fed_funds_rate",
        "us_unemployment", "gdp_growth_annual", "inflation_annual",
        "us_govt_spending", "th_exchange_rate_real",
    ]
    # Market features (L2)
    MARKET_COLS = [
        "sp500_ret_w", "nasdaq_ret_w", "vix_price", "dxy_ret_w",
        "gold_ret_w", "oil_ret_w", "us_10yr_treasury_ret_w",
        "sp500_ret_d_lag1", "nasdaq_ret_d_lag1", "eem_ret_d_lag1",
        "eem_vs_sp500_lag1", "em_outflow_pressure", "risk_on_signal",
        "set_above_ma52w", "real_yield", "yield_curve",
        "vix_velocity", "set_deep_drawdown",
        # regime interactions
        "sp500_ret_w__x__is_bull", "sp500_ret_w__x__is_crisis",
        "vix_price_z__x__is_crisis", "dxy_ret_w__x__is_crisis",
    ]

    # Combine into aligned DataFrame
    common = sent_weekly.index.intersection(market_weekly.index)
    s = sent_weekly.loc[common, leading_features].copy()
    m = market_weekly.loc[common].copy()

    # Build engineered features on market_weekly
    if "vix_price" in m.columns:
        v = m["vix_price"]
        p33, p67 = v.quantile(0.33), v.quantile(0.67)
        m["is_bull"]   = (v < p33).astype(float)
        m["is_normal"] = ((v >= p33) & (v < p67)).astype(float)
        m["is_crisis"] = (v >= p67).astype(float)
        m["vix_price_z"] = (v - v.rolling(52).mean()) / (v.rolling(52).std() + 1e-9)
    for sig, reg in [("sp500_ret_w","is_bull"),("sp500_ret_w","is_crisis"),
                     ("vix_price_z","is_crisis"),("dxy_ret_w","is_crisis")]:
        if sig in m.columns and reg in m.columns:
            m[f"{sig}__x__{reg}"] = m[sig] * m[reg]
    if "vix_price" in m.columns:
        m["vix_velocity"] = m["vix_price"].pct_change(4).shift(1)
    if "SET_index_price" in m.columns:
        px = m["SET_index_price"]
        m["set_above_ma52w"] = (px > px.rolling(52).mean()).astype(float).shift(1)
        hi52 = px.rolling(52).max()
        m["SET_index_pct_from_52w_high"] = (px / hi52 - 1).shift(1)
        m["set_deep_drawdown"] = (m["SET_index_pct_from_52w_high"] < -0.15).astype(float)
    if "eem_ret_d_lag1" in m.columns and "sp500_ret_d_lag1" in m.columns:
        m["eem_vs_sp500_lag1"] = m["eem_ret_d_lag1"] - m["sp500_ret_d_lag1"]
    _dxy_up = (m.get("dxy_ret_d_lag1", pd.Series(0, index=m.index)) > 0).astype(float)
    _vix_ma = m["vix_price"].rolling(4).mean() if "vix_price" in m.columns else pd.Series(20, index=m.index)
    _vix_up = (m.get("vix_price", pd.Series(20, index=m.index)) > _vix_ma).astype(float)
    _thb_up = (m.get("USD_THB_ret_d_lag1", pd.Series(0, index=m.index)) > 0).astype(float)
    _sp_up  = (m.get("sp500_ret_d_lag1", pd.Series(0, index=m.index)) > 0).astype(float)
    m["em_outflow_pressure"] = _dxy_up * _vix_up * _thb_up
    m["risk_on_signal"]      = _sp_up * (1 - _vix_up)
    if "us_10yr_treasury_yield" not in m.columns and "us_10yr_treasury_price" in m.columns:
        m["us_10yr_treasury_yield"] = -np.log(m["us_10yr_treasury_price"]).diff() * 100
    if "us_10yr_treasury_yield" in m.columns and "us_cpi_monthly" in m.columns:
        m["real_yield"] = m["us_10yr_treasury_yield"] - m["us_cpi_monthly"]
    if "us_10yr_treasury_yield" in m.columns and "us_fed_funds_rate" in m.columns:
        m["yield_curve"] = m["us_10yr_treasury_yield"] - m["us_fed_funds_rate"]

    full_df = pd.concat([m, s], axis=1)
    # Drop duplicate columns (can arise if a NLP feature name collides with market col)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]

    def _valid_cols(df, cols):
        seen: set = set()
        result = []
        for c in cols:
            if c in df.columns and c not in seen and df[c].isna().mean() < 0.35:
                result.append(c)
                seen.add(c)
        return result

    def _train_eval(X, y):
        # Convert to numpy to avoid XGBoost/Pandas 2.x dtype compatibility issues
        X_np = X.to_numpy().astype(np.float64)
        y_np = y.to_numpy().astype(np.float64)
        feat_names = list(X.columns)

        def _obj(trial):
            p = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
                "max_depth":        trial.suggest_int("max_depth", 2, 6),
                "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
                "subsample":        trial.suggest_float("ss", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("cbt", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("ra", 1e-3, 10, log=True),
                "reg_lambda":       trial.suggest_float("rl", 1e-3, 10, log=True),
                "tree_method": "hist", "verbosity": 0,
                "feature_names": feat_names,
            }
            scores = []
            for tr, te in TSCV.split(X_np):
                mod = xgb.XGBRegressor(**p)
                mod.fit(X_np[tr], y_np[tr])
                scores.append(r2_score(y_np[te], mod.predict(X_np[te])))
            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
        params = {**study.best_params, "tree_method": "hist", "verbosity": 0,
                  "feature_names": feat_names}

        r2s, das, preds_all, y_all = [], [], [], []
        for tr, te in TSCV.split(X_np):
            mod = xgb.XGBRegressor(**params)
            mod.fit(X_np[tr], y_np[tr])
            p_te = mod.predict(X_np[te])
            r2s.append(r2_score(y_np[te], p_te))
            das.append(np.mean(np.sign(p_te) == np.sign(y_np[te])))
            preds_all.extend(p_te)
            y_all.extend(y_np[te])

        strat = np.sign(np.array(preds_all)) * np.array(y_all)
        sharpe = (strat.mean() / strat.std()) * np.sqrt(WEEKS_PER_YEAR) if strat.std() > 0 else 0
        return np.mean(r2s), np.mean(das), sharpe

    records = []
    model_defs = [
        ("A: Macro+Market+NLP", MACRO_COLS + MARKET_COLS + leading_features),
        ("B: Macro+Market",     MACRO_COLS + MARKET_COLS),
        ("C: Macro only",       MACRO_COLS),
    ]

    for tgt in TARGETS:
        if tgt not in full_df.columns:
            continue
        print(f"\n  {tgt}")
        for model_name, feat_list in model_defs:
            # Exclude the target itself from the feature list to prevent look-ahead
            cols = _valid_cols(full_df, [c for c in feat_list if c != tgt])
            tmp  = full_df[[tgt] + cols].dropna()
            if len(tmp) < 60:
                print(f"    {model_name}: insufficient data")
                continue
            X, y = tmp[cols], tmp[tgt]
            r2, da, sh = _train_eval(X, y)
            records.append({
                "target": tgt, "model": model_name,
                "dir_acc": da, "r2": r2, "sharpe": sh, "n_feat": len(cols),
            })
            print(f"    {model_name:30s}  DirAcc={da:.1%}  R²={r2:+.4f}  "
                  f"Sharpe={sh:+.2f}  n_feat={len(cols)}")

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NLP Sentiment Pipeline")
    parser.add_argument("--fetch",    action="store_true", help="Fetch GDELT data")
    parser.add_argument("--bpcdx",   action="store_true",
                        help="Fetch Bangkok Post history via Wayback CDX (reliable fallback)")
    parser.add_argument("--sitemap", action="store_true",
                        help="Fetch recent Bangkok Post articles via XML sitemap (full titles)")
    parser.add_argument("--gnews",   action="store_true",
                        help="Fetch Thailand financial news via Google News RSS (fallback, no API key)")
    parser.add_argument("--finbert",  action="store_true", help="Re-score with FinBERT")
    parser.add_argument("--aggregate",action="store_true", help="Aggregate to weekly features")
    parser.add_argument("--leadlag",  action="store_true", help="Run lead-lag test")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--years",    nargs=2, type=int, default=[2019, 2025],
                        metavar=("START", "END"))
    args = parser.parse_args()

    if not any([args.fetch, args.bpcdx, args.sitemap, args.gnews, args.finbert,
                args.aggregate, args.leadlag, args.ablation]):
        parser.print_help()
        return

    # ── Google News RSS (fallback when GDELT/CDX are throttled) ──────────
    if args.gnews:
        fetcher = GoogleNewsFetcher(GDELT_RAW)
        fetcher.fetch()

    # ── Bangkok Post Sitemap (recent, full titles) ─────────────────────────
    if args.sitemap:
        fetcher = BangkokPostSitemapFetcher(GDELT_RAW)
        fetcher.fetch()

    # ── Bangkok Post CDX (reliable, fast) ─────────────────────────────────
    if args.bpcdx:
        fetcher = BangkokPostCDXFetcher(GDELT_RAW)
        fetcher.fetch(start_year=args.years[0], end_year=args.years[1])

    # ── GDELT fetch (use when API is accessible) ───────────────────────────
    if args.fetch:
        fetcher = GDELTFetcher(GDELT_RAW)
        fetcher.fetch(start_year=args.years[0], end_year=args.years[1])

    # ── FinBERT re-score ───────────────────────────────────────────────────
    if args.finbert:
        if not GDELT_RAW.exists():
            print("[Error] Run --fetch first")
            return
        raw = pd.read_csv(GDELT_RAW)
        scorer = FinBERTScorer()
        raw = scorer.rescore_df(raw)
        raw.to_csv(GDELT_RAW, index=False)
        print(f"[FinBERT] Saved rescored data → {GDELT_RAW.name}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    if args.aggregate:
        if not GDELT_RAW.exists():
            print("[Error] Run --fetch first")
            return
        raw = pd.read_csv(GDELT_RAW)
        use_fb = "finbert_compound" in raw.columns
        agg = SentimentAggregator()
        weekly = agg.aggregate(raw, use_finbert=use_fb)
        agg.save(weekly)
        print(weekly.describe().round(3))

    # ── Lead-lag ───────────────────────────────────────────────────────────
    if args.leadlag:
        if not SENT_WEEKLY.exists():
            print("[Error] Run --aggregate first")
            return
        sent = pd.read_csv(SENT_WEEKLY, index_col=0, parse_dates=True)
        market = pd.read_csv(PROC_DIR / "unified_weekly.csv",
                             index_col=0, parse_dates=True)
        tester = LeadLagTester()
        ll_df  = tester.run(sent, market)
        ll_df.to_csv(PROC_DIR / "sentiment_leadlag.csv", index=False)
        gate = tester.print_summary(ll_df)
        leading = [f for f, passed in gate.items() if passed]
        print(f"\n  Leading features: {leading}")

    # ── Ablation ───────────────────────────────────────────────────────────
    if args.ablation:
        if not SENT_WEEKLY.exists():
            print("[Error] Run --aggregate first")
            return
        sent   = pd.read_csv(SENT_WEEKLY, index_col=0, parse_dates=True)
        market = pd.read_csv(PROC_DIR / "unified_weekly.csv",
                             index_col=0, parse_dates=True)

        # Determine leading features from saved lead-lag (or use all)
        ll_path = PROC_DIR / "sentiment_leadlag.csv"
        if ll_path.exists():
            ll_df   = pd.read_csv(ll_path)
            tester  = LeadLagTester()
            gate    = tester.print_summary(ll_df)
            leading = [f for f, passed in gate.items() if passed]
        else:
            leading = list(sent.columns)

        if not leading:
            print("  No features passed the lead-lag gate. Ablation skipped.")
            return

        print(f"\n  Using {len(leading)} leading NLP features: {leading}")
        results = run_ablation(sent, market, leading)
        results.to_csv(PROC_DIR / "sentiment_ablation.csv", index=False)
        print("\n  Saved → sentiment_ablation.csv")


if __name__ == "__main__":
    main()
