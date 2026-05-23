"""
gen_nb11_longshort.py -- generate notebooks/eda/11_longshort_backtest.ipynb

Fix NB10's "always-long" problem with two approaches:
  A) Demeaned target: remove drift bias from y_train before fitting
  B) Quantile signal: force 50/50 long/short per fold (market-neutral)

Compare 4 signal versions:
  1. Original (NB10) -- almost always long (bad)
  2. Demeaned target -- unbiased regression
  3. Quantile signal -- force 50/50 split
  4. Demeaned + Quantile -- both fixes combined
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'notebooks' / 'eda' / '11_longshort_backtest.ipynb'

def md(source): return {"cell_type": "markdown", "metadata": {}, "source": source}
def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}

cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────────
cells.append(md(
    "# Notebook 11 -- True Long-Short Backtest\n"
    "\n"
    "> *Fix the 'always-long' bias. Find the real market-neutral alpha.*\n"
    "\n"
    "**Problem from NB10:** Model predicted Long 99.7% of weeks — equivalent to Buy-and-Hold.  \n"
    "**Root cause:** XGBoost learned SET's positive long-run drift, not directional timing.\n"
    "\n"
    "**Two fixes applied:**\n"
    "- **A) Demeaned target** — subtract expanding mean from y_train so model learns *relative* moves\n"
    "- **B) Quantile signal** — force 50/50 Long/Short per fold based on prediction rank\n"
    "\n"
    "| Version | Description |\n"
    "|---|---|\n"
    "| V1 Original | NB10 baseline (sign of raw prediction) |\n"
    "| V2 Demeaned | Train on y minus expanding mean |\n"
    "| V3 Quantile | Signal = rank(pred) > median per fold |\n"
    "| V4 Both | Demeaned training + quantile signal |\n"
))

# ── Cell 1: Setup ──────────────────────────────────────────────────────────────
cells.append(code(
    "from pathlib import Path\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.gridspec as gridspec\n"
    "from scipy.stats import spearmanr, ttest_1samp\n"
    "from sklearn.base import clone\n"
    "from xgboost import XGBRegressor\n"
    "warnings.filterwarnings('ignore')\n"
    "np.random.seed(42)\n"
    "\n"
    "ROOT = Path('../..')\n"
    "PROC = ROOT / 'data' / 'processed'\n"
    "FIGS = Path('figs')\n"
    "FIGS.mkdir(exist_ok=True)\n"
    "\n"
    "plt.rcParams.update({\n"
    "    'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,\n"
    "    'font.size': 10, 'axes.titlesize': 11,\n"
    "})\n"
    "\n"
    "def dir_acc(pred, true):\n"
    "    return float(np.mean(np.sign(pred) == np.sign(true)))\n"
    "\n"
    "def ic(pred, true):\n"
    "    mask = np.isfinite(pred) & np.isfinite(true)\n"
    "    return float(spearmanr(pred[mask], true[mask])[0]) if mask.sum() >= 10 else np.nan\n"
    "\n"
    "def risk_metrics(rets, freq=52):\n"
    "    r = pd.Series(rets)\n"
    "    ann_ret = (1 + r).prod() ** (freq / len(r)) - 1\n"
    "    ann_vol = r.std() * np.sqrt(freq)\n"
    "    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan\n"
    "    neg     = r[r < 0]\n"
    "    sortino = ann_ret / (neg.std() * np.sqrt(freq)) if len(neg) > 0 else np.nan\n"
    "    cum     = (1 + r).cumprod()\n"
    "    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()\n"
    "    return {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,\n"
    "            'sortino': sortino, 'max_dd': max_dd,\n"
    "            'win_rate': (r > 0).mean(), 'n': len(r)}\n"
    "\n"
    "print('Setup complete.')"
))

# ── Cell 2: Load data ──────────────────────────────────────────────────────────
cells.append(md("---\n## Setup: Data & Features\n"))

cells.append(code(
    "clean = pd.read_csv(PROC / 'unified_weekly_clean.csv', index_col=0, parse_dates=True)\n"
    "TARGET = 'SET_index_ret_w_fwd1'\n"
    "df = clean.dropna(subset=[TARGET]).copy()\n"
    "\n"
    "FEATURES = [\n"
    "    'sp500_ret_w', 'SET_index_rvol_4w', 'sp500_rvol_4w',\n"
    "    'us_10yr_treasury_ret_w', 'oil_ret_d_lag1', 'eem_ret_d_lag1',\n"
    "    'gold_rvol_4w', 'us_2yr_treasury_ret_w', 'oil_ret_w',\n"
    "    'SET_index_ret_w', 'USD_THB_ret_w',\n"
    "]\n"
    "FEATURES = [f for f in FEATURES if f in df.columns]\n"
    "\n"
    "MODEL = XGBRegressor(\n"
    "    n_estimators=50, max_depth=2, learning_rate=0.05,\n"
    "    subsample=0.7, colsample_bytree=0.5,\n"
    "    min_child_weight=20, reg_alpha=1.0, reg_lambda=10.0,\n"
    "    random_state=42, verbosity=0, n_jobs=1\n"
    ")\n"
    "MIN_TRAIN, FOLD_SIZE = 156, 52\n"
    "print(f'Features: {len(FEATURES)}, OOS start: {df.index[MIN_TRAIN].date()}')"
))

# ── Cell 3: Walk-forward engine ────────────────────────────────────────────────
cells.append(md("---\n## Fix A: Demeaned Target  |  Fix B: Quantile Signal\n\n"
    "**Fix A — Demeaned target:** Before fitting each fold, subtract the expanding mean of  \n"
    "y_train from y. The model learns *excess return above trend*, not raw return.  \n"
    "At prediction time, we add the training mean back before generating the signal.\n\n"
    "**Fix B — Quantile signal:** Instead of `sign(pred)`, rank all predictions in the fold  \n"
    "and go Long if rank ≥ median, Short if rank < median. Forces exactly 50/50 split.\n"))

cells.append(code(
    "def walk_forward_versions(df_all, features, target_col, model,\n"
    "                          min_train=156, fold_size=52):\n"
    "    \"\"\"Run 4 signal versions in one pass.\"\"\"\n"
    "    data = df_all[features + [target_col]].dropna(subset=[target_col])\n"
    "    n = len(data)\n"
    "    all_records = {v: [] for v in ['v1_orig','v2_demean','v3_quant','v4_both']}\n"
    "    t = min_train\n"
    "\n"
    "    while t + fold_size <= n:\n"
    "        train = data.iloc[:t]\n"
    "        test  = data.iloc[t:t+fold_size]\n"
    "        medians = train[features].median()\n"
    "        Xtr = train[features].fillna(medians).fillna(0.).values.astype(float)\n"
    "        Xte = test[features].fillna(medians).fillna(0.).values.astype(float)\n"
    "        ytr = train[target_col].values\n"
    "        yte = test[target_col].values\n"
    "\n"
    "        # ── V1: Original (raw y, sign signal) ─────────────────────────────\n"
    "        m1 = clone(model); m1.fit(Xtr, ytr)\n"
    "        p1 = m1.predict(Xte)\n"
    "        s1 = np.sign(p1)\n"
    "\n"
    "        # ── V2: Demeaned target, sign signal ──────────────────────────────\n"
    "        y_mean = ytr.mean()          # expanding mean of training y\n"
    "        ytr_dm = ytr - y_mean        # remove drift\n"
    "        m2 = clone(model); m2.fit(Xtr, ytr_dm)\n"
    "        p2 = m2.predict(Xte) + y_mean  # add mean back for direction\n"
    "        s2 = np.sign(m2.predict(Xte))  # sign of demeaned pred (no drift)\n"
    "\n"
    "        # ── V3: Original model, quantile signal ───────────────────────────\n"
    "        median_pred = np.median(p1)\n"
    "        s3 = np.where(p1 >= median_pred, 1.0, -1.0)\n"
    "\n"
    "        # ── V4: Demeaned model, quantile signal ───────────────────────────\n"
    "        p4_dm = m2.predict(Xte)       # demeaned predictions\n"
    "        median_dm = np.median(p4_dm)\n"
    "        s4 = np.where(p4_dm >= median_dm, 1.0, -1.0)\n"
    "\n"
    "        for date, y_true, sv1, sv2, sv3, sv4 in zip(\n"
    "                test.index, yte, s1, s2, s3, s4):\n"
    "            row = {'date': date, 'actual': y_true,\n"
    "                   'pred_raw': p1[list(test.index).index(date)] if date in test.index else np.nan}\n"
    "            all_records['v1_orig'].append({**row, 'signal': sv1})\n"
    "            all_records['v2_demean'].append({**row, 'signal': sv2})\n"
    "            all_records['v3_quant'].append({**row, 'signal': sv3})\n"
    "            all_records['v4_both'].append({**row, 'signal': sv4})\n"
    "        t += fold_size\n"
    "\n"
    "    results = {}\n"
    "    for v, recs in all_records.items():\n"
    "        df_v = pd.DataFrame(recs).set_index('date').sort_index()\n"
    "        df_v['ret'] = df_v['signal'] * df_v['actual']\n"
    "        results[v] = df_v\n"
    "    return results\n"
    "\n"
    "print('Running walk-forward (4 versions)...')\n"
    "results = walk_forward_versions(df, FEATURES, TARGET, MODEL,\n"
    "                                min_train=MIN_TRAIN, fold_size=FOLD_SIZE)\n"
    "\n"
    "print(f'OOS weeks: {len(results[\"v1_orig\"])}')\n"
    "print()\n"
    "print(f'{\"Version\":<22} {\"Long%\":>7} {\"Short%\":>7} {\"DirAcc\":>8}')\n"
    "print('-' * 48)\n"
    "labels = {'v1_orig':'V1 Original','v2_demean':'V2 Demeaned',\n"
    "          'v3_quant':'V3 Quantile','v4_both':'V4 Both'}\n"
    "for v, lbl in labels.items():\n"
    "    pnl = results[v]\n"
    "    long_pct  = (pnl['signal'] ==  1).mean()\n"
    "    short_pct = (pnl['signal'] == -1).mean()\n"
    "    da = dir_acc(pnl['signal'].values, pnl['actual'].values)\n"
    "    print(f'{lbl:<22} {long_pct:>7.1%} {short_pct:>7.1%} {da:>8.3f}')"
))

# ── Cell 5: TC + cumulative PnL ───────────────────────────────────────────────
cells.append(code(
    "# Apply 0.1% TC (flip cost) to all versions\n"
    "TC = 0.001\n"
    "for v in results:\n"
    "    pnl = results[v]\n"
    "    flip = pnl['signal'].diff().abs() > 0\n"
    "    pnl['ret_tc'] = pnl['ret'] - flip * 2 * TC\n"
    "\n"
    "# Buy-and-hold benchmark\n"
    "bnh = results['v1_orig']['actual']\n"
    "\n"
    "# Print summary\n"
    "print(f'{\"Version\":<22} {\"AnnRet\":>8} {\"Sharpe\":>8} {\"MaxDD\":>8} {\"WinRate\":>8}')\n"
    "print('-' * 60)\n"
    "for v, lbl in labels.items():\n"
    "    m = risk_metrics(results[v]['ret_tc'])\n"
    "    print(f'{lbl:<22} {m[\"ann_ret\"]:>8.2%} {m[\"sharpe\"]:>8.2f} '\n"
    "          f'{m[\"max_dd\"]:>8.1%} {m[\"win_rate\"]:>8.1%}')\n"
    "m_bnh = risk_metrics(bnh)\n"
    "print(f'{\"Buy-and-Hold\":<22} {m_bnh[\"ann_ret\"]:>8.2%} {m_bnh[\"sharpe\"]:>8.2f} '\n"
    "      f'{m_bnh[\"max_dd\"]:>8.1%} {m_bnh[\"win_rate\"]:>8.1%}')"
))

# ── Cell 6: Cumulative PnL plot ────────────────────────────────────────────────
cells.append(md("---\n## Cumulative PnL: All 4 Versions vs Buy-and-Hold\n"))

cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
    "fig.suptitle('Cumulative PnL: 4 Signal Versions (0.1% TC) vs Buy-and-Hold', fontsize=13)\n"
    "\n"
    "colors = {'v1_orig':'#78909C','v2_demean':'#1976D2','v3_quant':'#FF5722','v4_both':'#4CAF50'}\n"
    "axes_flat = axes.flatten()\n"
    "\n"
    "for ax, (v, lbl) in zip(axes_flat, labels.items()):\n"
    "    pnl = results[v]\n"
    "    cum_sig = (1 + pnl['ret_tc']).cumprod()\n"
    "    cum_bnh = (1 + bnh.reindex(pnl.index)).cumprod()\n"
    "    m = risk_metrics(pnl['ret_tc'])\n"
    "    long_pct = (pnl['signal'] == 1).mean()\n"
    "\n"
    "    ax.plot(pnl.index, cum_sig, color=colors[v], lw=2, label=f'{lbl} (0.1%TC)')\n"
    "    ax.plot(pnl.index, cum_bnh, color='#E53935', lw=1.2, ls='--',\n"
    "            alpha=0.7, label='Buy-and-Hold')\n"
    "    ax.axhline(1.0, color='black', lw=0.5, ls=':')\n"
    "    ax.set_yscale('log')\n"
    "    ax.set_title(f'{lbl}\\n'\n"
    "                 f'Sharpe={m[\"sharpe\"]:+.2f} | AnnRet={m[\"ann_ret\"]:.1%} | '\n"
    "                 f'Long={long_pct:.0%}')\n"
    "    ax.legend(fontsize=8)\n"
    "    ax.set_ylabel('Cumulative Return (log)')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'longshort_pnl.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/longshort_pnl.png')"
))

# ── Cell 7: Focus on best version ─────────────────────────────────────────────
cells.append(md("---\n## Deep Dive: Best Version\n\n"
    "Focus on the version with highest Sharpe. Drawdown, regime breakdown, annual returns.\n"))

cells.append(code(
    "# Pick best version by Sharpe\n"
    "sharpes = {v: risk_metrics(results[v]['ret_tc'])['sharpe'] for v in results}\n"
    "best_v  = max(sharpes, key=sharpes.get)\n"
    "best_lbl = labels[best_v]\n"
    "print(f'Best version: {best_lbl} (Sharpe={sharpes[best_v]:.3f})')\n"
    "print()\n"
    "\n"
    "pnl_best = results[best_v]\n"
    "m_best   = risk_metrics(pnl_best['ret_tc'])\n"
    "\n"
    "# TC sensitivity for best version\n"
    "tc_levels = [0.0, 0.001, 0.002, 0.003]\n"
    "print(f'{\"TC\":>8} {\"AnnRet\":>8} {\"Sharpe\":>8} {\"MaxDD\":>8}')\n"
    "print('-' * 38)\n"
    "flip = pnl_best['signal'].diff().abs() > 0\n"
    "for tc in tc_levels:\n"
    "    ret_tc = pnl_best['ret'] - flip * 2 * tc\n"
    "    m = risk_metrics(ret_tc)\n"
    "    print(f'{tc:.1%}  {m[\"ann_ret\"]:>8.2%} {m[\"sharpe\"]:>8.2f} {m[\"max_dd\"]:>8.1%}')\n"
    "print(f'\\nPosition flips: {flip.sum()} total, {flip.mean()*52:.1f}/year')"
))

# ── Cell 8: Annual breakdown ───────────────────────────────────────────────────
cells.append(code(
    "pnl_best['year'] = pnl_best.index.year\n"
    "annual = pnl_best.groupby('year').apply(\n"
    "    lambda g: pd.Series({\n"
    "        'signal': (1+g['ret_tc']).prod()-1,\n"
    "        'bnh':    (1+g['actual']).prod()-1,\n"
    "        'long_pct': (g['signal']==1).mean(),\n"
    "    })\n"
    ").reset_index()\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)\n"
    "fig.suptitle(f'Annual Performance: {best_lbl} (0.1% TC) vs Buy-and-Hold', fontsize=12)\n"
    "\n"
    "x = range(len(annual))\n"
    "w = 0.35\n"
    "ax1.bar([i-w/2 for i in x], annual['signal']*100, width=w,\n"
    "        color=[colors[best_v] if v>0 else '#EF5350' for v in annual['signal']],\n"
    "        label=f'{best_lbl} (0.1%TC)')\n"
    "ax1.bar([i+w/2 for i in x], annual['bnh']*100, width=w,\n"
    "        color='#78909C', alpha=0.5, label='Buy-and-Hold')\n"
    "ax1.axhline(0, color='black', lw=0.8)\n"
    "ax1.set_ylabel('Annual Return (%)')\n"
    "ax1.legend(fontsize=9)\n"
    "\n"
    "ax2.bar(x, annual['long_pct']*100, color=colors[best_v], alpha=0.7)\n"
    "ax2.axhline(50, color='black', lw=0.8, ls='--', label='50% (neutral)')\n"
    "ax2.set_ylabel('Long Fraction (%)')\n"
    "ax2.set_xlabel('Year')\n"
    "ax2.legend(fontsize=9)\n"
    "ax2.set_ylim(0, 100)\n"
    "ax2.set_xticks(x)\n"
    "ax2.set_xticklabels(annual['year'], rotation=45)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'longshort_annual.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "pos_yrs = (annual['signal'] > 0).sum()\n"
    "print(f'{\"Year\":>6} {best_lbl[:14]:>14} {\"Buy-Hold\":>10} {\"Long%\":>7}')\n"
    "print('-' * 42)\n"
    "for _, r in annual.iterrows():\n"
    "    flag = '✓' if r['signal'] > 0 else '✗'\n"
    "    print(f'{int(r[\"year\"]):>6} {r[\"signal\"]:>13.1%} {flag}  {r[\"bnh\"]:>9.1%}  {r[\"long_pct\"]:>6.0%}')\n"
    "print(f'\\nPositive years: {pos_yrs}/{len(annual)}')"
))

# ── Cell 9: Regime breakdown ───────────────────────────────────────────────────
cells.append(code(
    "regime_col = clean['regime'].reindex(pnl_best.index)\n"
    "pnl_best['regime'] = regime_col\n"
    "regime_labels = {-1: 'Bull (low VIX)', 0: 'Normal', 1: 'Crisis (high VIX)'}\n"
    "\n"
    "print(f'Regime breakdown ({best_lbl}, 0.1%TC):')\n"
    "print(f'{\"Regime\":<22} {\"N\":>5} {\"AnnRet\":>8} {\"Sharpe\":>8} {\"Long%\":>7}')\n"
    "print('-' * 55)\n"
    "for r, lbl in sorted(regime_labels.items()):\n"
    "    sub = pnl_best[pnl_best['regime'] == r]\n"
    "    if len(sub) < 10:\n"
    "        print(f'{lbl:<22} {len(sub):>5} -- insufficient')\n"
    "        continue\n"
    "    m = risk_metrics(sub['ret_tc'])\n"
    "    long_pct = (sub['signal'] == 1).mean()\n"
    "    print(f'{lbl:<22} {len(sub):>5} {m[\"ann_ret\"]:>8.1%} {m[\"sharpe\"]:>8.2f} {long_pct:>7.0%}')"
))

# ── Cell 10: IC analysis ───────────────────────────────────────────────────────
cells.append(code(
    "# IC analysis: does prediction rank correlate with actual return?\n"
    "print('IC Analysis (Spearman rank correlation, all OOS weeks):')\n"
    "print()\n"
    "for v, lbl in labels.items():\n"
    "    pnl_v = results[v]\n"
    "    # Use pred_raw for V1/V3 (raw model output), recompute for V2/V4 via signal proxy\n"
    "    ic_val = ic(pnl_v['signal'].values, pnl_v['actual'].values)\n"
    "    da_val = dir_acc(pnl_v['signal'].values, pnl_v['actual'].values)\n"
    "    long_pct = (pnl_v['signal'] == 1).mean()\n"
    "    print(f'{lbl:<22}  IC={ic_val:+.4f}  DirAcc={da_val:.3f}  Long={long_pct:.0%}')\n"
    "\n"
    "print()\n"
    "print('Note: V3/V4 signal = ±1 binary (quantile), so DirAcc = win rate')"
))

# ── Cell 11: Rolling Sharpe ────────────────────────────────────────────────────
cells.append(code(
    "# Rolling 3-year Sharpe for best version vs V1\n"
    "ROLL = 156\n"
    "fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)\n"
    "fig.suptitle('Rolling 3-Year Sharpe: Best Version vs Original', fontsize=12)\n"
    "\n"
    "for ax, (v, lbl, col) in zip(axes, [\n"
    "    ('v1_orig', 'V1 Original (always-long bias)', '#78909C'),\n"
    "    (best_v,    f'{best_lbl} (fixed)',            colors[best_v]),\n"
    "]):\n"
    "    r = results[v]['ret_tc']\n"
    "    roll_sh = r.rolling(ROLL).mean() / r.rolling(ROLL).std() * np.sqrt(52)\n"
    "    ax.plot(r.index, roll_sh, color=col, lw=1.5, label=lbl)\n"
    "    ax.axhline(0, color='black', lw=0.8, ls='--')\n"
    "    ax.axhline(0.5, color='green', lw=0.8, ls=':', alpha=0.7)\n"
    "    ax.fill_between(r.index, roll_sh, 0,\n"
    "                    where=roll_sh > 0, alpha=0.12, color='green')\n"
    "    ax.fill_between(r.index, roll_sh, 0,\n"
    "                    where=roll_sh < 0, alpha=0.12, color='red')\n"
    "    ax.set_ylabel('Rolling Sharpe (3yr)')\n"
    "    ax.legend(fontsize=9)\n"
    "    ax.set_ylim(-2, 3)\n"
    "\n"
    "axes[1].set_xlabel('Date')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'longshort_rolling.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/longshort_rolling.png')"
))

# ── Cell 12: Final verdict ─────────────────────────────────────────────────────
cells.append(md("---\n## Final Verdict\n"))

cells.append(code(
    "print('=' * 65)\n"
    "print('LONG-SHORT BACKTEST -- FINAL VERDICT')\n"
    "print('=' * 65)\n"
    "print()\n"
    "\n"
    "# Summary table\n"
    "print(f'{\"Version\":<22} {\"AnnRet\":>8} {\"Sharpe\":>8} {\"MaxDD\":>8} {\"Long%\":>7}')\n"
    "print('-' * 55)\n"
    "for v, lbl in labels.items():\n"
    "    m = risk_metrics(results[v]['ret_tc'])\n"
    "    lp = (results[v]['signal']==1).mean()\n"
    "    mark = ' <-- BEST' if v == best_v else ''\n"
    "    print(f'{lbl:<22} {m[\"ann_ret\"]:>8.2%} {m[\"sharpe\"]:>8.2f} '\n"
    "          f'{m[\"max_dd\"]:>8.1%} {lp:>7.0%}{mark}')\n"
    "m_b = risk_metrics(bnh)\n"
    "print(f'{\"Buy-and-Hold\":<22} {m_b[\"ann_ret\"]:>8.2%} {m_b[\"sharpe\"]:>8.2f} '\n"
    "      f'{m_b[\"max_dd\"]:>8.1%} {\"100%\":>7}')\n"
    "print()\n"
    "\n"
    "m_best_full = risk_metrics(pnl_best['ret_tc'])\n"
    "excess_sharpe = m_best_full['sharpe'] - m_b['sharpe']\n"
    "excess_ret    = m_best_full['ann_ret'] - m_b['ann_ret']\n"
    "\n"
    "print(f'Best version: {best_lbl}')\n"
    "print(f'  Sharpe vs Buy-and-Hold: {m_best_full[\"sharpe\"]:+.2f} vs {m_b[\"sharpe\"]:+.2f} '\n"
    "      f'(excess = {excess_sharpe:+.2f})')\n"
    "print(f'  Annual return excess:   {excess_ret:+.2%}')\n"
    "print()\n"
    "\n"
    "if m_best_full['sharpe'] > 0.5 and excess_sharpe > 0.1:\n"
    "    verdict = 'TRUE ALPHA EXISTS -- long-short edge is real'\n"
    "    detail  = ('After removing drift bias and forcing market-neutral positioning, '\n"
    "               'the signal generates meaningful risk-adjusted return. '\n"
    "               'Sharpe > 0.5 and excess Sharpe > 0.1 vs Buy-and-Hold.')\n"
    "elif m_best_full['sharpe'] > 0.3:\n"
    "    verdict = 'WEAK ALPHA -- marginal edge after fixing bias'\n"
    "    detail  = ('Signal has a positive but small edge. Sharpe is low. '\n"
    "               'May be worth paper-trading but not deploying capital yet. '\n"
    "               'Need more NLP data and longer OOS window.')\n"
    "else:\n"
    "    verdict = 'NO ALPHA -- signal does not beat passive after fixing bias'\n"
    "    detail  = ('After removing the always-long bias, the signal has no real edge. '\n"
    "               'The original DirAcc of ~54% was mostly from market drift. '\n"
    "               'This dataset does not contain a tradeable timing signal.')\n"
    "\n"
    "print('=' * 65)\n"
    "print(f'VERDICT: {verdict}')\n"
    "print('=' * 65)\n"
    "print()\n"
    "print(detail)"
))

# ── Write ──────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Generated: {OUT}")
print(f"Cells: {len(cells)}")
