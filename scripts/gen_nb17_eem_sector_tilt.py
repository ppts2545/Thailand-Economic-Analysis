"""
gen_nb17_eem_sector_tilt.py -- generate notebooks/eda/17_eem_sector_tilt.ipynb

Combines the two main findings of this project:
  1. EEM lag signal (NB16): strong market-level predictor (IC=0.11, Sharpe +0.58)
  2. Sector momentum (NB13/15): weak cross-sectional signal, works long-only monthly

Strategy logic:
  - EEM signal determines MARKET position (long / flat / short)
  - When long: tilt toward top-momentum sectors instead of equal-weight
  - When flat/short: hold cash or equal-weight short

Three variants:
  A. EEM L/flat + sector tilt (long top-2 when EEM+, flat otherwise)
  B. EEM L/S + sector tilt (long top-2 when EEM+, short bottom-2 when EEM-)
  C. EEM L/flat + full sector rotation (long top-3, short bottom-3, conditional on EEM+)

Benchmark: EEM L/flat equal-weight (NB16 best strategy)
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'notebooks' / 'eda' / '17_eem_sector_tilt.ipynb'

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": s}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Notebook 17 — EEM Signal + Sector Tilt\n\n"
    "> *Can combining the EEM market signal with sector momentum improve on either alone?*\n\n"
    "**Hypothesis:** EEM tells us *when* to be long the market. Sector momentum tells us  \n"
    "*which sectors* to hold. Combining both should improve risk-adjusted returns.\n\n"
    "**Architecture:**\n"
    "```\n"
    "EEM lag signal  →  market on/off switch  →  position sizing\n"
    "Sector momentum →  sector ranking        →  position tilt\n"
    "```\n\n"
    "| Section | Description |\n"
    "|---|---|\n"
    "| 1. Setup | Data, signals, shared utilities |\n"
    "| 2. Signals | EEM signal + sector momentum features |\n"
    "| 3. Variant A | EEM L/flat + sector tilt (recommended) |\n"
    "| 4. Variant B | EEM L/S + sector tilt |\n"
    "| 5. Variant C | EEM-gated sector L/S |\n"
    "| 6. Comparison | All variants vs benchmarks |\n"
    "| 7. Verdict | Does combination add value? |"
))

# ── Section 1 — Setup ─────────────────────────────────────────────────────────
cells.append(md("---\n## Section 1 — Setup & Data"))

cells.append(code(
    "from pathlib import Path\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from scipy.stats import spearmanr, ttest_1samp\n"
    "from xgboost import XGBRegressor\n"
    "warnings.filterwarnings('ignore')\n"
    "np.random.seed(42)\n\n"
    "ROOT = Path('../..')\n"
    "PROC = ROOT / 'data' / 'processed'\n"
    "FIGS = Path('figs')\n"
    "FIGS.mkdir(exist_ok=True)\n\n"
    "plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})\n\n"
    "TC      = 0.001   # 0.1% one-way\n"
    "SECTORS = ['BANK','ENERGY','ICT','COMMERCE','HEALTH','PROPERTY','FOOD']\n"
    "LONG_K  = 2\n"
    "SHORT_K = 2\n\n"
    "# ── Macro / market data ──\n"
    "macro = pd.read_csv(PROC / 'unified_weekly_clean.csv', index_col=0, parse_dates=True)\n"
    "macro.index = macro.index.normalize()\n\n"
    "# ── Sector returns ──\n"
    "sector = pd.read_csv(PROC / 'sector_weekly.csv', index_col=0, parse_dates=True)\n"
    "sector = sector[SECTORS]\n"
    "sector.index = sector.index.normalize()\n\n"
    "# Align\n"
    "common = macro.index.intersection(sector.index)\n"
    "macro  = macro.loc[common]\n"
    "sector = sector.loc[common]\n\n"
    "# Targets\n"
    "SET_FWD = 'SET_index_ret_w_fwd1'\n"
    "if SET_FWD not in macro.columns:\n"
    "    macro[SET_FWD] = macro['SET_index_ret_w'].shift(-1)\n\n"
    "print(f'Data: {len(common)} weeks  ({common[0].date()} → {common[-1].date()})')\n"
    "print(f'Macro features: {macro.shape[1]}')\n"
    "print(f'Sector returns: {sector.shape[1]} sectors')"
))

# ── Section 2 — Signals ───────────────────────────────────────────────────────
cells.append(md(
    "---\n## Section 2 — Signal Construction\n\n"
    "**EEM signal** (from NB16): `sign(eem_ret_d_lag1)` → market on/off  \n"
    "**Sector momentum** (from NB13): 4-week lagged return, cross-sectionally ranked"
))

cells.append(code(
    "# ── EEM signal ──\n"
    "eem_sig  = macro['eem_ret_d_lag1'].copy()   # raw signal\n"
    "eem_pos  = (eem_sig > 0).astype(int)        # 1=long, 0=flat\n"
    "eem_sign = np.sign(eem_sig)                  # +1/-1 for L/S\n\n"
    "# ── Sector momentum features ──\n"
    "# mom4 = 4-week lagged return (rank within each week)\n"
    "mom4 = sector.shift(1).rolling(4).mean()\n"
    "mom1 = sector.shift(1)\n"
    "mom12= sector.shift(1).rolling(12).mean()\n\n"
    "# Cross-sectional rank (0=worst, 1=best) each week\n"
    "mom4_rank  = mom4.rank(axis=1, pct=True)\n"
    "mom1_rank  = mom1.rank(axis=1, pct=True)\n"
    "mom12_rank = mom12.rank(axis=1, pct=True)\n\n"
    "# Composite momentum score (simple average of ranks)\n"
    "mom_score = (mom4_rank + mom1_rank + mom12_rank) / 3\n\n"
    "# Forward returns for sectors\n"
    "sector_fwd = sector.shift(-1)   # next week's actual return\n\n"
    "print('Signal summary:')\n"
    "print(f'  EEM+ weeks: {eem_pos.sum()} / {len(eem_pos)} ({eem_pos.mean():.1%})')\n"
    "print(f'  EEM- weeks: {(eem_pos==0).sum()} / {len(eem_pos)} ({(eem_pos==0).mean():.1%})')\n\n"
    "# Validate sector momentum IC\n"
    "# Flatten: for each week, correlate mom_score ranks with next week's sector returns\n"
    "ics = []\n"
    "for date in mom_score.dropna().index:\n"
    "    if date not in sector_fwd.index: continue\n"
    "    scores = mom_score.loc[date].dropna()\n"
    "    fwds   = sector_fwd.loc[date, scores.index].dropna()\n"
    "    if len(fwds) < 4: continue\n"
    "    common_s = scores.index.intersection(fwds.index)\n"
    "    ic, _ = spearmanr(scores[common_s], fwds[common_s])\n"
    "    ics.append(ic)\n"
    "ic_mom = pd.Series(ics)\n"
    "print(f'\\nSector momentum IC: mean={ic_mom.mean():+.4f}  std={ic_mom.std():.4f}  IC+={( ic_mom>0).mean():.1%}')"
))

# ── Section 3 — Variant A ─────────────────────────────────────────────────────
cells.append(md(
    "---\n## Section 3 — Variant A: EEM L/flat + Sector Tilt\n\n"
    "**Rule:**\n"
    "- EEM+ week → long top-2 momentum sectors (equal weight)\n"
    "- EEM− week → flat (cash)\n\n"
    "TC is paid only when positions change."
))

cells.append(code(
    "def risk_metrics(rets, freq=52):\n"
    "    r = pd.Series(rets).dropna()\n"
    "    if len(r) < 10: return {}\n"
    "    ann_ret = (1+r).prod()**(freq/len(r)) - 1\n"
    "    ann_vol = r.std() * np.sqrt(freq)\n"
    "    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan\n"
    "    cum = (1+r).cumprod()\n"
    "    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()\n"
    "    return {'ann_ret':ann_ret,'ann_vol':ann_vol,'sharpe':sharpe,\n"
    "            'max_dd':max_dd,'win_rate':(r>0).mean(),'n':len(r)}\n\n"
    "def backtest_varA(eem_pos, mom_score, sector_fwd, long_k=2, tc=TC):\n"
    "    \"\"\"EEM L/flat + sector tilt: long top-k when EEM+, flat when EEM-.\"\"\"\n"
    "    dates = mom_score.dropna().index.intersection(eem_pos.index).intersection(sector_fwd.index)\n"
    "    records = []\n"
    "    prev_longs = set()\n\n"
    "    for date in sorted(dates):\n"
    "        eem_on = eem_pos.loc[date] == 1\n"
    "        scores = mom_score.loc[date].dropna()\n"
    "        fwds   = sector_fwd.loc[date].dropna()\n"
    "        valid  = scores.index.intersection(fwds.index)\n"
    "        scores = scores[valid]; fwds = fwds[valid]\n"
    "        if len(valid) < long_k:\n"
    "            continue\n\n"
    "        if eem_on:\n"
    "            top_k = set(scores.nlargest(long_k).index)\n"
    "            changed = len(top_k - prev_longs) + (1 if not prev_longs and eem_on else 0)\n"
    "            # simplified: pay TC on sectors that changed\n"
    "            new_entries = top_k - prev_longs\n"
    "            old_exits   = prev_longs - top_k\n"
    "            tc_cost = (len(new_entries) + len(old_exits)) * tc\n"
    "            # if coming from flat, pay entry TC\n"
    "            if not prev_longs:\n"
    "                tc_cost = long_k * tc\n"
    "            port_ret = fwds[list(top_k)].mean()\n"
    "            prev_longs = top_k\n"
    "        else:\n"
    "            # going to flat\n"
    "            tc_cost = len(prev_longs) * tc if prev_longs else 0\n"
    "            port_ret = 0.0\n"
    "            prev_longs = set()\n\n"
    "        bench_ret = fwds.mean()   # equal-weight all sectors\n"
    "        records.append({'date':date,'gross_ret':port_ret,\n"
    "                        'net_ret':port_ret - tc_cost,\n"
    "                        'bench_ret':bench_ret,\n"
    "                        'eem_on':eem_on,'tc_cost':tc_cost})\n\n"
    "    return pd.DataFrame(records).set_index('date')\n\n"
    "res_A = backtest_varA(eem_pos, mom_score, sector_fwd)\n"
    "m_A   = risk_metrics(res_A['net_ret'])\n"
    "print(f'Variant A (EEM L/flat + sector tilt):')\n"
    "print(f'  Gross Sharpe: {risk_metrics(res_A[\"gross_ret\"])[\"sharpe\"]:+.3f}')\n"
    "print(f'  Net Sharpe:   {m_A[\"sharpe\"]:+.3f}')\n"
    "print(f'  Ann Return:   {m_A[\"ann_ret\"]:+.2%}')\n"
    "print(f'  Max DD:       {m_A[\"max_dd\"]:+.1%}')\n"
    "print(f'  Avg TC/yr:    {res_A[\"tc_cost\"].mean()*52:.2%}')"
))

# ── Section 4 — Variant B ─────────────────────────────────────────────────────
cells.append(md(
    "---\n## Section 4 — Variant B: EEM L/S + Sector Tilt\n\n"
    "**Rule:**\n"
    "- EEM+ week → long top-2 momentum sectors\n"
    "- EEM− week → short bottom-2 momentum sectors\n\n"
    "Market-neutral on EEM− weeks."
))

cells.append(code(
    "def backtest_varB(eem_sign, mom_score, sector_fwd, long_k=2, short_k=2, tc=TC):\n"
    "    \"\"\"EEM L/S + sector tilt.\"\"\"\n"
    "    dates = mom_score.dropna().index.intersection(eem_sign.index).intersection(sector_fwd.index)\n"
    "    records = []\n"
    "    prev_longs  = set()\n"
    "    prev_shorts = set()\n\n"
    "    for date in sorted(dates):\n"
    "        direction = eem_sign.loc[date]   # +1 or -1\n"
    "        scores = mom_score.loc[date].dropna()\n"
    "        fwds   = sector_fwd.loc[date].dropna()\n"
    "        valid  = scores.index.intersection(fwds.index)\n"
    "        scores = scores[valid]; fwds = fwds[valid]\n"
    "        if len(valid) < max(long_k, short_k):\n"
    "            continue\n\n"
    "        if direction >= 0:   # long mode\n"
    "            new_longs  = set(scores.nlargest(long_k).index)\n"
    "            new_shorts = set()\n"
    "        else:                # short mode\n"
    "            new_longs  = set()\n"
    "            new_shorts = set(scores.nsmallest(short_k).index)\n\n"
    "        # TC on position changes\n"
    "        long_change  = len((new_longs  - prev_longs)  | (prev_longs  - new_longs))\n"
    "        short_change = len((new_shorts - prev_shorts) | (prev_shorts - new_shorts))\n"
    "        tc_cost = (long_change + short_change) * tc\n\n"
    "        long_ret  = fwds[list(new_longs)].mean()  if new_longs  else 0.0\n"
    "        short_ret = fwds[list(new_shorts)].mean() if new_shorts else 0.0\n\n"
    "        if direction >= 0:\n"
    "            port_ret = long_ret\n"
    "        else:\n"
    "            port_ret = -short_ret   # short = negative of sector return\n\n"
    "        prev_longs  = new_longs\n"
    "        prev_shorts = new_shorts\n\n"
    "        records.append({'date':date,'gross_ret':port_ret,\n"
    "                        'net_ret':port_ret - tc_cost,\n"
    "                        'bench_ret':fwds.mean(),'tc_cost':tc_cost,\n"
    "                        'direction':direction})\n\n"
    "    return pd.DataFrame(records).set_index('date')\n\n"
    "res_B = backtest_varB(eem_sign, mom_score, sector_fwd)\n"
    "m_B   = risk_metrics(res_B['net_ret'])\n"
    "print(f'Variant B (EEM L/S + sector tilt):')\n"
    "print(f'  Net Sharpe:  {m_B[\"sharpe\"]:+.3f}')\n"
    "print(f'  Ann Return:  {m_B[\"ann_ret\"]:+.2%}')\n"
    "print(f'  Max DD:      {m_B[\"max_dd\"]:+.1%}')\n"
    "print(f'  Avg TC/yr:   {res_B[\"tc_cost\"].mean()*52:.2%}')"
))

# ── Section 5 — Variant C ─────────────────────────────────────────────────────
cells.append(md(
    "---\n## Section 5 — Variant C: EEM-Gated Sector L/S\n\n"
    "**Rule:**\n"
    "- EEM+ week → long top-2, short bottom-2 (full sector L/S)\n"
    "- EEM− week → flat\n\n"
    "Only runs the L/S sector strategy when global sentiment is positive.  \n"
    "Hypothesis: sector rotation signal is stronger in bull/normal regimes."
))

cells.append(code(
    "def backtest_varC(eem_pos, mom_score, sector_fwd, long_k=2, short_k=2, tc=TC):\n"
    "    \"\"\"EEM-gated sector L/S: only run L/S when EEM+, otherwise flat.\"\"\"\n"
    "    dates = mom_score.dropna().index.intersection(eem_pos.index).intersection(sector_fwd.index)\n"
    "    records = []\n"
    "    prev_longs  = set()\n"
    "    prev_shorts = set()\n\n"
    "    for date in sorted(dates):\n"
    "        eem_on = eem_pos.loc[date] == 1\n"
    "        scores = mom_score.loc[date].dropna()\n"
    "        fwds   = sector_fwd.loc[date].dropna()\n"
    "        valid  = scores.index.intersection(fwds.index)\n"
    "        scores = scores[valid]; fwds = fwds[valid]\n"
    "        if len(valid) < (long_k + short_k):\n"
    "            continue\n\n"
    "        if eem_on:\n"
    "            new_longs  = set(scores.nlargest(long_k).index)\n"
    "            new_shorts = set(scores.nsmallest(short_k).index)\n"
    "            long_change  = len((new_longs  - prev_longs)  | (prev_longs  - new_longs))\n"
    "            short_change = len((new_shorts - prev_shorts) | (prev_shorts - new_shorts))\n"
    "            if not prev_longs:  # entering from flat\n"
    "                tc_cost = (long_k + short_k) * tc\n"
    "            else:\n"
    "                tc_cost = (long_change + short_change) * tc\n"
    "            long_ret  = fwds[list(new_longs)].mean()\n"
    "            short_ret = fwds[list(new_shorts)].mean()\n"
    "            port_ret  = (long_ret - short_ret) / 2\n"
    "            prev_longs  = new_longs\n"
    "            prev_shorts = new_shorts\n"
    "        else:\n"
    "            tc_cost = (len(prev_longs) + len(prev_shorts)) * tc if prev_longs else 0\n"
    "            port_ret = 0.0\n"
    "            prev_longs  = set()\n"
    "            prev_shorts = set()\n\n"
    "        records.append({'date':date,'gross_ret':port_ret,\n"
    "                        'net_ret':port_ret - tc_cost,\n"
    "                        'bench_ret':fwds.mean(),'tc_cost':tc_cost,\n"
    "                        'eem_on':eem_on})\n\n"
    "    return pd.DataFrame(records).set_index('date')\n\n"
    "res_C = backtest_varC(eem_pos, mom_score, sector_fwd)\n"
    "m_C   = risk_metrics(res_C['net_ret'])\n"
    "print(f'Variant C (EEM-gated sector L/S):')\n"
    "print(f'  Net Sharpe:  {m_C[\"sharpe\"]:+.3f}')\n"
    "print(f'  Ann Return:  {m_C[\"ann_ret\"]:+.2%}')\n"
    "print(f'  Max DD:      {m_C[\"max_dd\"]:+.1%}')\n"
    "print(f'  Avg TC/yr:   {res_C[\"tc_cost\"].mean()*52:.2%}')"
))

# ── Section 6 — Comparison ────────────────────────────────────────────────────
cells.append(md("---\n## Section 6 — Full Comparison"))

cells.append(code(
    "# ── Build benchmark: EEM L/flat equal-weight (NB16 best) ──\n"
    "eem_sig_raw  = macro['eem_ret_d_lag1']\n"
    "target_set   = macro[SET_FWD]\n"
    "pos_lf       = eem_pos.copy().reindex(target_set.index).fillna(0)\n"
    "pos_change   = pos_lf.diff().abs().fillna(pos_lf.abs())\n"
    "gross_eem_lf = pos_lf * target_set\n"
    "net_eem_lf   = gross_eem_lf - pos_change * TC\n\n"
    "bnh          = target_set   # SET buy & hold\n\n"
    "all_strats = [\n"
    "    ('EEM L/flat EW (NB16)',      net_eem_lf,      gross_eem_lf),\n"
    "    ('A: EEM L/flat + tilt',      res_A['net_ret'], res_A['gross_ret']),\n"
    "    ('B: EEM L/S + tilt',         res_B['net_ret'], res_B['gross_ret']),\n"
    "    ('C: EEM-gated sector L/S',   res_C['net_ret'], res_C['gross_ret']),\n"
    "    ('SET Buy & Hold',            bnh,              bnh),\n"
    "]\n\n"
    "print(f'{\"Strategy\":<30} {\"Gross Sharpe\":>13} {\"Net Sharpe\":>11} {\"AnnRet\":>8} {\"MaxDD\":>7} {\"WinRate\":>8}')\n"
    "print('-' * 82)\n"
    "for name, net, gross in all_strats:\n"
    "    mn = risk_metrics(net)\n"
    "    mg = risk_metrics(gross)\n"
    "    if not mn: continue\n"
    "    flag = ' ✓' if mn['sharpe'] > 0.5 else ''\n"
    "    print(f'{name:<30} {mg[\"sharpe\"]:>13.3f} {mn[\"sharpe\"]:>11.3f} '\n"
    "          f'{mn[\"ann_ret\"]:>8.2%} {mn[\"max_dd\"]:>7.1%} {mn[\"win_rate\"]:>8.1%}{flag}')"
))

cells.append(code(
    "# ── Equity curves ──\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)\n"
    "fig.suptitle('EEM Signal + Sector Tilt: All Variants vs Benchmarks', fontsize=12)\n\n"
    "plot_config = [\n"
    "    (net_eem_lf,      'EEM L/flat EW (NB16)',     '#4CAF50', '-',  1.5),\n"
    "    (res_A['net_ret'],'A: EEM L/flat + tilt',     '#1976D2', '-',  2.0),\n"
    "    (res_B['net_ret'],'B: EEM L/S + tilt',        '#7B1FA2', '-',  2.0),\n"
    "    (res_C['net_ret'],'C: EEM-gated sector L/S',  '#FF6F00', '-',  2.0),\n"
    "    (bnh,             'SET Buy & Hold',            '#9E9E9E', ':',  1.5),\n"
    "]\n\n"
    "ax = axes[0]\n"
    "for rets, lbl, col, ls, lw in plot_config:\n"
    "    m   = risk_metrics(rets)\n"
    "    cum = (1 + rets.dropna()).cumprod()\n"
    "    ax.plot(cum, lw=lw, ls=ls, color=col,\n"
    "            label=f'{lbl}  Sharpe={m[\"sharpe\"]:+.2f}  Ret={m[\"ann_ret\"]:+.1%}')\n"
    "ax.axhline(1.0, color='gray', lw=0.5, ls=':')\n"
    "ax.set_yscale('log')\n"
    "ax.set_ylabel('Cumulative Return (log)')\n"
    "ax.legend(fontsize=8, loc='upper left')\n"
    "ax.set_title('Equity Curves: EEM + Sector Tilt Variants')\n\n"
    "# Annual returns for best variant\n"
    "best_net = max(\n"
    "    [res_A['net_ret'], res_B['net_ret'], res_C['net_ret'], net_eem_lf],\n"
    "    key=lambda r: risk_metrics(r).get('sharpe', -999)\n"
    ")\n"
    "annual_best = best_net.groupby(best_net.index.year).apply(lambda r: (1+r).prod()-1)\n"
    "annual_bnh  = bnh.groupby(bnh.index.year).apply(lambda r: (1+r).prod()-1)\n"
    "ax2 = axes[1]\n"
    "x = range(len(annual_best))\n"
    "w = 0.35\n"
    "ax2.bar([i-w/2 for i in x], annual_best*100, w,\n"
    "        color=['#1976D2' if v>0 else '#EF5350' for v in annual_best],\n"
    "        label='Best variant (net TC)')\n"
    "ax2.bar([i+w/2 for i in x], annual_bnh*100, w,\n"
    "        color='#78909C', alpha=0.5, label='SET B&H')\n"
    "ax2.axhline(0, color='black', lw=0.8)\n"
    "ax2.set_xticks(x); ax2.set_xticklabels(annual_best.index, rotation=45)\n"
    "ax2.set_ylabel('Annual Return (%)')\n"
    "ax2.set_title('Annual Returns: Best Variant vs SET B&H')\n"
    "ax2.legend(fontsize=9)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'eem_sector_equity.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/eem_sector_equity.png')"
))

# ── Section 7 — Verdict ───────────────────────────────────────────────────────
cells.append(md("---\n## Section 7 — Verdict: Does the Combination Add Value?"))

cells.append(code(
    "from scipy.stats import ttest_1samp\n\n"
    "def score_strat(net_ret, gross_ret, label):\n"
    "    m  = risk_metrics(net_ret)\n"
    "    mg = risk_metrics(gross_ret)\n"
    "    if not m: return 0\n"
    "    annual = net_ret.groupby(net_ret.index.year).apply(lambda r: (1+r).prod()-1)\n"
    "    _, p_ret = ttest_1samp(net_ret.dropna(), 0)\n"
    "    ev = [\n"
    "        ('Gross Sharpe > 0.5',          mg['sharpe'],             mg['sharpe'] > 0.5),\n"
    "        ('Net Sharpe > 0.5',            m['sharpe'],              m['sharpe'] > 0.5),\n"
    "        ('Net Sharpe > EEM L/flat(0.58)',m['sharpe'],              m['sharpe'] > 0.58),\n"
    "        ('Annual return > 7%',          m['ann_ret'],             m['ann_ret'] > 0.07),\n"
    "        ('Max DD > -40%',               m['max_dd'],              m['max_dd'] > -0.40),\n"
    "        ('Positive years > 60%',        (annual>0).mean(),        (annual>0).mean() > 0.60),\n"
    "        ('Win rate > 52%',              m['win_rate'],            m['win_rate'] > 0.52),\n"
    "        ('Return t-test p < 0.05',      p_ret,                    p_ret < 0.05),\n"
    "    ]\n"
    "    score = sum(1 for *_, p in ev if p)\n"
    "    print(f'\\n{label}  (Score {score}/{len(ev)})')\n"
    "    print('-' * 56)\n"
    "    for desc, val, passed in ev:\n"
    "        mark = '[+]' if passed else '[-]'\n"
    "        if abs(val) < 2:\n"
    "            print(f'  {mark} {desc:<38} {val:>+.3f}')\n"
    "        else:\n"
    "            print(f'  {mark} {desc:<38} {val:>+.1%}')\n"
    "    return score\n\n"
    "print('='*60)\n"
    "print('EEM + SECTOR TILT VERDICT')\n"
    "print('='*60)\n\n"
    "scores = {}\n"
    "scores['EEM L/flat EW (benchmark)'] = score_strat(net_eem_lf,      gross_eem_lf,      'EEM L/flat EW (NB16 benchmark)')\n"
    "scores['Variant A']                 = score_strat(res_A['net_ret'], res_A['gross_ret'], 'A: EEM L/flat + sector tilt')\n"
    "scores['Variant B']                 = score_strat(res_B['net_ret'], res_B['gross_ret'], 'B: EEM L/S + sector tilt')\n"
    "scores['Variant C']                 = score_strat(res_C['net_ret'], res_C['gross_ret'], 'C: EEM-gated sector L/S')\n\n"
    "best = max(scores, key=lambda k: scores[k])\n"
    "print(f'\\n{\"=\"*60}')\n"
    "print('SUMMARY')\n"
    "print(f'{\"=\"*60}')\n"
    "for name, s in scores.items():\n"
    "    improvement = s - scores['EEM L/flat EW (benchmark)']\n"
    "    marker = ' ← BEST' if name == best else ''\n"
    "    imp_str = f'(+{improvement} vs benchmark)' if improvement > 0 else f'({improvement} vs benchmark)'\n"
    "    print(f'  {name:<32} Score={s}/8  {imp_str}{marker}')\n\n"
    "bench_sharpe = risk_metrics(net_eem_lf)['sharpe']\n"
    "best_sharpe  = max(risk_metrics(r)['sharpe'] for r in\n"
    "                   [res_A['net_ret'], res_B['net_ret'], res_C['net_ret']])\n"
    "if best_sharpe > bench_sharpe + 0.05:\n"
    "    print(f'\\n=> COMBINATION ADDS VALUE: best variant Sharpe {best_sharpe:+.2f} vs benchmark {bench_sharpe:+.2f}')\n"
    "else:\n"
    "    print(f'\\n=> COMBINATION MARGINAL: best variant Sharpe {best_sharpe:+.2f} vs benchmark {bench_sharpe:+.2f}')\n"
    "    print('   EEM signal alone captures most of the alpha.')"
))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {OUT}  ({len(cells)} cells)")
