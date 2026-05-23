"""
gen_nb18_gold_fx.py -- generate notebooks/eda/18_gold_fx_signal.ipynb

Option 2: Asset class shift — does the EEM-style approach work for Gold and USD/THB?
Tests cross-asset signal discovery:
  - EEM lag → Gold, USD/THB (same signal, different targets)
  - Gold lag → SET (does gold predict Thai stocks?)
  - Signal grid: all pairwise (feature → target) IC map
  - Rule-based strategies per asset
  - Combined multi-asset portfolio
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'notebooks' / 'eda' / '18_gold_fx_signal.ipynb'

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": s}

cells = []

cells.append(md(
    "# Notebook 18 — Gold & FX Signal Discovery\n\n"
    "> *Does the EEM-lag approach extend to Gold and USD/THB? What cross-asset signals exist?*\n\n"
    "**Motivation:** NB12 showed Gold DirAcc = 65.2% (best of 3 targets) and NLP added +0.20 Sharpe.  \n"
    "Thai Gold market has high retail participation and tight USD/THB linkage — potentially more predictable than equities.\n\n"
    "| Section | Content |\n"
    "|---|---|\n"
    "| 1. Signal Grid | IC map: all key signals vs all targets |\n"
    "| 2. Gold Strategy | Best signal(s) for gold_ret_w |\n"
    "| 3. FX Strategy | Best signal(s) for USD/THB |\n"
    "| 4. Multi-Asset Portfolio | Combine SET + Gold + FX strategies |\n"
    "| 5. Verdict | Which assets are most predictable? |"
))

cells.append(md("---\n## Section 1 — Setup & Signal Grid"))

cells.append(code(
    "from pathlib import Path\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.colors as mcolors\n"
    "from scipy.stats import spearmanr\n"
    "warnings.filterwarnings('ignore')\n"
    "np.random.seed(42)\n\n"
    "ROOT = Path('../..')\n"
    "PROC = ROOT / 'data' / 'processed'\n"
    "FIGS = Path('figs')\n"
    "FIGS.mkdir(exist_ok=True)\n\n"
    "plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})\n\n"
    "df = pd.read_csv(PROC / 'unified_weekly_clean.csv', index_col=0, parse_dates=True).sort_index()\n\n"
    "TC = 0.001\n\n"
    "# ── Targets ──\n"
    "TARGETS = {\n"
    "    'SET':   'SET_index_ret_w_fwd1',\n"
    "    'Gold':  'gold_ret_w_fwd1',\n"
    "    'USDTHB':'USD_THB_ret_w_fwd1',\n"
    "}\n"
    "for k, col in TARGETS.items():\n"
    "    if col not in df.columns:\n"
    "        src = col.replace('_fwd1', '')\n"
    "        if src in df.columns:\n"
    "            df[col] = df[src].shift(-1)\n\n"
    "# ── Candidate signals (lag-1, no lookahead) ──\n"
    "SIGNALS = [\n"
    "    'eem_ret_d_lag1',\n"
    "    'sp500_ret_w',\n"
    "    'sp500_ret_d_lag1',\n"
    "    'gold_ret_w',           # gold predicts gold next week? (momentum)\n"
    "    'gold_ret_d_lag1',\n"
    "    'dxy_ret_w',\n"
    "    'dxy_ret_d_lag1',\n"
    "    'oil_ret_w',\n"
    "    'oil_ret_d_lag1',\n"
    "    'us_10yr_treasury_ret_w',\n"
    "    'vix_z',\n"
    "    'SET_index_ret_w',      # SET momentum\n"
    "    'SET_index_rvol_4w',\n"
    "    'gold_rvol_4w',\n"
    "    'yield_curve_slope',\n"
    "]\n"
    "SIGNALS = [s for s in SIGNALS if s in df.columns]\n"
    "print(f'Signals: {len(SIGNALS)}, Targets: {len(TARGETS)}')\n"
    "print(f'Data: {len(df)} weeks ({df.index[0].date()} -> {df.index[-1].date()})')"
))

cells.append(code(
    "# ── IC grid: all signals vs all targets ──\n"
    "ic_grid = pd.DataFrame(index=SIGNALS, columns=list(TARGETS.keys()), dtype=float)\n"
    "p_grid  = pd.DataFrame(index=SIGNALS, columns=list(TARGETS.keys()), dtype=float)\n\n"
    "for sig in SIGNALS:\n"
    "    for tgt_name, tgt_col in TARGETS.items():\n"
    "        sub = df[[sig, tgt_col]].dropna()\n"
    "        if len(sub) < 50:\n"
    "            continue\n"
    "        ic, p = spearmanr(sub[sig], sub[tgt_col])\n"
    "        ic_grid.loc[sig, tgt_name] = ic\n"
    "        p_grid.loc[sig,  tgt_name] = p\n\n"
    "# Plot heatmap\n"
    "fig, ax = plt.subplots(figsize=(8, 9))\n"
    "ic_vals = ic_grid.astype(float).values\n"
    "im = ax.imshow(ic_vals, cmap='RdBu', vmin=-0.15, vmax=0.15, aspect='auto')\n"
    "ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(list(TARGETS.keys()), fontsize=11)\n"
    "ax.set_yticks(range(len(SIGNALS))); ax.set_yticklabels(SIGNALS, fontsize=9)\n"
    "for i, sig in enumerate(SIGNALS):\n"
    "    for j, tgt in enumerate(TARGETS.keys()):\n"
    "        ic_v = ic_grid.loc[sig, tgt]\n"
    "        p_v  = p_grid.loc[sig, tgt]\n"
    "        if pd.isna(ic_v): continue\n"
    "        star = '*' if p_v < 0.05 else ''\n"
    "        ax.text(j, i, f'{ic_v:+.3f}{star}', ha='center', va='center', fontsize=8,\n"
    "                color='white' if abs(ic_v) > 0.08 else 'black')\n"
    "plt.colorbar(im, ax=ax, fraction=0.03, label='Spearman IC')\n"
    "ax.set_title('Cross-Asset IC Grid (* = p<0.05)', fontsize=11)\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'cross_asset_ic_grid.png', bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print('\\nTop signals per target:')\n"
    "for tgt in TARGETS:\n"
    "    top = ic_grid[tgt].abs().nlargest(3)\n"
    "    print(f'  {tgt}: ' + ', '.join(f'{s}={ic_grid.loc[s,tgt]:+.3f}(p={p_grid.loc[s,tgt]:.3f})' for s in top.index))"
))

cells.append(md("---\n## Section 2 — Gold Signal Strategy"))

cells.append(code(
    "TARGET_GOLD = TARGETS['Gold']\n\n"
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
    "def backtest_rule(signal, target, variant='lf', tc=TC):\n"
    "    sub = pd.concat([signal, target], axis=1).dropna()\n"
    "    sub.columns = ['sig', 'tgt']\n"
    "    pos = np.sign(sub['sig'])\n"
    "    if variant == 'lf': pos = pos.clip(lower=0)\n"
    "    pos_change = pos.diff().abs().fillna(pos.abs())\n"
    "    gross = pos * sub['tgt']\n"
    "    net   = gross - pos_change * tc\n"
    "    return gross, net\n\n"
    "# Find best signal for Gold\n"
    "gold_sigs = ic_grid['Gold'].dropna().abs().sort_values(ascending=False)\n"
    "print('Gold signal ranking (by |IC|):')\n"
    "print(f'{\"Signal\":<30} {\"IC\":>8} {\"p\":>8} {\"Sharpe L/flat\":>14}')\n"
    "print('-' * 65)\n"
    "for sig in gold_sigs.head(8).index:\n"
    "    ic_v = ic_grid.loc[sig, 'Gold']\n"
    "    p_v  = p_grid.loc[sig, 'Gold']\n"
    "    # sign-flip if IC is negative (short when signal up)\n"
    "    sig_ser = df[sig] * np.sign(ic_v)\n"
    "    _, net = backtest_rule(sig_ser, df[TARGET_GOLD], 'lf')\n"
    "    m = risk_metrics(net)\n"
    "    sig_flag = ' *' if p_v < 0.05 else ''\n"
    "    print(f'{sig:<30} {ic_v:>+8.4f} {p_v:>8.4f}{sig_flag} {m.get(\"sharpe\",np.nan):>14.3f}')"
))

cells.append(code(
    "# Best Gold strategy: top-2 signals combined\n"
    "best_gold_sigs = ic_grid['Gold'].dropna().abs().nlargest(2).index.tolist()\n"
    "print(f'Best Gold signals: {best_gold_sigs}')\n\n"
    "# Composite signal: average of normalised best signals\n"
    "gold_composite = pd.Series(0.0, index=df.index)\n"
    "for sig in best_gold_sigs:\n"
    "    ic_v   = ic_grid.loc[sig, 'Gold']\n"
    "    s_norm = df[sig] / df[sig].rolling(52, min_periods=26).std()\n"
    "    gold_composite += np.sign(ic_v) * s_norm\n"
    "gold_composite = gold_composite / len(best_gold_sigs)\n\n"
    "# Backtest\n"
    "gross_gold, net_gold = backtest_rule(gold_composite, df[TARGET_GOLD], 'lf')\n"
    "bnh_gold = df[TARGET_GOLD]\n\n"
    "m_gold = risk_metrics(net_gold)\n"
    "m_bnh_gold = risk_metrics(bnh_gold)\n"
    "print(f'Gold composite L/flat:  Sharpe={m_gold[\"sharpe\"]:+.3f}  AnnRet={m_gold[\"ann_ret\"]:+.2%}  MaxDD={m_gold[\"max_dd\"]:+.1%}')\n"
    "print(f'Gold Buy & Hold:        Sharpe={m_bnh_gold[\"sharpe\"]:+.3f}  AnnRet={m_bnh_gold[\"ann_ret\"]:+.2%}  MaxDD={m_bnh_gold[\"max_dd\"]:+.1%}')\n\n"
    "# Also test single best signal\n"
    "best1 = best_gold_sigs[0]\n"
    "_, net_gold1 = backtest_rule(df[best1] * np.sign(ic_grid.loc[best1,'Gold']), df[TARGET_GOLD], 'lf')\n"
    "print(f'Gold single-signal ({best1[:20]}): Sharpe={risk_metrics(net_gold1)[\"sharpe\"]:+.3f}')"
))

cells.append(md("---\n## Section 3 — FX (USD/THB) Signal Strategy"))

cells.append(code(
    "TARGET_FX = TARGETS['USDTHB']\n\n"
    "# Find best signal for USD/THB\n"
    "fx_sigs = ic_grid['USDTHB'].dropna().abs().sort_values(ascending=False)\n"
    "print('USD/THB signal ranking (by |IC|):')\n"
    "print(f'{\"Signal\":<30} {\"IC\":>8} {\"p\":>8} {\"Sharpe L/flat\":>14}')\n"
    "print('-' * 65)\n"
    "for sig in fx_sigs.head(8).index:\n"
    "    ic_v = ic_grid.loc[sig, 'USDTHB']\n"
    "    p_v  = p_grid.loc[sig, 'USDTHB']\n"
    "    sig_ser = df[sig] * np.sign(ic_v)\n"
    "    _, net = backtest_rule(sig_ser, df[TARGET_FX], 'lf')\n"
    "    m = risk_metrics(net)\n"
    "    sig_flag = ' *' if p_v < 0.05 else ''\n"
    "    print(f'{sig:<30} {ic_v:>+8.4f} {p_v:>8.4f}{sig_flag} {m.get(\"sharpe\",np.nan):>14.3f}')"
))

cells.append(code(
    "# Best FX composite\n"
    "best_fx_sigs = ic_grid['USDTHB'].dropna().abs().nlargest(2).index.tolist()\n"
    "print(f'Best FX signals: {best_fx_sigs}')\n\n"
    "fx_composite = pd.Series(0.0, index=df.index)\n"
    "for sig in best_fx_sigs:\n"
    "    ic_v   = ic_grid.loc[sig, 'USDTHB']\n"
    "    s_norm = df[sig] / df[sig].rolling(52, min_periods=26).std()\n"
    "    fx_composite += np.sign(ic_v) * s_norm\n"
    "fx_composite /= len(best_fx_sigs)\n\n"
    "gross_fx, net_fx = backtest_rule(fx_composite, df[TARGET_FX], 'lf')\n"
    "bnh_fx = df[TARGET_FX]\n"
    "m_fx = risk_metrics(net_fx)\n"
    "print(f'FX composite L/flat:  Sharpe={m_fx[\"sharpe\"]:+.3f}  AnnRet={m_fx[\"ann_ret\"]:+.2%}')\n"
    "print(f'FX Buy & Hold:        Sharpe={risk_metrics(bnh_fx)[\"sharpe\"]:+.3f}')"
))

cells.append(md(
    "---\n## Section 4 — Multi-Asset Portfolio\n\n"
    "Combine SET + Gold + FX strategies into a diversified portfolio.  \n"
    "Equal weight each asset's signal-driven position."
))

cells.append(code(
    "# ── SET signal (from NB16/17) ──\n"
    "TARGET_SET  = TARGETS['SET']\n"
    "eem_sig     = df['eem_ret_d_lag1']\n"
    "pos_set_lf  = np.sign(eem_sig).clip(lower=0)\n"
    "pos_chg_set = pos_set_lf.diff().abs().fillna(pos_set_lf.abs())\n"
    "net_set = pos_set_lf * df[TARGET_SET] - pos_chg_set * TC\n\n"
    "# ── Align all series to common dates ──\n"
    "common = net_set.dropna().index\\\n"
    "         .intersection(net_gold.dropna().index)\\\n"
    "         .intersection(net_fx.dropna().index)\n\n"
    "r_set   = net_set.reindex(common)\n"
    "r_gold  = net_gold.reindex(common)\n"
    "r_fx    = net_fx.reindex(common)\n"
    "r_bnh   = df[TARGET_SET].reindex(common)   # SET benchmark\n\n"
    "# Equal-weight multi-asset portfolio\n"
    "r_multi = (r_set + r_gold + r_fx) / 3\n\n"
    "# Also: SET + Gold only (exclude FX if low IC)\n"
    "r_set_gold = (r_set + r_gold) / 2\n\n"
    "print(f'{\"Strategy\":<30} {\"Sharpe\":>8} {\"AnnRet\":>9} {\"MaxDD\":>8} {\"WinRate\":>8}')\n"
    "print('-' * 68)\n"
    "for name, rets in [\n"
    "    ('SET only (EEM L/flat)',       r_set),\n"
    "    ('Gold only',                   r_gold),\n"
    "    ('FX only',                     r_fx),\n"
    "    ('SET + Gold (50/50)',           r_set_gold),\n"
    "    ('SET + Gold + FX (33/33/33)',   r_multi),\n"
    "    ('SET Buy & Hold',              r_bnh),\n"
    "]:\n"
    "    m = risk_metrics(rets)\n"
    "    if not m: continue\n"
    "    flag = ' ✓' if m['sharpe'] > 0.5 else ''\n"
    "    print(f'{name:<30} {m[\"sharpe\"]:>8.3f} {m[\"ann_ret\"]:>9.2%} {m[\"max_dd\"]:>8.1%} {m[\"win_rate\"]:>8.1%}{flag}')"
))

cells.append(code(
    "# ── Equity curves ──\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)\n"
    "fig.suptitle('Multi-Asset Signal Portfolio: SET + Gold + FX', fontsize=12)\n\n"
    "ax = axes[0]\n"
    "for rets, lbl, col, lw, ls in [\n"
    "    (r_set,       'SET (EEM L/flat)',       '#1976D2', 1.5, '--'),\n"
    "    (r_gold,      'Gold signal',            '#FF8F00', 1.5, '--'),\n"
    "    (r_fx,        'FX signal',              '#7B1FA2', 1.5, '--'),\n"
    "    (r_set_gold,  'SET+Gold (50/50)',        '#2E7D32', 2.0, '-'),\n"
    "    (r_multi,     'SET+Gold+FX (equal wt)', '#C62828', 2.0, '-'),\n"
    "    (r_bnh,       'SET B&H',                '#9E9E9E', 1.0, ':'),\n"
    "]:\n"
    "    m   = risk_metrics(rets)\n"
    "    cum = (1 + rets.dropna()).cumprod()\n"
    "    ax.plot(cum, lw=lw, ls=ls, color=col,\n"
    "            label=f'{lbl}  Sharpe={m.get(\"sharpe\",0):+.2f}')\n"
    "ax.axhline(1.0, color='gray', lw=0.5, ls=':')\n"
    "ax.set_yscale('log')\n"
    "ax.set_ylabel('Cumulative Return (log)')\n"
    "ax.legend(fontsize=8, loc='upper left')\n"
    "ax.set_title('Strategy Equity Curves (net of TC)')\n\n"
    "# Correlation matrix of strategy returns\n"
    "ret_mat = pd.DataFrame({'SET':r_set,'Gold':r_gold,'FX':r_fx,'B&H':r_bnh}).dropna()\n"
    "corr = ret_mat.corr()\n"
    "ax2 = axes[1]\n"
    "im2 = ax2.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1)\n"
    "ax2.set_xticks(range(4)); ax2.set_xticklabels(corr.columns)\n"
    "ax2.set_yticks(range(4)); ax2.set_yticklabels(corr.index)\n"
    "for i in range(4):\n"
    "    for j in range(4):\n"
    "        ax2.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=10)\n"
    "plt.colorbar(im2, ax=ax2, fraction=0.03)\n"
    "ax2.set_title('Strategy Return Correlations')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'multi_asset_equity.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/multi_asset_equity.png')"
))

cells.append(md("---\n## Section 5 — Verdict"))

cells.append(code(
    "from scipy.stats import ttest_1samp\n\n"
    "print('='*65)\n"
    "print('CROSS-ASSET SIGNAL VERDICT')\n"
    "print('='*65)\n"
    "print(f'{\"Asset\":<12} {\"Best signal\":<28} {\"IC\":>6} {\"p\":>7} {\"Sharpe\":>8} {\"Viable?\"}')\n"
    "print('-'*75)\n\n"
    "for asset, tgt_col, strategy_ret in [\n"
    "    ('SET',    TARGET_SET,  r_set),\n"
    "    ('Gold',   TARGET_GOLD, r_gold),\n"
    "    ('USD/THB',TARGET_FX,   r_fx),\n"
    "]:\n"
    "    best_sig = ic_grid[asset if asset != 'USD/THB' else 'USDTHB'].dropna().abs().idxmax()\n"
    "    tgt_key  = asset if asset != 'USD/THB' else 'USDTHB'\n"
    "    ic_v  = ic_grid.loc[best_sig, tgt_key]\n"
    "    p_v   = p_grid.loc[best_sig, tgt_key]\n"
    "    m     = risk_metrics(strategy_ret)\n"
    "    viable = 'YES ✓' if m.get('sharpe',0) > 0.3 and p_v < 0.05 else 'WEAK' if m.get('sharpe',0) > 0 else 'NO'\n"
    "    print(f'{asset:<12} {best_sig:<28} {ic_v:>+6.3f} {p_v:>7.4f} {m.get(\"sharpe\",0):>8.3f} {viable}')\n\n"
    "print(f'\\nMulti-asset portfolio (SET+Gold+FX):')\n"
    "m_multi = risk_metrics(r_multi)\n"
    "_, p_multi = ttest_1samp(r_multi.dropna(), 0)\n"
    "print(f'  Sharpe={m_multi[\"sharpe\"]:+.3f}  AnnRet={m_multi[\"ann_ret\"]:+.2%}  MaxDD={m_multi[\"max_dd\"]:+.1%}  p={p_multi:.4f}')\n"
    "if m_multi['sharpe'] > risk_metrics(r_set)['sharpe']:\n"
    "    print('  => Multi-asset IMPROVES on single-asset SET strategy')\n"
    "else:\n"
    "    print('  => Multi-asset does NOT improve on SET-only')"
))

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
