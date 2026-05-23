"""
gen_nb16_eem_signal.py -- generate notebooks/eda/16_eem_signal.ipynb

EEM Lag Signal Strategy: the clearest alpha found in this project.
eem_ret_d_lag1 = EEM's last daily return (Friday close, available at Monday open)
IC = 0.11, perm p<0.05, sign-stable 2000-2025 (from NB08 permutation tests)

Tests:
  Rule-based L/S:   sign(eem_ret_d_lag1) × SET position
  Rule-based L/flat: long when positive, flat when negative
  Magnitude-scaled: position ∝ eem_ret_d_lag1 (normalised)
  1-feature XGB:    walk-forward ML with eem only (benchmark for rule vs ML)

Regime and TC sensitivity analysis included.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'notebooks' / 'eda' / '16_eem_signal.ipynb'

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": s}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Notebook 16 — EEM Lag Signal Strategy\n\n"
    "> *The clearest alpha in the Thai market: EEM's last daily return predicts SET next week.*\n\n"
    "**Signal:** `eem_ret_d_lag1` = EEM (emerging-market ETF) daily return on the most recent day  \n"
    "**Evidence from NB08:** IC = +0.11, permutation p < 0.05, sign-stable 2000–2025, ICIR = 1.47  \n"
    "**Intuition:** Global EM risk sentiment spills into SET with a 1-week lag (liquidity/reporting delay)\n\n"
    "| Section | Content |\n"
    "|---|---|\n"
    "| 1. Signal Analysis | Distribution, rolling IC, sign stability |\n"
    "| 2. Rule-Based Strategy | Long/short + long/flat variants |\n"
    "| 3. Magnitude-Scaled | Position ∝ signal strength |\n"
    "| 4. 1-Feature XGB | Walk-forward ML (rule vs ML comparison) |\n"
    "| 5. TC Sensitivity | Break-even TC, optimal rebalancing |\n"
    "| 6. Regime Analysis | Does signal hold across regimes? |\n"
    "| 7. Verdict | Final scorecard |"
))

# ── Section 1 — Setup & Signal Analysis ───────────────────────────────────────
cells.append(md("---\n## Section 1 — Setup & Signal Analysis"))

cells.append(code(
    "from pathlib import Path\n"
    "import warnings\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from scipy.stats import spearmanr, ttest_1samp, pearsonr\n"
    "from xgboost import XGBRegressor\n"
    "warnings.filterwarnings('ignore')\n"
    "np.random.seed(42)\n\n"
    "ROOT = Path('../..')\n"
    "PROC = ROOT / 'data' / 'processed'\n"
    "FIGS = Path('figs')\n"
    "FIGS.mkdir(exist_ok=True)\n\n"
    "plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})\n\n"
    "# Load clean weekly data\n"
    "df = pd.read_csv(PROC / 'unified_weekly_clean.csv', index_col=0, parse_dates=True)\n"
    "df = df.sort_index()\n\n"
    "SIGNAL   = 'eem_ret_d_lag1'\n"
    "TARGET   = 'SET_index_ret_w_fwd1'\n"
    "TC       = 0.001   # 0.1% one-way\n\n"
    "# Use pre-computed forward target if available, else compute it\n"
    "if TARGET not in df.columns:\n"
    "    df[TARGET] = df['SET_index_ret_w'].shift(-1)\n\n"
    "data = df[[SIGNAL, TARGET, 'regime']].dropna().copy()\n"
    "print(f'Data: {len(data)} weeks  ({data.index[0].date()} → {data.index[-1].date()})')\n"
    "print(f'Signal:  {SIGNAL}')\n"
    "print(f'Target:  {TARGET}')\n"
    "print(f'\\nSignal stats:')\n"
    "print(data[SIGNAL].describe().round(4))"
))

cells.append(code(
    "# ── Rolling IC (52-week window) ──\n"
    "ic_roll = []\n"
    "dates_ic = []\n"
    "WIN = 52\n"
    "for i in range(WIN, len(data)):\n"
    "    sub = data.iloc[i-WIN:i]\n"
    "    ic, _ = spearmanr(sub[SIGNAL], sub[TARGET])\n"
    "    ic_roll.append(ic)\n"
    "    dates_ic.append(data.index[i])\n"
    "ic_roll = pd.Series(ic_roll, index=dates_ic)\n\n"
    "# Full-period IC\n"
    "ic_full, p_full = spearmanr(data[SIGNAL], data[TARGET])\n"
    "t_stat = ic_full / (np.sqrt((1-ic_full**2)/(len(data)-2)))\n"
    "print(f'Full-period Spearman IC: {ic_full:+.4f}  (t={t_stat:+.2f}, p={p_full:.4f})')\n"
    "print(f'Rolling IC (52w):  mean={ic_roll.mean():+.4f}  std={ic_roll.std():.4f}')\n"
    "print(f'IC > 0: {(ic_roll>0).mean():.1%} of rolling windows')\n\n"
    "# Plot\n"
    "fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)\n"
    "fig.suptitle('EEM Lag Signal: Distribution & Rolling IC', fontsize=12)\n\n"
    "ax = axes[0]\n"
    "ax.bar(ic_roll.index, ic_roll, color=['#1976D2' if x>0 else '#EF5350' for x in ic_roll],\n"
    "       alpha=0.5, width=7)\n"
    "ax.axhline(0, color='black', lw=0.8, ls='--')\n"
    "ax.axhline(ic_full, color='#1976D2', lw=1.5, ls='-',\n"
    "           label=f'Full-period IC = {ic_full:+.4f} (p={p_full:.3f})')\n"
    "ax.set_ylabel('Spearman IC')\n"
    "ax.set_title(f'52-Week Rolling IC: eem_ret_d_lag1 vs SET fwd1w  (IC>0 in {(ic_roll>0).mean():.0%} of windows)')\n"
    "ax.legend(fontsize=9)\n"
    "ax.set_ylim(-0.5, 0.5)\n\n"
    "# Scatter: signal vs target\n"
    "ax2 = axes[1]\n"
    "ax2.scatter(data[SIGNAL], data[TARGET], alpha=0.2, s=8, color='#1976D2')\n"
    "# Regression line\n"
    "m, b = np.polyfit(data[SIGNAL], data[TARGET], 1)\n"
    "xs = np.linspace(data[SIGNAL].quantile(0.01), data[SIGNAL].quantile(0.99), 100)\n"
    "ax2.plot(xs, m*xs+b, color='#EF5350', lw=2, label=f'slope={m:.2f}')\n"
    "ax2.axhline(0, color='gray', lw=0.5, ls=':')\n"
    "ax2.axvline(0, color='gray', lw=0.5, ls=':')\n"
    "ax2.set_xlabel('eem_ret_d_lag1 (signal)')\n"
    "ax2.set_ylabel('SET 1w fwd return (target)')\n"
    "ax2.set_title(f'Signal vs Target Scatter (IC={ic_full:+.4f})')\n"
    "ax2.legend(fontsize=9)\n"
    "ax2.set_xlim(data[SIGNAL].quantile(0.01), data[SIGNAL].quantile(0.99))\n"
    "ax2.set_ylim(data[TARGET].quantile(0.01), data[TARGET].quantile(0.99))\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'eem_signal_ic.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/eem_signal_ic.png')"
))

cells.append(code(
    "# ── IC at different forecast horizons ──\n"
    "print('IC vs forecast horizon (full period):')\n"
    "print(f'{\"Horizon\":>10} {\"IC\":>10} {\"p\":>10} {\"IC+%\":>8}')\n"
    "print('-' * 42)\n"
    "for h in [1, 2, 4, 8, 13]:\n"
    "    fwd_h = df['SET_index_ret_w'].shift(-h)\n"
    "    sub = pd.concat([df[SIGNAL], fwd_h], axis=1).dropna()\n"
    "    sub.columns = ['sig', 'tgt']\n"
    "    ic, p = spearmanr(sub['sig'], sub['tgt'])\n"
    "    # rolling IC\n"
    "    ic_r = [spearmanr(sub.iloc[i-52:i]['sig'], sub.iloc[i-52:i]['tgt'])[0]\n"
    "            for i in range(52, len(sub))]\n"
    "    pos_frac = (np.array(ic_r) > 0).mean()\n"
    "    sig = ' *' if p < 0.05 else ''\n"
    "    print(f'{h:>7}w fwd {ic:>+10.4f} {p:>10.4f}{sig} {pos_frac:>8.1%}')"
))

# ── Section 2 — Rule-Based Strategy ───────────────────────────────────────────
cells.append(md(
    "---\n## Section 2 — Rule-Based Strategy\n\n"
    "**Simplest possible implementation:**\n"
    "- `sign(eem_ret_d_lag1) > 0` → **Long SET** (+1)\n"
    "- `sign(eem_ret_d_lag1) < 0` → **Short SET** (−1) [L/S variant] or **Flat** [L/flat variant]\n"
    "- Weekly rebalancing (position changes when signal sign changes)\n"
    "- TC applied only when position changes"
))

cells.append(code(
    "def backtest_rule(data, variant='ls', tc=TC):\n"
    "    \"\"\"\n"
    "    variant: 'ls' = long/short, 'lf' = long/flat\n"
    "    Returns series of weekly net returns.\n"
    "    \"\"\"\n"
    "    sig = np.sign(data[SIGNAL])   # +1 or -1\n"
    "    if variant == 'lf':\n"
    "        sig = sig.clip(lower=0)   # 0 or +1\n\n"
    "    # Position changes → incur TC\n"
    "    pos_change = sig.diff().abs().fillna(sig.abs())  # first bar = entry\n"
    "    tc_series  = pos_change * tc\n\n"
    "    gross_ret = sig * data[TARGET]\n"
    "    net_ret   = gross_ret - tc_series\n"
    "    return gross_ret, net_ret, sig\n\n"
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
    "gross_ls, net_ls, pos_ls = backtest_rule(data, 'ls')\n"
    "gross_lf, net_lf, pos_lf = backtest_rule(data, 'lf')\n"
    "bnh = data[TARGET]   # buy & hold SET\n\n"
    "m_ls = risk_metrics(net_ls)\n"
    "m_lf = risk_metrics(net_lf)\n"
    "m_bnh = risk_metrics(bnh)\n\n"
    "# Signal turnover\n"
    "turn_ls = pos_ls.diff().abs().mean()\n"
    "turn_lf = pos_lf.diff().abs().mean()\n\n"
    "print(f'{\"Strategy\":<28} {\"AnnRet\":>8} {\"Sharpe\":>8} {\"MaxDD\":>8} {\"WinRate\":>8} {\"Turnover\":>10}')\n"
    "print('-' * 75)\n"
    "for lbl, gross, net, pos in [\n"
    "    ('EEM Rule L/S (gross)', gross_ls, gross_ls, pos_ls),\n"
    "    ('EEM Rule L/S (net TC)', gross_ls, net_ls,  pos_ls),\n"
    "    ('EEM Rule L/flat (gross)', gross_lf, gross_lf, pos_lf),\n"
    "    ('EEM Rule L/flat (net TC)', gross_lf, net_lf, pos_lf),\n"
    "    ('SET Buy & Hold', bnh, bnh, None),\n"
    "]:\n"
    "    m = risk_metrics(net)\n"
    "    turn = pos.diff().abs().mean() if pos is not None else np.nan\n"
    "    print(f'{lbl:<28} {m[\"ann_ret\"]:>8.2%} {m[\"sharpe\"]:>8.3f} '\n"
    "          f'{m[\"max_dd\"]:>8.1%} {m[\"win_rate\"]:>8.1%} {turn:>10.3f}')"
))

cells.append(code(
    "# ── Equity curves ──\n"
    "fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={'height_ratios':[3,1]})\n"
    "fig.suptitle('EEM Rule-Based Signal: Equity Curves', fontsize=12)\n\n"
    "ax = axes[0]\n"
    "for rets, lbl, col, lw, ls in [\n"
    "    (net_ls,  'EEM L/S (net TC)',    '#1976D2', 2.0, '-'),\n"
    "    (net_lf,  'EEM L/flat (net TC)', '#7B1FA2', 2.0, '-'),\n"
    "    (gross_ls,'EEM L/S (gross)',     '#90CAF9', 1.0, '--'),\n"
    "    (bnh,     'SET Buy & Hold',      '#4CAF50', 1.5, ':'),\n"
    "]:\n"
    "    cum = (1 + rets.dropna()).cumprod()\n"
    "    m   = risk_metrics(rets)\n"
    "    ax.plot(cum, lw=lw, ls=ls, color=col,\n"
    "            label=f'{lbl}  Sharpe={m[\"sharpe\"]:+.2f}  AnnRet={m[\"ann_ret\"]:+.1%}')\n"
    "ax.axhline(1.0, color='gray', lw=0.5, ls=':')\n"
    "ax.set_yscale('log')\n"
    "ax.set_ylabel('Cumulative Return (log)')\n"
    "ax.legend(fontsize=8, loc='upper left')\n"
    "ax.set_title('EEM Lag Signal Strategy — Full Period 2000–2025')\n\n"
    "# Drawdown of L/S net\n"
    "cum_ls = (1 + net_ls.dropna()).cumprod()\n"
    "dd = (cum_ls - cum_ls.cummax()) / cum_ls.cummax()\n"
    "axes[1].fill_between(dd.index, dd, 0, color='#EF5350', alpha=0.5)\n"
    "axes[1].set_ylabel('Drawdown')\n"
    "axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f'{y:.0%}'))\n"
    "axes[1].set_title('EEM L/S Drawdown')\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'eem_equity.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/eem_equity.png')"
))

# ── Section 3 — Magnitude-Scaled ──────────────────────────────────────────────
cells.append(md(
    "---\n## Section 3 — Magnitude-Scaled Position\n\n"
    "Scale position by **signal strength** (not just sign).  \n"
    "Intuition: bigger EEM move → higher conviction → larger SET position.\n\n"
    "`position(t) = clip( eem_ret_d_lag1(t) / vol_normaliser, -1, +1 )`  \n"
    "where `vol_normaliser` = rolling 52w std of eem_ret_d_lag1."
))

cells.append(code(
    "# Normalise signal by rolling vol → position in [-1, +1]\n"
    "sig_vol  = data[SIGNAL].rolling(52, min_periods=26).std()\n"
    "pos_scaled = (data[SIGNAL] / sig_vol).clip(-2, 2) / 2   # scale to [-1, +1]\n"
    "pos_scaled = pos_scaled.fillna(np.sign(data[SIGNAL]))   # fallback to sign\n\n"
    "# Position change for TC\n"
    "pos_change_sc = pos_scaled.diff().abs().fillna(pos_scaled.abs())\n"
    "gross_sc = pos_scaled * data[TARGET]\n"
    "net_sc   = gross_sc - pos_change_sc * TC\n\n"
    "m_sc = risk_metrics(net_sc)\n"
    "print(f'Magnitude-scaled (net TC):')\n"
    "print(f'  Sharpe:     {m_sc[\"sharpe\"]:+.3f}')\n"
    "print(f'  Ann Return: {m_sc[\"ann_ret\"]:+.2%}')\n"
    "print(f'  Max DD:     {m_sc[\"max_dd\"]:+.1%}')\n"
    "print(f'  Win rate:   {m_sc[\"win_rate\"]:.1%}')\n"
    "print(f'  Avg pos:    {pos_scaled.abs().mean():.3f} (0=flat, 1=full)')"
))

# ── Section 4 — 1-Feature XGB ─────────────────────────────────────────────────
cells.append(md(
    "---\n## Section 4 — 1-Feature XGB (Walk-Forward)\n\n"
    "Train XGBoost on **only `eem_ret_d_lag1`** — asks: does ML extract more from this  \n"
    "single feature than the simple sign rule? If rule ≈ XGB, simple wins (Occam's razor)."
))

cells.append(code(
    "def wf_xgb_1feat(data, signal, target, min_train=156, fold_size=52, tc=TC):\n"
    "    dates = data.index.tolist()\n"
    "    n = len(dates)\n"
    "    records = []\n"
    "    t = min_train\n"
    "    prev_pos = 0.0\n"
    "    while t < n:\n"
    "        fold_end = min(t + fold_size, n)\n"
    "        train = data.iloc[:t]\n"
    "        test  = data.iloc[t:fold_end]\n\n"
    "        Xtr = train[[signal]].values.astype(float)\n"
    "        ytr = train[target].values\n"
    "        Xte = test[[signal]].values.astype(float)\n\n"
    "        m = XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1,\n"
    "                         subsample=0.8, min_child_weight=10,\n"
    "                         random_state=42, verbosity=0, n_jobs=1)\n"
    "        m.fit(Xtr, ytr)\n"
    "        preds = m.predict(Xte)\n\n"
    "        # Convert continuous prediction to ±1 position\n"
    "        pos = np.sign(preds)\n"
    "        for i, (date, pred, p) in enumerate(zip(test.index, preds, pos)):\n"
    "            actual = test[target].iloc[i]\n"
    "            tc_cost = abs(p - prev_pos) * tc\n"
    "            gross   = p * actual\n"
    "            records.append({'date':date,'pred':pred,'pos':p,\n"
    "                            'gross_ret':gross,'net_ret':gross-tc_cost,\n"
    "                            'actual':actual})\n"
    "            prev_pos = p\n"
    "        t = fold_end\n"
    "    return pd.DataFrame(records).set_index('date')\n\n"
    "print('Running 1-feature XGB walk-forward...')\n"
    "res_xgb1 = wf_xgb_1feat(data, SIGNAL, TARGET)\n"
    "m_xgb1 = risk_metrics(res_xgb1['net_ret'])\n"
    "ic_xgb1, p_xgb1 = spearmanr(res_xgb1['pred'], res_xgb1['actual'])\n\n"
    "print(f'1-Feature XGB (net TC):')\n"
    "print(f'  Sharpe:     {m_xgb1[\"sharpe\"]:+.3f}')\n"
    "print(f'  Ann Return: {m_xgb1[\"ann_ret\"]:+.2%}')\n"
    "print(f'  IC:         {ic_xgb1:+.4f}  (p={p_xgb1:.4f})')\n"
    "print(f'\\nRule L/S comparison:')\n"
    "print(f'  Rule Sharpe:  {m_ls[\"sharpe\"]:+.3f}')\n"
    "print(f'  XGB1 Sharpe:  {m_xgb1[\"sharpe\"]:+.3f}')\n"
    "print(f'  => {\"XGB better\" if m_xgb1[\"sharpe\"] > m_ls[\"sharpe\"] else \"Rule better (simpler wins)\"}')"
))

# ── Section 5 — TC Sensitivity ────────────────────────────────────────────────
cells.append(md(
    "---\n## Section 5 — TC Sensitivity & Break-Even\n\n"
    "How much TC can the strategy absorb before going negative?"
))

cells.append(code(
    "tc_levels = [0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]\n"
    "print(f'{\"TC (one-way)\":>14} {\"L/S Sharpe\":>12} {\"L/flat Sharpe\":>14} {\"L/S AnnRet\":>12}')\n"
    "print('-' * 56)\n"
    "breakeven_ls = None\n"
    "breakeven_lf = None\n"
    "tc_results = []\n"
    "for tc_val in tc_levels:\n"
    "    _, net_ls_tc, _ = backtest_rule(data, 'ls', tc=tc_val)\n"
    "    _, net_lf_tc, _ = backtest_rule(data, 'lf', tc=tc_val)\n"
    "    m_ls_tc = risk_metrics(net_ls_tc)\n"
    "    m_lf_tc = risk_metrics(net_lf_tc)\n"
    "    print(f'{tc_val:>14.3%} {m_ls_tc[\"sharpe\"]:>12.3f} {m_lf_tc[\"sharpe\"]:>14.3f} '\n"
    "          f'{m_ls_tc[\"ann_ret\"]:>12.2%}')\n"
    "    tc_results.append((tc_val, m_ls_tc['sharpe'], m_lf_tc['sharpe']))\n"
    "    if breakeven_ls is None and m_ls_tc['sharpe'] < 0:\n"
    "        breakeven_ls = tc_val\n"
    "    if breakeven_lf is None and m_lf_tc['sharpe'] < 0:\n"
    "        breakeven_lf = tc_val\n\n"
    "print(f'\\nBreak-even TC (Sharpe = 0):')\n"
    "print(f'  L/S strategy: ~{breakeven_ls:.2%} one-way' if breakeven_ls else '  L/S: survives all TC levels')\n"
    "print(f'  L/flat strategy: ~{breakeven_lf:.2%} one-way' if breakeven_lf else '  L/flat: survives all TC levels')\n\n"
    "# Plot\n"
    "tc_arr = np.linspace(0, 0.01, 100)\n"
    "sharpe_ls_arr = [risk_metrics(backtest_rule(data,'ls',tc=t)[1])['sharpe'] for t in tc_arr]\n"
    "sharpe_lf_arr = [risk_metrics(backtest_rule(data,'lf',tc=t)[1])['sharpe'] for t in tc_arr]\n\n"
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "ax.plot(tc_arr*100, sharpe_ls_arr, lw=2, color='#1976D2', label='L/S')\n"
    "ax.plot(tc_arr*100, sharpe_lf_arr, lw=2, color='#7B1FA2', label='L/flat')\n"
    "ax.axhline(0, color='gray', lw=1, ls='--')\n"
    "ax.axhline(0.5, color='green', lw=0.8, ls=':', label='Sharpe = 0.5 target')\n"
    "ax.set_xlabel('TC one-way (%)')\n"
    "ax.set_ylabel('Net Sharpe')\n"
    "ax.set_title('EEM Signal: Sharpe vs Transaction Cost')\n"
    "ax.legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'eem_tc_sensitivity.png', bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved: figs/eem_tc_sensitivity.png')"
))

# ── Section 6 — Regime Analysis ───────────────────────────────────────────────
cells.append(md(
    "---\n## Section 6 — Regime Analysis\n\n"
    "Does the EEM signal hold across market regimes (VIX-based)?  \n"
    "Regime: −1 = Bull (low VIX), 0 = Normal, +1 = Crisis (high VIX)"
))

cells.append(code(
    "regime_labels = {-1:'Bull (low VIX)', 0:'Normal', 1:'Crisis (high VIX)'}\n\n"
    "print(f'{\"Regime\":<22} {\"N wks\":>6} {\"IC\":>8} {\"p\":>8} {\"Sharpe L/S\":>12} {\"Sharpe L/flat\":>14}')\n"
    "print('-' * 74)\n\n"
    "for r, lbl in sorted(regime_labels.items()):\n"
    "    sub = data[data['regime'] == r]\n"
    "    if len(sub) < 20:\n"
    "        print(f'{lbl:<22} {len(sub):>6}  -- insufficient')\n"
    "        continue\n"
    "    ic_r, p_r = spearmanr(sub[SIGNAL], sub[TARGET])\n"
    "    _, net_ls_r, _ = backtest_rule(sub, 'ls')\n"
    "    _, net_lf_r, _ = backtest_rule(sub, 'lf')\n"
    "    m_ls_r = risk_metrics(net_ls_r)\n"
    "    m_lf_r = risk_metrics(net_lf_r)\n"
    "    sig = ' *' if p_r < 0.05 else ''\n"
    "    print(f'{lbl:<22} {len(sub):>6} {ic_r:>+8.4f} {p_r:>8.4f}{sig} '\n"
    "          f'{m_ls_r[\"sharpe\"]:>12.3f} {m_lf_r[\"sharpe\"]:>14.3f}')\n\n"
    "# Annual performance\n"
    "annual_ls  = net_ls.groupby(net_ls.index.year).apply(lambda r: (1+r).prod()-1)\n"
    "annual_bnh = bnh.groupby(bnh.index.year).apply(lambda r: (1+r).prod()-1)\n\n"
    "fig, ax = plt.subplots(figsize=(13, 4))\n"
    "x = range(len(annual_ls))\n"
    "w = 0.35\n"
    "ax.bar([i-w/2 for i in x], annual_ls*100, w,\n"
    "       color=['#1976D2' if v>0 else '#EF5350' for v in annual_ls],\n"
    "       label='EEM L/S (net TC)')\n"
    "ax.bar([i+w/2 for i in x], annual_bnh*100, w,\n"
    "       color='#78909C', alpha=0.5, label='SET B&H')\n"
    "ax.axhline(0, color='black', lw=0.8)\n"
    "ax.set_xticks(x); ax.set_xticklabels(annual_ls.index, rotation=45)\n"
    "ax.set_ylabel('Annual Return (%)')\n"
    "ax.set_title('Annual Returns: EEM L/S vs SET Buy & Hold')\n"
    "ax.legend(fontsize=9)\n"
    "pos_yrs = (annual_ls > 0).sum()\n"
    "print(f'\\nPositive years: {pos_yrs}/{len(annual_ls)} ({pos_yrs/len(annual_ls):.0%})')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGS / 'eem_annual.png', bbox_inches='tight')\n"
    "plt.show()"
))

# ── Section 7 — Verdict ───────────────────────────────────────────────────────
cells.append(md("---\n## Section 7 — Final Verdict"))

cells.append(code(
    "from scipy.stats import ttest_1samp\n\n"
    "def full_verdict(net_ret, gross_ret, label):\n"
    "    m  = risk_metrics(net_ret)\n"
    "    mg = risk_metrics(gross_ret)\n"
    "    ic, p_ic = spearmanr(data[SIGNAL], data[TARGET])\n"
    "    weekly_ret = pd.Series(net_ret).dropna()\n"
    "    t_ret, p_ret = ttest_1samp(weekly_ret, 0)\n"
    "    annual_ret = net_ret.groupby(net_ret.index.year).apply(lambda r: (1+r).prod()-1)\n\n"
    "    ev = [\n"
    "        ('IC (signal vs target) > 0.05',      ic,              0.05,  ic > 0.05),\n"
    "        ('IC t-test p < 0.05',                p_ic,            0.05,  p_ic < 0.05),\n"
    "        ('Gross Sharpe > 0.5',                mg['sharpe'],    0.5,   mg['sharpe'] > 0.5),\n"
    "        ('Net Sharpe > 0.3',                  m['sharpe'],     0.3,   m['sharpe'] > 0.3),\n"
    "        ('Net Sharpe > 0',                    m['sharpe'],     0,     m['sharpe'] > 0),\n"
    "        ('Weekly ret t-test p < 0.05',        p_ret,           0.05,  p_ret < 0.05),\n"
    "        ('Annual return (net) > 5%',          m['ann_ret'],    0.05,  m['ann_ret'] > 0.05),\n"
    "        ('Max DD > -40%',                     m['max_dd'],    -0.40,  m['max_dd'] > -0.40),\n"
    "        ('Positive years > 55%',              (annual_ret>0).mean(), 0.55, (annual_ret>0).mean() > 0.55),\n"
    "        ('IC sign-stable > 55% of windows',  (ic_roll>0).mean(),    0.55, (ic_roll>0).mean() > 0.55),\n"
    "    ]\n"
    "    score = sum(1 for *_, p in ev if p)\n"
    "    print(f'\\n{\"=\"*65}')\n"
    "    print(f'VERDICT: {label}')\n"
    "    print(f'{\"=\"*65}')\n"
    "    print(f'{\"Evidence\":<42} {\"Value\":>10}  {\"Threshold\"}')\n"
    "    print('-'*65)\n"
    "    for desc, val, thr, passed in ev:\n"
    "        mark = '[+]' if passed else '[-]'\n"
    "        if abs(val) < 2:\n"
    "            print(f'{mark} {desc:<42} {val:>+10.4f}  >{thr}')\n"
    "        else:\n"
    "            print(f'{mark} {desc:<42} {val:>+10.1%}  >{thr:.0%}')\n"
    "    print(f'\\nScore: {score}/{len(ev)}')\n"
    "    if score >= 7:   v = 'STRONG SIGNAL — deployable'\n"
    "    elif score >= 5: v = 'MODERATE SIGNAL — viable with monitoring'\n"
    "    elif score >= 3: v = 'WEAK SIGNAL — research only'\n"
    "    else:            v = 'NO SIGNAL'\n"
    "    print(f'=> {v}\\n')\n"
    "    return score\n\n"
    "s1 = full_verdict(net_ls,  gross_ls, 'EEM Rule L/S')\n"
    "s2 = full_verdict(net_lf,  gross_lf, 'EEM Rule L/flat')\n"
    "s3 = full_verdict(net_sc, gross_sc, 'EEM Magnitude-Scaled L/S')\n"
    "s4 = full_verdict(res_xgb1['net_ret'], res_xgb1['net_ret'], '1-Feature XGB (walk-forward)')\n\n"
    "print('\\n=== FINAL COMPARISON ===')\n"
    "for lbl, net, gross in [\n"
    "    ('EEM Rule L/S',         net_ls,  gross_ls),\n"
    "    ('EEM Rule L/flat',      net_lf,  gross_lf),\n"
    "    ('EEM Magnitude-Scaled', net_sc,  gross_sc),\n"
    "    ('1-Feature XGB',        res_xgb1['net_ret'], res_xgb1['net_ret']),\n"
    "    ('SET Buy & Hold',       bnh,     bnh),\n"
    "]:\n"
    "    m = risk_metrics(net); mg = risk_metrics(gross)\n"
    "    print(f'{lbl:<26} gross={mg[\"sharpe\"]:+.2f}  net={m[\"sharpe\"]:+.2f}  ret={m[\"ann_ret\"]:+.1%}  dd={m[\"max_dd\"]:.0%}')"
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
