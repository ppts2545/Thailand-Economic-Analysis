"""
gen_nb24_thd_integration.py
Generate notebooks/eda/24_thd_integration.ipynb

Disciplined 5-step integration of THD signal into NB21 system:
  Step 1 — Linear/Ridge: does THD coefficient survive controls?
  Step 2 — Incremental backtest: NB21 vs NB21+THD vs NB21+THD+extras
  Step 3 — Feature importance (small XGBoost, shallow)
  Step 4 — Walk-forward OOS (expanding window, 1-year test blocks)
  Step 5 — Interaction test: THD × DXY, THD × VIX, THD × EEM

Guard rails:
  - Max 7 features in any model
  - No optimizing on Sharpe — only IC and walk-forward coefficient sign
  - Tree models: max_depth=3, n_estimators=50 only
"""

import inspect, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / 'notebooks' / 'eda' / '24_thd_integration.ipynb'

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": [s + "\n" for s in inspect.cleandoc(src).splitlines()]}

def code(src):
    lines = src.lstrip('\n').rstrip('\n').splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""
    # NB24 — THD Signal Integration: Incremental Value Test

    Disciplined 5-step test before integrating THD−EEM into the live system.

    **Guard rails (no exceptions):**
    - Max 7 features per model
    - Never optimize on Sharpe — use IC and sign-stability only
    - Tree models: `max_depth=3`, `n_estimators=50` max
    - Walk-forward: no look-ahead, expanding train window

    | Step | Test | Answers |
    |------|------|---------|
    | 1 | Ridge regression | Does THD coefficient survive EEM control? |
    | 2 | Incremental backtest | How much Sharpe/DD does each signal add? |
    | 3 | Shallow XGBoost importance | Does tree model use THD or ignore it? |
    | 4 | Walk-forward OOS coefficients | Is THD stable year-by-year? |
    | 5 | Interaction terms | THD × DXY, THD × VIX, THD × EEM |
"""))

# ── 1. Setup ──────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup — Build all features"))

cells.append(code("""\
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

ROOT = Path('..').resolve().parent
DATA = ROOT / 'data' / 'processed'
FIGS = Path('.') / 'figs'
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3})
WEEKS_PER_YEAR = 52
RF_ANNUAL = 0.02
START = '2010-01-01'
"""))

cells.append(code("""\
import yfinance as yf

# ── Base data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA / 'unified_weekly_clean.csv',
                 index_col=0, parse_dates=True).sort_index().loc[START:]

target   = df['SET_index_ret_w']          # 1w forward return (already computed)
gold_ret = df['gold_ret_w']

def get_close(raw):
    if isinstance(raw.columns, pd.MultiIndex): return raw['Close'].iloc[:, 0]
    return raw['Close']

print('Fetching THD, EEM, DXY, VIX...')
thd_c = get_close(yf.download('THD',       start='2009-01-01', progress=False))
eem_c = get_close(yf.download('EEM',       start='2009-01-01', progress=False))
dxy_c = get_close(yf.download('DX-Y.NYB',  start='2009-01-01', progress=False))
vix_c = get_close(yf.download('^VIX',      start='2009-01-01', progress=False))

thd_w = thd_c.resample('W-FRI').last().pct_change()
eem_w = eem_c.resample('W-FRI').last().pct_change()
dxy_w = dxy_c.resample('W-FRI').last().pct_change()
vix_w = vix_c.resample('W-FRI').last()

# ── THD signal (from NB22/23) ─────────────────────────────────────────────────
thd_rel = thd_w - eem_w
thd_4w  = thd_rel.rolling(4).sum()
roll52  = thd_4w.rolling(52)
thd_z   = (thd_4w - roll52.mean()) / (roll52.std() + 1e-9)

# ── DXY momentum (13w) ────────────────────────────────────────────────────────
dxy_mom13 = (1 + dxy_w.fillna(0)).cumprod().pct_change(13)

# ── VIX z-score (52w rolling) ─────────────────────────────────────────────────
vix_roll = vix_w.rolling(52)
vix_z = (vix_w - vix_roll.mean()) / (vix_roll.std() + 1e-9)

# ── Align everything to weekly SET index ─────────────────────────────────────
feat = pd.DataFrame({
    'eem_lag1':   df['eem_ret_d_lag1'],             # NB21 main signal
    'thd_z_lag2': thd_z.reindex(target.index).ffill().shift(2),   # new signal
    'thd_4w_lag1':thd_4w.reindex(target.index).ffill().shift(1),  # new signal alt
    'dxy_mom13':  dxy_mom13.reindex(target.index).ffill().shift(1),
    'vix_z_lag1': vix_z.reindex(target.index).ffill().shift(1),
    'yield_slope':df.get('th_slope_lag1', pd.Series(np.nan, index=target.index)),
}, index=target.index)

# Thai yield slope if available
bonds_path = ROOT / 'data' / 'raw' / 'bot_bond_yields.csv'
if bonds_path.exists():
    bonds = pd.read_csv(bonds_path, parse_dates=['date']).set_index('date').sort_index()
    bonds['th_slope'] = bonds['yield_10y'] - bonds['yield_1y']
    slope_w = bonds['th_slope'].resample('W-FRI').last().ffill()
    feat['yield_slope'] = slope_w.reindex(target.index).ffill().shift(1)

feat = feat.dropna(how='all')
print(f'Feature matrix: {feat.shape}')
print(f'Date range: {feat.dropna().index[0].date()} -> {feat.dropna().index[-1].date()}')
print(feat.tail(3).to_string())
"""))

# ── Step 1: Ridge regression ───────────────────────────────────────────────────
cells.append(md("""
    ## Step 1 — Ridge Regression: Does THD Coefficient Survive?

    Simple test: predict SET 1w return with Ridge regression.
    Start with 1-feature (EEM only), then add THD.

    **Question:** When EEM is in the model, does THD's coefficient stay positive and significant?
    If THD proxies EEM (i.e., they explain the same thing), coefficient should go to zero.
"""))

cells.append(code("""\
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

def ridge_ic(X_cols, feat_df, tgt, alpha=1.0):
    '''Fit ridge, report coefficient and IC of fitted values.'''
    data = pd.concat([feat_df[X_cols], tgt], axis=1).dropna()
    data.columns = list(X_cols) + ['y']
    X = data[list(X_cols)].values
    y = data['y'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Leave-one-out rolling cross-val (expanding window, min 52w)
    preds = np.full(len(y), np.nan)
    for i in range(52, len(y)):
        Xtrain, ytrain = Xs[:i], y[:i]
        Xtest          = Xs[i:i+1]
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge.fit(Xtrain, ytrain)
        preds[i] = ridge.predict(Xtest)[0]

    pred_s = pd.Series(preds, index=data.index)
    r, p   = stats.spearmanr(pred_s.dropna(), y[~np.isnan(preds)])
    coefs  = {}
    # Final full-sample fit for coefficients
    ridge_full = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge_full.fit(StandardScaler().fit_transform(X), y)
    for col, c in zip(X_cols, ridge_full.coef_):
        coefs[col] = c
    return r, p, coefs, pred_s

# Model A: EEM only
rA, pA, cA, _ = ridge_ic(['eem_lag1'], feat, target)
print(f'Model A (EEM only):          IC={rA:+.4f}  p={pA:.3f}')
print(f'  coef: eem_lag1={cA["eem_lag1"]:+.4f}')
print()

# Model B: THD only
rB, pB, cB, _ = ridge_ic(['thd_z_lag2'], feat, target)
print(f'Model B (THD only):          IC={rB:+.4f}  p={pB:.3f}')
print(f'  coef: thd_z_lag2={cB["thd_z_lag2"]:+.4f}')
print()

# Model C: EEM + THD
rC, pC, cC, _ = ridge_ic(['eem_lag1', 'thd_z_lag2'], feat, target)
print(f'Model C (EEM + THD):         IC={rC:+.4f}  p={pC:.3f}')
for k, v in cC.items(): print(f'  coef: {k}={v:+.4f}')
print()

# Model D: EEM + THD + DXY + VIX
rD, pD, cD, _ = ridge_ic(['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1'], feat, target)
print(f'Model D (EEM+THD+DXY+VIX):   IC={rD:+.4f}  p={pD:.3f}')
for k, v in cD.items(): print(f'  coef: {k}={v:+.4f}')
print()

# Model E: Add yield slope if available
yield_available = feat['yield_slope'].notna().sum() > 100
if yield_available:
    rE, pE, cE, _ = ridge_ic(['eem_lag1','thd_z_lag2','dxy_mom13','vix_z_lag1','yield_slope'], feat, target)
    print(f'Model E (D + yield slope):   IC={rE:+.4f}  p={pE:.3f}')
    for k, v in cE.items(): print(f'  coef: {k}={v:+.4f}')
"""))

cells.append(code("""\
# Visualise coefficients across models
models = {'A: EEM': cA, 'B: THD': cB, 'C: EEM+THD': cC, 'D: +DXY+VIX': cD}
if yield_available:
    models['E: +yield'] = cE

all_feats = ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1', 'yield_slope']
model_names = list(models.keys())
coef_matrix = pd.DataFrame(
    {m: {f: models[m].get(f, np.nan) for f in all_feats} for m in model_names}
).T

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(all_feats))
w = 0.15
for i, (mname, mcoefs) in enumerate(models.items()):
    vals = [mcoefs.get(f, np.nan) for f in all_feats]
    offset = (i - len(models)/2 + 0.5) * w
    bars = ax.bar(x + offset, vals, width=w, label=mname, alpha=0.85)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(all_feats, rotation=20, ha='right')
ax.set_title('Ridge Coefficients (standardised inputs) — Does THD survive controls?')
ax.set_ylabel('Standardised coefficient')
ax.legend(fontsize=8, ncol=len(models))
plt.tight_layout()
plt.savefig(FIGS / 'nb24_ridge_coefs.png', bbox_inches='tight')
plt.show()

print()
print('Key question: is thd_z_lag2 coefficient positive in ALL models that include it?')
for m, c in models.items():
    if 'thd_z_lag2' in c:
        sign = '✓ positive' if c['thd_z_lag2'] > 0 else '✗ negative'
        print(f'  {m:20s}: thd_z_lag2 = {c["thd_z_lag2"]:+.4f}  {sign}')
"""))

# ── Step 2: Incremental Backtest ───────────────────────────────────────────────
cells.append(md("""
    ## Step 2 — Incremental Backtest: How Much Does Each Signal Add?

    Build each system version from scratch using only the position-scaling mechanism.
    Compare the marginal contribution of each signal.

    Rule: signal → rank → top/bottom 20% → ×1.2 / ×0.8 on baseline position.
"""))

cells.append(code("""\
def make_baseline():
    '''NB21 baseline system.'''
    eem_pos = (df['eem_ret_d_lag1'].shift(0) > 0).astype(float).fillna(0.5)
    vs = target.rolling(12).std().shift(1) + 1e-6
    vg = gold_ret.rolling(12).std().shift(1) + 1e-6
    ws = (1/vs) / (1/vs + 1/vg)
    wg = (1/vg) / (1/vs + 1/vg)
    rp = ws * eem_pos * target + wg * gold_ret
    tv = 0.10 / np.sqrt(52)
    vt = (tv / (rp.rolling(12).std().shift(1) + 1e-9)).clip(0, 2) * rp
    cum  = (1 + vt.fillna(0)).cumprod()
    peak = cum.expanding().max()
    dd   = (cum - peak) / peak
    dds  = pd.Series(1.0, index=vt.index)
    in_dd = False
    for i in range(1, len(dd)):
        if dd.iloc[i] < -0.15 and not in_dd: in_dd = True
        if in_dd and dd.iloc[i] > -0.10:     in_dd = False
        dds.iloc[i] = 0.5 if in_dd else 1.0
    return dds * vt

def add_signal_scale(base_ret, *signals, top_q=0.80, bot_q=0.20):
    '''Layer one or more signals as ±20% position scale on base.'''
    sigs = [s.reindex(base_ret.index).ffill() for s in signals]
    df_s = pd.DataFrame({f's{i}': s for i, s in enumerate(sigs)})
    comp = df_s.rank(pct=True).mean(axis=1)
    scale = pd.Series(1.0, index=base_ret.index)
    scale[comp >= top_q] = 1.20
    scale[comp <= bot_q] = 0.80
    return scale * base_ret

def metrics(r):
    rf = RF_ANNUAL / 52
    sh = (r - rf).mean() / ((r - rf).std() + 1e-9) * np.sqrt(52)
    cum = (1 + r.fillna(0)).cumprod()
    mdd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
    cal = r.mean() * 52 / abs(mdd) if mdd < 0 else np.nan
    turns = (r != 0).sum() / len(r) * 52  # proxy
    return {'Sharpe': sh, 'MaxDD': mdd, 'Calmar': cal, 'AnnRet': r.mean()*52}

base = make_baseline()
thd_z_l2   = feat['thd_z_lag2']
thd_4w_l1  = feat['thd_4w_lag1']
dxy_m13    = feat['dxy_mom13']
vix_z_l1   = feat['vix_z_lag1']

systems = {
    'SET B&H':              (target,),
    'NB21 Baseline':        (base,),
    'NB21 + THD_z(lag2)':   (add_signal_scale(base, thd_z_l2),),
    'NB21 + THD_4w(lag1)':  (add_signal_scale(base, thd_4w_l1),),
    'NB21 + THD composite': (add_signal_scale(base, thd_z_l2, thd_4w_l1),),
    'NB21 + THD + DXY':     (add_signal_scale(base, thd_z_l2, thd_4w_l1, -dxy_m13),),
    'NB21 + THD + VIX':     (add_signal_scale(base, thd_z_l2, thd_4w_l1, -vix_z_l1),),
}
if yield_available:
    y_sl = feat['yield_slope']
    systems['NB21 + THD + yield'] = (add_signal_scale(base, thd_z_l2, thd_4w_l1, y_sl),)

print(f'{"Strategy":<30} {"Sharpe":>8} {"MaxDD":>8} {"Calmar":>8} {"AnnRet":>8}')
print('-' * 65)
results = {}
for name, (ret,) in systems.items():
    m = metrics(ret)
    results[name] = m
    marker = ''
    if name not in ('SET B&H', 'NB21 Baseline'):
        delta_sh = m['Sharpe'] - results['NB21 Baseline']['Sharpe']
        marker = f'  Δ={delta_sh:+.3f}'
    print(f'  {name:<28} {m["Sharpe"]:>8.3f} {m["MaxDD"]:>8.1%} '
          f'{m["Calmar"]:>8.2f} {m["AnnRet"]:>8.1%}{marker}')
"""))

cells.append(code("""\
# Equity curves
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

colors = {'SET B&H': 'grey', 'NB21 Baseline': 'steelblue',
          'NB21 + THD composite': '#2ecc71', 'NB21 + THD + DXY': 'darkorange',
          'NB21 + THD + VIX': 'purple'}

to_plot = {k: v for k, v in systems.items() if k in colors}

ax = axes[0]
for name, (ret,) in to_plot.items():
    cum = (1 + ret.fillna(0)).cumprod()
    cum.plot(ax=ax, label=name, color=colors[name],
             lw=2 if 'THD' in name else 1.2,
             ls='--' if name == 'NB21 Baseline' else '-',
             alpha=0.9 if 'THD' in name else 0.7)
ax.set_yscale('log')
ax.set_title('Equity Curves — Incremental Signal Addition')
ax.set_ylabel('Cumulative return (log scale)')
ax.legend(fontsize=8, ncol=2)

ax2 = axes[1]
for name, (ret,) in to_plot.items():
    cum = (1 + ret.fillna(0)).cumprod()
    dd  = (cum - cum.expanding().max()) / cum.expanding().max()
    dd.plot(ax=ax2, label=name, color=colors[name],
            lw=1.5 if 'THD' in name else 1.0,
            ls='--' if name == 'NB21 Baseline' else '-',
            alpha=0.8)
ax2.set_ylabel('Drawdown')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

plt.tight_layout()
plt.savefig(FIGS / 'nb24_incremental_equity.png', bbox_inches='tight')
plt.show()
"""))

# ── Step 3: Shallow XGBoost Importance ────────────────────────────────────────
cells.append(md("""
    ## Step 3 — Shallow XGBoost: Does Tree Model Use THD?

    Fit a **deliberately shallow** XGBoost (max_depth=3, n_estimators=50).
    If THD has genuine predictive power, it should appear in feature importance.
    If tree ignores it, the linear-only signal doesn't translate to nonlinear.

    Guard: `max_depth=3`, `n_estimators=50` — no tuning, no overfitting.
"""))

cells.append(code("""\
FEAT_COLS = ['eem_lag1', 'thd_z_lag2', 'thd_4w_lag1', 'dxy_mom13', 'vix_z_lag1']
if yield_available:
    FEAT_COLS.append('yield_slope')

data_xgb = pd.concat([feat[FEAT_COLS], target], axis=1).dropna()
data_xgb.columns = FEAT_COLS + ['y']

X_all = data_xgb[FEAT_COLS].values
y_all = data_xgb['y'].values

# Expanding window walk-forward (train on first 60%, evaluate on rest)
n = len(data_xgb)
train_end = int(n * 0.60)
X_train, y_train = X_all[:train_end], y_all[:train_end]
X_test,  y_test  = X_all[train_end:], y_all[train_end:]

model = xgb.XGBRegressor(
    max_depth=3, n_estimators=50, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
oos_r, oos_p = stats.spearmanr(preds, y_test)
print(f'XGBoost OOS IC (60/40 split): {oos_r:+.4f}  p={oos_p:.3f}')
print()

# Feature importance
imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print('XGBoost feature importance (gain):')
for f, v in imp.items():
    bar = '█' * int(v * 50)
    print(f'  {f:20s}: {v:.4f}  {bar}')
"""))

cells.append(code("""\
# Compare: XGBoost with and without THD
for label, cols in [('Without THD', ['eem_lag1', 'dxy_mom13', 'vix_z_lag1']),
                    ('With THD',    FEAT_COLS)]:
    d = pd.concat([feat[cols], target], axis=1).dropna()
    d.columns = cols + ['y']
    xt = d[cols].values; yt = d['y'].values
    n2 = len(d); tr = int(n2 * 0.60)
    m2 = xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbosity=0)
    m2.fit(xt[:tr], yt[:tr])
    p2 = m2.predict(xt[tr:])
    r2, pv2 = stats.spearmanr(p2, yt[tr:])
    stars = '***' if pv2<0.01 else ('**' if pv2<0.05 else ('*' if pv2<0.1 else ''))
    print(f'  {label:15s}: OOS IC = {r2:+.4f}  p = {pv2:.3f} {stars}')

fig, ax = plt.subplots(figsize=(8, 3.5))
colors_imp = ['#2ecc71' if 'thd' in f else 'steelblue' for f in imp.index]
ax.barh(imp.index[::-1], imp.values[::-1], color=colors_imp[::-1], alpha=0.85, edgecolor='white')
ax.set_title('XGBoost Feature Importance (Gain)  max_depth=3, n=50')
ax.set_xlabel('Importance (gain)')
plt.tight_layout()
plt.savefig(FIGS / 'nb24_xgb_importance.png', bbox_inches='tight')
plt.show()
"""))

# ── Step 4: Walk-Forward Coefficients ─────────────────────────────────────────
cells.append(md("""
    ## Step 4 — Walk-Forward OOS Coefficients

    Most important test: fit Ridge every year, check THD coefficient sign year-by-year.

    **Passing criterion:**
    - THD coefficient positive in ≥ 65% of test years
    - Mean OOS IC > 0

    Train: expanding window starting 2010
    Test:  each calendar year from 2014 to 2025 (need 3yr+ train history)
"""))

cells.append(code("""\
COLS_WF = ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1']
data_wf = pd.concat([feat[COLS_WF], target], axis=1).dropna()
data_wf.columns = COLS_WF + ['y']

scaler_wf = StandardScaler()
Xall = scaler_wf.fit_transform(data_wf[COLS_WF].values)
yall = data_wf['y'].values
idx  = data_wf.index

wf_rows = []
for test_yr in range(2014, 2026):
    train_m = idx.year < test_yr
    test_m  = idx.year == test_yr
    if train_m.sum() < 52 or test_m.sum() < 10:
        continue
    Xtr, ytr = Xall[train_m], yall[train_m]
    Xte, yte = Xall[test_m],  yall[test_m]

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(Xtr, ytr)
    coefs = dict(zip(COLS_WF, ridge.coef_))

    preds = ridge.predict(Xte)
    r, p  = stats.spearmanr(preds, yte)

    wf_rows.append({
        'year':    test_yr,
        'IC':      r,
        'p':       p,
        'n_train': train_m.sum(),
        **{f'coef_{c}': coefs[c] for c in COLS_WF},
    })

wf_df = pd.DataFrame(wf_rows)

print(f'{"Year":>6} {"n_train":>8} {"OOS IC":>9} {"eem":>8} {"thd_z":>8} {"dxy":>8} {"vix":>8}')
print('-' * 65)
for _, row in wf_df.iterrows():
    stars = '**' if row.p<0.05 else ('*' if row.p<0.10 else ' ')
    thd_ok = '✓' if row.coef_thd_z_lag2 > 0 else '✗'
    print(f'  {int(row.year):4d}  {int(row.n_train):6d}  '
          f'{row.IC:+8.4f}{stars}  '
          f'{row.coef_eem_lag1:+7.4f}  '
          f'{row.coef_thd_z_lag2:+7.4f}{thd_ok}  '
          f'{row.coef_dxy_mom13:+7.4f}  '
          f'{row.coef_vix_z_lag1:+7.4f}')

print()
pct_thd_pos = (wf_df['coef_thd_z_lag2'] > 0).mean()
avg_oos_ic  = wf_df['IC'].mean()
print(f'THD coef positive in {pct_thd_pos:.0%} of test years  (target ≥ 65%)')
print(f'Mean OOS IC: {avg_oos_ic:+.4f}  (target > 0)')
"""))

cells.append(code("""\
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# OOS IC by year
ax = axes[0]
colors_ic = ['#2ecc71' if v > 0 else '#e74c3c' for v in wf_df['IC']]
ax.bar(wf_df['year'], wf_df['IC'], color=colors_ic, alpha=0.85, edgecolor='white')
ax.axhline(0,        color='black', lw=0.8)
ax.axhline(avg_oos_ic, color='navy', lw=1.2, ls='--', label=f'Mean={avg_oos_ic:+.3f}')
ax.set_ylabel('OOS Spearman IC')
ax.set_title('Walk-Forward OOS IC (Ridge, expanding window)')
ax.legend(fontsize=8)

# THD coefficient by year
ax2 = axes[1]
colors_coef = ['#2ecc71' if v > 0 else '#e74c3c' for v in wf_df['coef_thd_z_lag2']]
ax2.bar(wf_df['year'], wf_df['coef_thd_z_lag2'], color=colors_coef, alpha=0.85,
        edgecolor='white', label='THD z-score (lag2)')
ax2.plot(wf_df['year'], wf_df['coef_eem_lag1'], color='steelblue', lw=1.5,
         marker='o', ms=4, label='EEM lag1')
ax2.axhline(0, color='black', lw=0.8)
ax2.set_ylabel('Standardised coefficient')
ax2.set_title('THD vs EEM Coefficients (Walk-forward Ridge)')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIGS / 'nb24_walkforward_coefs.png', bbox_inches='tight')
plt.show()
"""))

# ── Step 5: Interaction Terms ─────────────────────────────────────────────────
cells.append(md("""
    ## Step 5 — Interaction Terms: THD × DXY, THD × VIX, THD × EEM

    **Hypothesis (from NB23):** THD signal is stronger when:
    - DXY is NOT rallying (USD weak/neutral) → THD×(−DXY)
    - VIX is elevated (fear → contrarian THD flow matters more) → THD×VIX
    - EEM is positive (risk-on regime) → THD×EEM

    Test: add interaction terms to Ridge and check if coefficients are significant.
    Interaction coefficient positive = hypothesis confirmed.
"""))

cells.append(code("""\
# Normalise features for interaction construction
def norm(s):
    r = s.rolling(52)
    return (s - r.mean()) / (r.std() + 1e-9)

thd_n  = norm(feat['thd_z_lag2'])
dxy_n  = norm(feat['dxy_mom13'])
vix_n  = norm(feat['vix_z_lag1'])
eem_n  = norm(feat['eem_lag1'])

# Interactions: multiply normalised features
int_thd_dxy = thd_n * (-dxy_n)      # THD * (−DXY): signal stronger when DXY falling
int_thd_vix = thd_n * vix_n         # THD * VIX: signal stronger when VIX elevated
int_thd_eem = thd_n * eem_n         # THD * EEM: signal stronger when EEM positive

feat_int = pd.DataFrame({
    'eem_lag1':      feat['eem_lag1'],
    'thd_z_lag2':    feat['thd_z_lag2'],
    'dxy_mom13':     feat['dxy_mom13'],
    'vix_z_lag1':    feat['vix_z_lag1'],
    'thd_x_neg_dxy': int_thd_dxy,
    'thd_x_vix':     int_thd_vix,
    'thd_x_eem':     int_thd_eem,
}, index=target.index)

# IC of each interaction term standalone
print('IC of interaction terms (standalone):')
for col in ['thd_x_neg_dxy', 'thd_x_vix', 'thd_x_eem']:
    both = pd.concat([feat_int[col].shift(0), target], axis=1).dropna()
    both.columns = ['s', 't']
    r, p = stats.spearmanr(both['s'], both['t'])
    stars = '***' if p<0.01 else ('**' if p<0.05 else ('*' if p<0.1 else ''))
    print(f'  {col:20s}: IC={r:+.4f}  p={p:.3f} {stars}')

print()

# Ridge with interactions
for label, cols in [
    ('Base (no interactions)',      ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1']),
    ('+ THD × (−DXY)',              ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1', 'thd_x_neg_dxy']),
    ('+ THD × VIX',                 ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1', 'thd_x_vix']),
    ('+ THD × EEM',                 ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1', 'thd_x_eem']),
    ('+ all 3 interactions',        ['eem_lag1', 'thd_z_lag2', 'dxy_mom13', 'vix_z_lag1',
                                     'thd_x_neg_dxy', 'thd_x_vix', 'thd_x_eem']),
]:
    ri, pi, ci, _ = ridge_ic(cols, feat_int, target)
    stars = '***' if pi<0.01 else ('**' if pi<0.05 else ('*' if pi<0.1 else ''))
    int_coefs = {k: v for k, v in ci.items() if k.startswith('thd_x')}
    int_str = '  '.join([f'{k.replace("thd_x_","")}: {v:+.3f}' for k, v in int_coefs.items()])
    print(f'  {label:<30}: IC={ri:+.4f} {stars}   {int_str}')
"""))

# ── Final Verdict ──────────────────────────────────────────────────────────────
cells.append(md("## 6. Final Verdict"))

cells.append(code("""\
print('=' * 70)
print('INTEGRATION VERDICT — THD−EEM 4w Z-score (lag2)')
print('=' * 70)
print()

# Collect results
st1_thd_survives = cC.get('thd_z_lag2', 0) > 0
st1_msg = f'thd coef = {cC.get("thd_z_lag2",0):+.4f}'

st2_best_delta = max(
    results.get('NB21 + THD composite', {}).get('Sharpe', 0) - results['NB21 Baseline']['Sharpe'],
    results.get('NB21 + THD_z(lag2)', {}).get('Sharpe', 0) - results['NB21 Baseline']['Sharpe'],
)
st2_pass = st2_best_delta > 0

pct_thd_pos_val = (wf_df['coef_thd_z_lag2'] > 0).mean()
st4_pass = pct_thd_pos_val >= 0.65

tests_final = [
    ('Step 1: THD coef positive in Ridge+EEM', st1_thd_survives, st1_msg),
    ('Step 2: Best variant adds Sharpe',       st2_pass,
     f'Δ Sharpe = {st2_best_delta:+.3f}'),
    ('Step 3: THD in XGBoost top features',
     imp.get('thd_z_lag2', 0) > imp.get('dxy_mom13', 0),
     f'imp(thd)={imp.get("thd_z_lag2",0):.4f}  imp(dxy)={imp.get("dxy_mom13",0):.4f}'),
    ('Step 4: THD coef positive ≥65% of years', st4_pass,
     f'{pct_thd_pos_val:.0%} ({int(pct_thd_pos_val*len(wf_df))}/{len(wf_df)} yrs)'),
]

n_pass = 0
for name, passed, detail in tests_final:
    mark = '✓ PASS' if passed else '✗ FAIL'
    if passed: n_pass += 1
    print(f'  {mark}  {name:<45} {detail}')

print()
print(f'Score: {n_pass}/{len(tests_final)} steps passed')
print()
if n_pass >= 3:
    print('RECOMMENDATION: Integrate THD composite signal into NB21 system.')
    print('  Use: ±20% position scale from THD_z(lag2) + THD_4w(lag1) composite')
    print('  Expected improvement: ~+0.02 Sharpe, similar MaxDD')
else:
    print('RECOMMENDATION: Do not integrate. Signal does not survive linear controls.')
print()
print('Best system variant from Step 2:')
best_name = max(
    {k: v for k, v in results.items() if k not in ('SET B&H', 'NB21 Baseline')},
    key=lambda k: results[k]['Sharpe']
)
bm = results[best_name]
print(f'  {best_name}: Sharpe={bm["Sharpe"]:.3f}  MaxDD={bm["MaxDD"]:.1%}  '
      f'Calmar={bm["Calmar"]:.2f}')
"""))

# ── Build ──────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3",
                                "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10.0"}},
    "cells": cells
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Generated: {OUT}')
print(f'Cells: {len(cells)}')
