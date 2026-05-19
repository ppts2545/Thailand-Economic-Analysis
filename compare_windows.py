"""
compare_windows.py — Compare 3 training windows for SET model.
Tests: 2000→2019, 2010→2019, 2013→2019
Runs only SET_index_ret_w (1w + 4w) to keep it fast.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

# ── Load notebook core (imports, data, features, model functions) ─────────────
import os
os.chdir('notebooks/model')   # notebook uses relative paths from here
exec(open('/tmp/nb_core.py').read())
os.chdir('../..')              # back to project root

# ── Windows to compare ────────────────────────────────────────────────────────
WINDOWS = [
    ('2000→2019  (20yr)', '2019-12-31', '2020-01-01'),
    ('2010→2019  (10yr)', '2019-12-31', '2020-01-01'),
    ('2013→2019  ( 7yr)', '2019-12-31', '2020-01-01'),
]
# For shorter windows, we just restrict the training data inside fit_and_evaluate
# by filtering df to only include rows from window_start onward

WINDOW_STARTS = {
    '2000→2019  (20yr)': '2000-01-01',
    '2010→2019  (10yr)': '2010-01-01',
    '2013→2019  ( 7yr)': '2013-01-01',
}

TARGET     = 'SET_index_ret_w'
N_TRIALS   = 30   # fast
TEST_START = '2020-01-01'
TRAIN_END  = '2019-12-31'

# SET-specific feature pool (no annual macro)
set_features = [f for f in valid_features if '_annual' not in f]

# 3-driver MUST_INCLUDE for SET
MUST_INCLUDE = {
    'SET_index_ret_w': [
        'eem_ret_d_lag1', 'em_outflow_pressure', 'eem_vs_sp500_lag1',
        'sp500_ret_d_lag1', 'nasdaq_ret_d_lag1', 'risk_on_signal',
        'dxy_ret_d_lag1', 'USD_THB_ret_d_lag1', 'dxy_3w_mom',
    ],
}
COST_MAP = {'SET_index_ret_w': 0.0030}
must_inc = [f for f in MUST_INCLUDE.get(TARGET, []) if f in set_features]

print('\n' + '='*78)
print('SET Training Window Comparison  (DirAcc + Sharpe + AnnRet, test 2020–2025)')
print('='*78)
print(f'  {"Window":<22}  {"Train rows":>10}  {"DirAcc 1w":>10}  '
      f'{"DirAcc 4w":>10}  {"Sharpe 1w":>10}  {"Sharpe 4w":>10}  {"AnnRet 4w":>10}')
print('-'*78)

records = []
for label, train_end, test_start in WINDOWS:
    win_start = WINDOW_STARTS[label]
    df_win = df[df.index >= win_start].copy()
    train_rows = df_win[df_win.index <= train_end]
    print(f'\n  Running {label} ({len(train_rows)} train rows)...')

    try:
        # 1-week model
        res1 = fit_and_evaluate(
            TARGET, df_win, set_features,
            train_end, test_start,
            top_n_features=12, n_trials=N_TRIALS,
            horizon_weeks=1, must_include=must_inc,
        )
        da1   = res1[9]['dir_acc_te']
        fake1 = {TARGET: res1}
        bt1   = backtest_strategy(TARGET, label, fake1, df_win,
                                   cost_oneway=COST_MAP[TARGET])
        sh1   = bt1['sharpe'] if bt1 else float('nan')

        # 4-week model
        res4 = fit_and_evaluate(
            TARGET, df_win, set_features,
            train_end, test_start,
            top_n_features=12, n_trials=N_TRIALS,
            horizon_weeks=4, must_include=must_inc,
        )
        da4   = res4[9]['dir_acc_te']
        fake4 = {TARGET: res4}
        bt4   = backtest_strategy(TARGET, label, fake4, df_win,
                                   cost_oneway=COST_MAP[TARGET])
        sh4   = bt4['sharpe']   if bt4 else float('nan')
        ar4   = bt4['annual_return'] if bt4 else float('nan')

        print(f'  {"→ "+label:<22}  {len(train_rows):>10}  '
              f'{da1*100:>9.1f}%  {da4*100:>9.1f}%  '
              f'{sh1:>10.2f}  {sh4:>10.2f}  {ar4*100:>9.1f}%')
        records.append(dict(label=label, train_rows=len(train_rows),
                             da1=da1, da4=da4, sh1=sh1, sh4=sh4, ar4=ar4))
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print('\n' + '='*78)
print('SUMMARY')
print('='*78)
if records:
    best_da = max(records, key=lambda r: r['da1'])
    best_sh = max(records, key=lambda r: r['sh4'])
    print(f'  Best DirAcc 1w : {best_da["label"]}  ({best_da["da1"]*100:.1f}%)')
    print(f'  Best Sharpe 4w : {best_sh["label"]}  ({best_sh["sh4"]:.2f})')
print('='*78)
