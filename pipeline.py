"""
pipeline.py — Full data pipeline for Thailand Economic Analysis

รัน script นี้เพื่ออัปเดตข้อมูลทั้งหมดตั้งแต่ต้นจนจบ

Steps:
  1. fetch_raw_data.py        → data/raw/         (market + FRED + World Bank)
  2. fetch_sector_data.py     → sector_weekly.csv
  3. fetch_sector_nlp.py      → sector_sentiment_weekly.csv
  4. fetch_set_stocks.py      → set_stocks_weekly.csv
  5. src/preprocess_weekly.py → unified_weekly.csv
  6. NB07 (nbconvert)         → unified_weekly_clean.csv
  7. fetch_bot_bonds.py       → bot_bond_yields.csv  (Thai yield curve)
  8. fetch_set_flow.py        → set_fund_flow.csv    (foreign/inst/retail net flow)
  9. fetch_tfex_pcr.py        → tfex_pcr.csv         (SET50 put/call ratio)

Usage:
  python3 pipeline.py            # run all steps
  python3 pipeline.py --fetch    # step 1-4 only (data fetching)
  python3 pipeline.py --process  # step 5-6 only (preprocessing)
  python3 pipeline.py --alt      # step 7-9 only (alternative signals)
  python3 pipeline.py --step 3   # run single step
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = str(ROOT / '.venv' / 'bin' / 'python3')

# ── Pipeline steps ────────────────────────────────────────────────────────────
STEPS = [
    {
        'id':   1,
        'name': 'Fetch raw market/macro data',
        'cmd':  [PYTHON, 'scripts/fetch_raw_data.py'],
        'out':  'data/raw/SET_index_market_signals.csv',
    },
    {
        'id':   2,
        'name': 'Fetch sector weekly returns',
        'cmd':  [PYTHON, 'scripts/fetch_sector_data.py'],
        'out':  'data/processed/sector_weekly.csv',
    },
    {
        'id':   3,
        'name': 'Fetch sector NLP sentiment',
        'cmd':  [PYTHON, 'scripts/fetch_sector_nlp.py'],
        'out':  'data/processed/sector_sentiment_weekly.csv',
    },
    {
        'id':   4,
        'name': 'Fetch SET stock prices (54 stocks)',
        'cmd':  [PYTHON, 'scripts/fetch_set_stocks.py'],
        'out':  'data/processed/set_stocks_weekly.csv',
    },
    {
        'id':   5,
        'name': 'Preprocess weekly features',
        'cmd':  [PYTHON, 'src/preprocess_weekly.py'],
        'out':  'data/processed/unified_weekly.csv',
    },
    {
        'id':   6,
        'name': 'Leakage audit → unified_weekly_clean.csv',
        'cmd':  [
            PYTHON, '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute', '--inplace',
            'notebooks/eda/07_leakage_audit_pipeline.ipynb',
            '--ExecutePreprocessor.timeout=300',
        ],
        'out':  'data/processed/unified_weekly_clean.csv',
    },
    {
        'id':   7,
        'name': 'Fetch Thai bond yields (BOT)',
        'cmd':  [PYTHON, 'scripts/fetch_bot_bonds.py'],
        'out':  'data/raw/bot_bond_yields.csv',
    },
    {
        'id':   8,
        'name': 'Fetch SET fund flow by investor type',
        'cmd':  [PYTHON, 'scripts/fetch_set_flow.py'],
        'out':  'data/raw/set_fund_flow.csv',
    },
    {
        'id':   9,
        'name': 'Fetch TFEX SET50 put/call ratio',
        'cmd':  [PYTHON, 'scripts/fetch_tfex_pcr.py'],
        'out':  'data/raw/tfex_pcr.csv',
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────
def run_step(step: dict, dry_run: bool = False) -> bool:
    print(f'\n{"="*60}')
    print(f'Step {step["id"]}: {step["name"]}')
    print(f'{"="*60}')

    if dry_run:
        print(f'  [DRY RUN] Would run: {" ".join(step["cmd"])}')
        return True

    t0 = time.time()
    result = subprocess.run(
        step['cmd'],
        cwd=ROOT,
        capture_output=False,   # show output live
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'\n✗ Step {step["id"]} FAILED (exit code {result.returncode})')
        return False

    # Verify output file exists
    out_path = ROOT / step['out']
    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f'\n✓ Step {step["id"]} done in {elapsed:.0f}s  →  {step["out"]} ({size_kb:.0f} KB)')
    else:
        print(f'\n⚠ Step {step["id"]} finished but output not found: {step["out"]}')

    return True


def main():
    parser = argparse.ArgumentParser(description='Thailand alpha project pipeline')
    parser.add_argument('--fetch',   action='store_true', help='Run steps 1-4 (fetch only)')
    parser.add_argument('--process', action='store_true', help='Run steps 5-6 (preprocess only)')
    parser.add_argument('--alt',     action='store_true', help='Run steps 7-9 (alt signals: bonds/flow/pcr)')
    parser.add_argument('--step',    type=int,            help='Run single step by number (1-9)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run, no execution')
    args = parser.parse_args()

    # Select steps to run
    if args.step:
        steps = [s for s in STEPS if s['id'] == args.step]
        if not steps:
            print(f'Error: step {args.step} not found (valid: 1-9)')
            sys.exit(1)
    elif args.fetch:
        steps = [s for s in STEPS if s['id'] <= 4]
    elif args.process:
        steps = [s for s in STEPS if s['id'] >= 5 and s['id'] <= 6]
    elif args.alt:
        steps = [s for s in STEPS if s['id'] >= 7]
    else:
        steps = STEPS

    print('Thailand Economic Analysis — Data Pipeline')
    print(f'Running {len(steps)} step(s): {[s["id"] for s in steps]}')

    t_total = time.time()
    for step in steps:
        ok = run_step(step, dry_run=args.dry_run)
        if not ok:
            print(f'\nPipeline stopped at step {step["id"]}.')
            sys.exit(1)

    elapsed = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'Pipeline complete in {elapsed/60:.1f} min')
    print('Output files ready:')
    for step in steps:
        out = ROOT / step['out']
        status = '✓' if out.exists() else '✗ missing'
        print(f'  [{status}] {step["out"]}')


if __name__ == '__main__':
    main()
