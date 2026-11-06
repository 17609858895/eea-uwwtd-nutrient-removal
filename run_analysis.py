"""Select an analysis stage explicitly; --help never reads data or fits a model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'prepare': [('00_extract_gpkg.py',), ('01_build_analysis_table.py',)],
    'support': [('04b_causal_stable.py',)],
    'main': [('11_final_estimates.py', 'lgbm', '10')],
    'decomposition': [('18_ef_decomposition.py',)],
    'influent': [('08_sensitivity_influent.py',)],
    'reporting': [('09_mnar_tipping_point.py',)],
    'placebo': [('10_placebo_support.py',)],
    'confounding': [('12_ovb_sensitivity.py',)],
    'scenario': [('13_final_policy.py', '200')],
    'learner': [('11_final_estimates.py', 'sl', '3'), ('11b_learner_sensitivity.py',)],
    'simulation': [('05_semisynthetic.py',)],
    'benchmark': [('06_benchmark20.py', p, cv) for p in ['N', 'P'] for cv in ['random5', 'LOCO']],
    'longitudinal': [('19_panel_check.py',)],
}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=list(STAGES))
    ap.add_argument('--input-zip', type=Path, help='Official 2022 UWWTD ZIP, read without copying it')
    ap.add_argument('--dry-run', action='store_true', help='Print commands without importing analyses or fitting')
    args = ap.parse_args()
    env = dict(os.environ)
    if args.input_zip:
        if not args.dry_run and not args.input_zip.is_file():
            ap.error('--input-zip must point to an existing official ZIP')
        env['UWWTD_SOURCE_ZIP'] = str(args.input_zip.resolve())
    for row in STAGES[args.stage]:
        cmd = [sys.executable, str(ROOT / 'analysis/scripts' / row[0]), *row[1:]]
        print(' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

if __name__ == '__main__':
    main()
