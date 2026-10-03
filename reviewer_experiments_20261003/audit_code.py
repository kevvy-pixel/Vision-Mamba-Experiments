from pathlib import Path
import ast, re, subprocess, sys

ROOT = Path(__file__).resolve().parent
errors = []
e2e = ROOT / 'code/temporal_mamba/train_end_to_end_densenet_mamba.py'
text = e2e.read_text(encoding='utf-8')
if not re.search(r'--batch-size"?,?\s*type=int,\s*default=4', text): errors.append('DenseNet E2E default batch-size is not 4')
if 'stratified_split(samples, args.split_seed)' not in text: errors.append('DenseNet E2E does not use split_seed')
if '--batch-size 4' not in (ROOT/'run_phase1.ps1').read_text(encoding='utf-8'): errors.append('run_phase1.ps1 does not launch E2E with batch-size 4')
for p in (ROOT/'code').rglob('*.py'):
    try: ast.parse(p.read_text(encoding='utf-8'))
    except Exception as exc: errors.append(f'Python syntax: {p}: {exc}')
for p in [ROOT/'run_phase1.ps1', ROOT/'run_phase2.ps1']:
    t=p.read_text(encoding='utf-8')
    if re.search(r"[A-Z]:\\[^'\"]*E:\\|E:\\", t): errors.append(f'Non-portable E: path in {p}')
    if '$Bundle' in t and p.name == 'run_phase1.ps1': errors.append('run_phase1.ps1 still references $Bundle')
print('AUDIT_PASS' if not errors else 'AUDIT_FAIL')
for e in errors: print(' -', e)
sys.exit(1 if errors else 0)
