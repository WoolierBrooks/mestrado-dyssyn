# Command Log

- `rg -n "Dirichlet|dirichlet|MN|multinomial|beta" docs README.md project models experiments -S`
- `rg -n "dirichlet|Dirichlet|multinomial|MN" docs README.md project models experiments -S`
- `rg -n "synthetic_distributions\.pkl" -S experiments docs`
- `python - <<'PY'
import json
from pathlib import Path
path=Path('experiments/exp_001/exp_001.ipynb')
nb=json.loads(path.read_text())
# find markdown/code cells mentioning synthetic_distributions
for i,cell in enumerate(nb.get('cells',[])):
    src=''.join(cell.get('source',''))
    if 'synthetic_distributions' in src or 'MoSS' in src and 'pkl' in src:
        if 'synthetic_distributions' in src and 'pkl' in src:
            print('cell', i, 'type', cell.get('cell_type'))
            print(src[:1000])
            print('---')
PY`
- `python - <<'PY'
from pathlib import Path
path=Path('experiments/exp_001/exp_001.ipynb')
lines=path.read_text().splitlines()
for i,line in enumerate(lines,1):
    if 'synthetic_distributions' in line or 'pickle.dump' in line or 'syn_scores' in line:
        if any(k in line for k in ['syn_scores','synthetic_distributions','pickle.dump','MoSS']):
            if 'image/png' in line: continue
            print(i, line[:200])
PY`
