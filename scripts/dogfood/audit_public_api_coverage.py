"""For every top-level styxx module that is re-exported through __init__.py
or registered as a CLI subcommand, count test files that mention it.
"""
import pathlib, re

root = pathlib.Path('styxx')
top_mods = sorted({p.stem if p.is_file() else p.name
                   for p in root.iterdir()
                   if (p.is_file() and p.suffix == '.py' and not p.name.startswith('_'))
                   or (p.is_dir() and not p.name.startswith('_') and p.name not in ('__pycache__', 'fonts', 'recipes', 'centroids'))})
top_mods = [m for m in top_mods if not m.startswith('_')]

test_files = list(pathlib.Path('tests').rglob('*.py')) if pathlib.Path('tests').exists() else []
print(f'Tests files: {len(test_files)}')

cov = {}
for m in top_mods:
    pat = re.compile(
        r'(?:from\s+(?:\.\.?|styxx\.)?' + m + r'\b'
        r'|import\s+styxx\.' + m + r'\b'
        r'|from\s+styxx\s+import\s+[^#\n]*\b' + m + r'\b'
        r'|styxx\.' + m + r'\b)'
    )
    hits = 0
    for p in test_files:
        try:
            t = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if pat.search(t):
            hits += 1
    cov[m] = hits

ZERO = [m for m, n in cov.items() if n == 0]
print(f'\nMODULES WITH ZERO TEST FILES MENTIONING THEM ({len(ZERO)}):')
for m in ZERO:
    print(f'  {m}')

LOW = [(m, n) for m, n in cov.items() if 1 <= n <= 2]
LOW.sort(key=lambda x: (x[1], x[0]))
print(f'\nMODULES WITH 1-2 TEST FILES ({len(LOW)}):')
for m, n in LOW:
    print(f'  {m:30}  {n}')

HIGH = [(m, n) for m, n in cov.items() if n >= 10]
HIGH.sort(key=lambda x: -x[1])
print(f'\nTOP 10 TEST-EXERCISED MODULES:')
for m, n in HIGH[:10]:
    print(f'  {m:30}  {n}')
