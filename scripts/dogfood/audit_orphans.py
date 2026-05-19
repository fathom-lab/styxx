"""Corrected orphan audit accounting for __init__.py re-exports + CLI subcommands + tests + scripts."""
import pathlib, re

root = pathlib.Path('styxx')
top_mods = sorted({p.stem if p.is_file() else p.name
                   for p in root.iterdir()
                   if (p.is_file() and p.suffix == '.py' and not p.name.startswith('_'))
                   or (p.is_dir() and not p.name.startswith('_') and p.name not in ('__pycache__', 'fonts', 'recipes', 'centroids'))})
top_mods = [m for m in top_mods if not m.startswith('_')]

SKIP = ('site-packages', '.pytest_cache')
all_py = []
for p in pathlib.Path('.').rglob('*.py'):
    s = str(p).replace('\\', '/')
    if any(k in s for k in SKIP):
        continue
    if '/build/' in s:
        continue
    all_py.append(p)

print(f'Top-level modules in styxx/: {len(top_mods)}')
print(f'Python files scanned: {len(all_py)}')

unwired = []
wired_by = {}
for m in top_mods:
    pat = re.compile(
        r'(?:from\s+(?:\.\.?|styxx\.)?' + m + r'\b'
        r'|import\s+styxx\.' + m + r'\b'
        r'|from\s+styxx\s+import\s+[^#\n]*\b' + m + r'\b'
        r'|from\s+\.\s+import\s+[^#\n]*\b' + m + r'\b)'
    )
    importers = set()
    for p in all_py:
        ps = str(p).replace('\\', '/')
        if f'/styxx/{m}.py' in ps or f'/styxx/{m}/' in ps:
            continue
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if pat.search(text):
            importers.add(ps)
    wired_by[m] = importers
    if not importers:
        unwired.append(m)

print()
print(f'GENUINELY UNWIRED ({len(unwired)} of {len(top_mods)}):')
for m in unwired:
    print(f'  {m}')

print()
print('LIGHTLY WIRED (1-2 importers):')
for m in sorted(top_mods):
    n = len(wired_by[m])
    if 0 < n <= 2:
        names = sorted(wired_by[m])
        print(f'  {m:30} {n}  <-  {", ".join(names)}')

print()
print('HEAVILY WIRED (top 10):')
heavy = sorted(top_mods, key=lambda m: -len(wired_by[m]))[:10]
for m in heavy:
    print(f'  {m:30} {len(wired_by[m])}')
