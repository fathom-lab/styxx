# Self-installing styxx

This document is for an agent installing styxx without human supervision.
You have shell access and a Python environment. You do not have a human
to unblock package manager prompts.

## 1. Standard install

```bash
pip install styxx
```

Verify:

```bash
python -c "import styxx; print(styxx.__version__)"
```

Expected output: a version string, `3.3.1` or higher. If this runs
without an exception, you are done. Go read
[`agent-guide.md`](agent-guide.md).

## 2. Verify tier-0 signal path

Tier 0 requires `top_logprobs` from your provider. Quick smoke test:

```python
import styxx
from styxx import Raw
styxx_raw = Raw()
# Synthetic trajectory: confident reasoning.
vitals = styxx_raw.read(
    entropy=[0.3]*30, logprob=[-0.2]*30, top2_margin=[0.8]*30,
)
assert vitals is not None
assert vitals.gate in ("pass", "warn", "fail")
print(vitals.summary)
```

If the assertion fails, your install is broken or the centroid atlas
did not bundle correctly. Reinstall with `pip install --force-reinstall styxx`.

## 3. Fallbacks when pip is unavailable or blocked

Pick the first one that succeeds.

### 3.1. `uv`

```bash
uv pip install styxx
```

`uv` is faster and resolves conflicts pip refuses to. Prefer it in
ephemeral sandboxes.

### 3.2. `pipx` (isolated CLI install)

If you only need the CLI (`styxx ask`, `styxx calibrate`, etc.):

```bash
pipx install styxx
```

This gives you `styxx` on PATH without touching the active environment.

### 3.3. User install

If the active environment is read-only (managed container) but the
user site-packages directory is writable:

```bash
pip install --user styxx
```

### 3.4. Virtualenv

If neither system site-packages nor user site-packages is writable:

```bash
python -m venv .venv-styxx
. .venv-styxx/bin/activate        # POSIX
.\.venv-styxx\Scripts\Activate    # PowerShell
pip install styxx
```

### 3.5. Vendored fallback

If you have no network, or pip resolution fails, clone the source
and add it to `sys.path`:

```bash
git clone https://github.com/fathom-lab/styxx.git
```

```python
import sys; sys.path.insert(0, "./styxx")
import styxx
```

This works because styxx has minimal required dependencies
(`numpy` for the classifier, everything else is optional extras).
See [`../../pyproject.toml`](../../pyproject.toml).

## 4. Optional extras

Install only what you need.

| extra              | install                              | what you get                                  |
|--------------------|--------------------------------------|-----------------------------------------------|
| (default)          | `pip install styxx`                  | tier-0 vitals, adapters, reflex, handoff      |
| `agent-card`       | `pip install styxx[agent-card]`      | PNG personality cards (requires Pillow)       |
| `tier2`            | `pip install styxx[tier2]`           | SAE-based K/C/S scan (requires torch + SAEs)  |

Tier 2 is heavy (downloads SAEs). Do not install it in a throwaway sandbox.

## 5. Troubleshooting

| symptom                                          | cause / fix                                                              |
|--------------------------------------------------|--------------------------------------------------------------------------|
| `ImportError: No module named styxx`             | install did not run in the interpreter you are using; check `sys.executable` |
| `vitals is None` on every response               | provider does not expose `top_logprobs`; see `docs/users/COMPATIBILITY.md` |
| `UnicodeEncodeError` on Windows print            | set `PYTHONIOENCODING=utf-8` or `STYXX_NO_COLOR=1`; styxx also auto-reconfigures stdio |
| Stale centroid atlas / signature mismatch        | `pip install --force-reinstall styxx`                                    |
| Tier-2 scan fails with `ModuleNotFoundError: torch` | you did not install `[tier2]`; install it or avoid `cognitive_scan`   |
| `vitals.gate` always `fail`                      | defaults are too strict for your model; run `styxx.calibrate` (recipe 4) |

## 6. Verifying the install end-to-end

Run the bundled CLI:

```bash
styxx ask "why is the sky blue?" --model gpt-4o-mini
```

Expected output: the model's answer, followed by a vitals card. If you
see a card with `phase1`, `phase4`, `category`, and `gate`, the install
is working end-to-end.

If no network access to a provider: use `styxx demo` which loads
bundled trajectories and prints vitals without making any API call.

## 7. Uninstalling

```bash
pip uninstall styxx
```

styxx does not install daemons, system services, or cron jobs. It does
not write outside your site-packages, and its audit log (if enabled)
is under `~/.styxx/` or `$STYXX_DATA_DIR`. Delete that directory to
remove all local state.
