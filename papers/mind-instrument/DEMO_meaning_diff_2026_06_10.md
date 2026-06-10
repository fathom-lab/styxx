# DEMO — styxx.meaning_diff on real models

**2026-06-10 · styxx 7.15.0.** The product moment, run on the norm-equalized Atlas reps
(`normeq_reps.npz`). Sender concept set = the 96-concept battery. Receipt for these numbers:
`meaning_diff_v0_validation.json` (the instrument) + the reps npz.

| comparison | agreement | verdict | top moved concepts |
| --- | --- | --- | --- |
| Qwen2.5-1.5B vs itself | 1.000 | HEALTHY | (none) |
| Qwen2.5-1.5B vs Qwen2.5-3B (same family) | 0.930 | HEALTHY | mirror, mouse, lamp, drum |
| Qwen2.5-1.5B vs Llama-3.2-1B (cross family) | 0.915 | HEALTHY | mirror, lamp, mouse, cat |
| Qwen2.5-1.5B vs a shuffled copy of itself | 0.017 | BROKEN | boat, cello, ear, drizzle |

The instrument does exactly what a migration/regression tool must: a model is identical to itself
(no false divergence), close to its sibling and a cross-family peer (HEALTHY, with the few genuinely
differently-encoded concepts named — note "mirror" and "mouse", both polysemous, surface as the
movers), and a destroyed geometry reads BROKEN. The same call powers: did GPT-N+1 keep GPT-N's
meaning, did 4-bit quantization break any concepts, did this fine-tune regress — answered, with the
lost concepts named, in one function and zero labeled data.

Reproduce:
```python
import numpy as np, styxx
from styxx.meaning_diff import meaning_diff
z = np.load("papers/mind-instrument/normeq_reps.npz")
words = styxx.mind.BATTERY
print(meaning_diff(z["Qwen2.5-1.5B"], z["Qwen2.5-3B"], words=words)["agreement"])  # 0.93
```
