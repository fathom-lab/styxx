# -*- coding: utf-8 -*-
"""eval.py: the run of the prereg-ci five-minute example (examples/prereg-ci-demo/README.md).

    python eval.py                          score under PREREG_digits.md and write receipt.json
    python eval.py --plan FILE --out PATH   score under FILE and write PATH; the receipt still names
                                            PREREG_digits.md as its PREREG, whichever file was read
    python eval.py --spread                 the dev seeds the bars and the spread come from; writes nothing

The run: scikit-learn's bundled digits set (no download), split in half by class with random_state 0;
a StandardScaler and a LogisticRegression fitted on one half and read on the other. Two numbers:
held-out accuracy, and the expected calibration error (ECE: ten equal-width bins of the top class's
predicted probability; each bin's share of the held-out half times |its accuracy - its mean
confidence|, summed). Both are rounded to six decimals before anything reads them, so the receipt
holds the numbers the verdict was read from. One thread, so the sums run in one order.

The verdict is the plan's gates table read by ``styxx.prereg_ci.rescore`` over the receipt, and the
receipt records it with the gates map and the digest of the fence it was read under: the record a
merge gate compares with the sealed plan. Every gate's margin is printed (how far the number sits on
the passing side of its bar; negative when the gate does not fire), and MARGIN_INSIDE_SPREAD beside a
margin smaller than that metric's spread over the dev seeds (``SPREAD``, measured by ``--spread``),
so a number another machine's library versions move reads as a print, not a surprise.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"                  # set before numpy loads: one thread, one summation order

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PREREG = "PREREG_digits.md"
SEED = 0                                     # the example's one run
BAR_SEED = 7                                 # the dev run the ECE bar was read from (README.md)
SPREAD_SEEDS = tuple(range(101, 111))        # the dev runs the spread was read from (README.md)
BINS = 10
DIGITS = 6
# max - min over SPREAD_SEEDS, rounded to four decimals; `python eval.py --spread` measures it again
SPREAD = {"result.accuracy": 0.0267, "result.ece": 0.0123}
FENCE = "`" * 3


def _ece(proba: "np.ndarray", hit: "np.ndarray") -> float:
    conf = proba.max(axis=1)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (conf > lo) & (conf <= hi)
        if inside.any():
            total += float(inside.mean()) * abs(float(hit[inside].mean()) - float(conf[inside].mean()))
    return total


def measure(seed: int) -> dict:
    """Held-out accuracy and ECE of the pipeline on the split *seed* draws."""
    x, y = load_digits(return_X_y=True)
    x_fit, x_held, y_fit, y_held = train_test_split(x, y, test_size=0.5, stratify=y, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x_fit, y_fit)
    proba = model.predict_proba(x_held)
    hit = model.classes_[proba.argmax(axis=1)] == y_held
    return {"accuracy": round(float(hit.mean()), DIGITS), "ece": round(_ece(proba, hit), DIGITS),
            "n_fit": int(len(y_fit)), "n_held_out": int(len(y_held))}


def _gates_text(text: str) -> str:
    """The lines between the plan's ```gates line and the closing fence line after it. The scanner
    styxx uses refuses more shapes than this; ``main`` checks that the two read the same text."""
    lines = text.split("\n")
    opens = [i for i, ln in enumerate(lines) if ln == FENCE + "gates"]
    if len(opens) != 1:
        raise SystemExit(f"eval.py: the plan holds {len(opens)} lines reading {FENCE}gates, not one")
    i = opens[0]
    for j in range(i + 1, len(lines)):
        if lines[j].rstrip() == FENCE:
            return "\n".join(lines[i + 1:j])
    raise SystemExit("eval.py: the plan's gates fence is never closed")


def _margin(op: str, value: float, bar: float) -> float:
    """How far *value* sits on the passing side of *bar* (negative on the failing side)."""
    return value - bar if op in (">=", ">") else bar - value if op in ("<=", "<") else -abs(value - bar)


def _dig(obj: dict, dotted: str):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def _spread() -> int:
    print(f"dev runs (the example's own seed, {SEED}, is not among them); scikit-learn {sklearn.__version__}, "
          f"numpy {np.__version__}, scipy {scipy.__version__}")
    bar_run = measure(BAR_SEED)
    print(f"seed {BAR_SEED:>3}: accuracy {bar_run['accuracy']:.6f}  ece {bar_run['ece']:.6f}   "
          f"(the run the ECE bar is read from: its ECE rounded up to the next 0.01 is "
          f"{math.ceil(bar_run['ece'] * 100) / 100:.2f})")
    runs = [measure(s) for s in SPREAD_SEEDS]
    for s, r in zip(SPREAD_SEEDS, runs):
        print(f"seed {s:>3}: accuracy {r['accuracy']:.6f}  ece {r['ece']:.6f}")
    for key in ("accuracy", "ece"):
        vals = [r[key] for r in runs]
        spread = max(vals) - min(vals)
        held = SPREAD[f"result.{key}"]
        print(f"{key}: {min(vals):.6f} to {max(vals):.6f} over seeds {SPREAD_SEEDS[0]}-{SPREAD_SEEDS[-1]}, "
              f"spread {spread:.6f} (SPREAD holds {held}, " + ("the same to four decimals)" if round(spread, 4) == held
                                                              else "which differs: these versions move it)"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", default=PREREG, help="the file whose gates table scores the run (default "
                                                   f"{PREREG}); the receipt names {PREREG} either way")
    ap.add_argument("--out", default="receipt.json", help="where the receipt is written (default receipt.json)")
    ap.add_argument("--spread", action="store_true", help="run the dev seeds and print their spread; write nothing")
    args = ap.parse_args(argv)
    if args.spread:
        return _spread()
    try:
        from styxx.prereg_ci import rescore
    except ImportError as e:
        print(f"eval.py: styxx.prereg_ci does not import ({e}); install styxx (examples/prereg-ci-demo/README.md)",
              file=sys.stderr)
        return 2
    # read as text, so a checkout that writes CRLF scores the bytes git stores (LF), as styxx.protocol reads them
    text = Path(args.plan).read_text(encoding="utf-8")
    gates_text = _gates_text(text)
    table = json.loads(gates_text)
    result = measure(SEED)
    receipt = {"prereg": PREREG, "smoke": False, "seed": SEED,
               "versions": {"python": f"{sys.version_info[0]}.{sys.version_info[1]}", "numpy": np.__version__,
                            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
               "result": result}
    scored = rescore(text, receipt)
    if scored["code"] is not None:
        print(f"eval.py: the plan's table does not score this run: {scored['code']} {scored['detail']}",
              file=sys.stderr)
        return 2
    if scored["gates_sha256"] != hashlib.sha256(gates_text.encode("utf-8")).hexdigest():
        print("eval.py: styxx's scanner read another gates fence than this script did; margins not printed",
              file=sys.stderr)
        return 2
    receipt.update(verdict=scored["verdict"], gates=scored["gates"], gates_sha256=scored["gates_sha256"])
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(receipt, indent=1) + "\n")
    print(f"{args.out}: {scored['verdict']} under {args.plan} (gates {scored['gates_sha256'][:12]}...); "
          f"accuracy {result['accuracy']:.6f}, ece {result['ece']:.6f} on {result['n_held_out']} held-out images")
    for name, gate in table["gates"].items():
        value = _dig(receipt, gate["metric"])
        margin = _margin(gate["op"], value, gate["value"])
        spread = SPREAD.get(gate["metric"])
        flag = (f"  MARGIN_INSIDE_SPREAD (spread {spread} over seeds {SPREAD_SEEDS[0]}-{SPREAD_SEEDS[-1]})"
                if spread is not None and abs(margin) < spread else "")
        print(f"  {name} {gate['metric']} {value:.6f} {gate['op']} {gate['value']} "
              f"{'fires' if scored['gates'][name] else 'does not fire'}, margin {margin:+.6f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
