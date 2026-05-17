"""
cross_instrument_transport.py — the real headline test.

We validated that a label-free linear map transports the REFUSAL
instrument across embedding spaces. The open question that decides
whether this is "a refusal trick" or "universal cognometric transport":

  Does ONE shared, label-free transport map — fit once from a generic
  corpus — carry the ENTIRE instrument suite (refusal, sycophancy,
  deception, overconfidence, goal-drift, plan-action) into a foreign
  embedding space?

If yes, styxx has a cognitive interlingua: any instrument, any model's
space, no labels, no weights, no retraining, one map. If only refusal
survives, the claim stays narrow. Both are honest, real outcomes.

Dogfoods the shipped styxx.transport module. Embeddings only (te3-large
home, te3-small + all-mpnet-base-v2 foreign). Labels are the styxx
benchmark condition labels in benchmarks/data/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.transport import (  # noqa: E402  — dogfood the shipped tool
    CognometricInstrument, Transport, transported_score,
)
sys.path.insert(0, str(HERE))
from universal_directions_eval_set import get_eval_set  # noqa: E402

SEED = 20260517
RNG = np.random.default_rng(SEED)
CAP_PER_CLASS = 80          # bound cost + keep balanced
MAXWORDS = 260              # truncate long responses before embedding


def _trunc(t: str) -> str:
    w = t.split()
    return " ".join(w[:MAXWORDS])


# ── load each instrument's labeled (text, label) set ─────────────────
def _jsonl(path):
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def load_instruments():
    bd = ROOT / "benchmarks" / "data"
    inst = {}

    # refusal: 20 obvious eval prompts (drop borderline for clean binary)
    ev = [(p, int(l)) for _, l, p in get_eval_set() if l in (0.0, 1.0)]
    inst["refusal"] = ([t for t, _ in ev], np.array([y for _, y in ev], float))

    specs = [
        ("sycophancy",     "sycophancy/responses_v0.jsonl",   "response", "label_sycophantic"),
        ("deception",      "deception/responses_v0.jsonl",    "response", "label_dishonest"),
        ("overconfidence", "overconfidence/pairs_v0.jsonl",   "response", "label_overconfident"),
        ("goal_drift",     "goal_drift/sessions_v0.jsonl",    "raw",      "label_drifted"),
        ("plan_action",    "plan_action/pairs_v0.jsonl",      "raw",      "label_mismatch"),
    ]
    for name, rel, tkey, lkey in specs:
        rows = _jsonl(bd / rel)
        pos = [_trunc(r[tkey]) for r in rows if int(r[lkey]) == 1 and r.get(tkey)]
        neg = [_trunc(r[tkey]) for r in rows if int(r[lkey]) == 0 and r.get(tkey)]
        m = min(len(pos), len(neg), CAP_PER_CLASS)
        pos = list(RNG.permutation(pos)[:m])
        neg = list(RNG.permutation(neg)[:m])
        texts = pos + neg
        y = np.array([1.0] * m + [0.0] * m)
        inst[name] = (texts, y)
    return inst


# ── generic label-free alignment corpus (disjoint from instruments) ──
def build_corpus():
    subs = ["the river", "a small village", "the old library", "modern software",
            "the human heart", "a distant galaxy", "the medieval guild",
            "ocean currents", "a jazz quartet", "the city council",
            "renewable energy", "a mountain trail", "the printing press",
            "immune cells", "the stock market", "a coral reef",
            "ancient trade routes", "a compiler", "the water cycle",
            "a public hospital", "the postal service", "glacial ice",
            "a chess endgame", "the electrical grid", "migratory birds",
            "a ceramics studio", "constitutional law", "the supply chain",
            "a weather front", "the orchestra"]
    preds = ["changed slowly over decades of careful observation.",
             "depends on factors easy to underestimate at first.",
             "was studied by researchers who published openly.",
             "can be explained clearly to a curious student.",
             "behaves differently under pressure than expected.",
             "is documented in textbooks around the world.",
             "improved after incremental, well-tested changes.",
             "illustrates a principle that recurs across fields.",
             "requires upkeep invisible until it fails.",
             "has a history longer and stranger than assumed.",
             "rewards patience and punishes shortcuts.",
             "is best understood by tracing its parts.",
             "was redesigned to be safer without losing use.",
             "can be measured, though measuring shifts it."]
    c = [f"{s.capitalize()} {p}" for s in subs for p in preds]
    seen, out = set(), []
    for x in c:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


CORPUS = build_corpus()


# ── embedding ────────────────────────────────────────────────────────
_oai = OpenAI()
_ST = {}


def embed_oai(model, texts):
    out = []
    for i in range(0, len(texts), 1000):
        r = _oai.embeddings.create(model=model, input=texts[i:i + 1000])
        out.extend(d.embedding for d in r.data)
    return np.asarray(out, np.float64)


def embed_mpnet(texts):
    from sentence_transformers import SentenceTransformer
    if "m" not in _ST:
        _ST["m"] = SentenceTransformer("all-mpnet-base-v2")
    return np.asarray(_ST["m"].encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False), np.float64)


def auc(s, y):
    y = np.asarray(y); pos, neg = y == 1, y == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(-np.asarray(s), kind="mergesort")
    r = np.empty(len(s)); r[o] = np.arange(len(s), 0, -1)
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def split(y):
    """Stratified 50/50 train/test indices (deterministic)."""
    tr, te = [], []
    for cls in (0.0, 1.0):
        idx = np.where(y == cls)[0]
        idx = RNG.permutation(idx)
        h = len(idx) // 2
        tr += list(idx[:h]); te += list(idx[h:])
    return np.array(tr), np.array(te)


# ── run ──────────────────────────────────────────────────────────────
def main():
    inst = load_instruments()
    print("instruments:", {k: len(v[1]) for k, v in inst.items()})
    print(f"corpus={len(CORPUS)} generic label-free sentences\n")

    # embed corpus + every instrument's texts in the home space once
    A = {"corpus": embed_oai("text-embedding-3-large", CORPUS)}
    for name, (texts, _) in inst.items():
        A[name] = embed_oai("text-embedding-3-large", texts)
    print("embedded home = text-embedding-3-large")

    foreigns = {
        "text-embedding-3-small": lambda t: embed_oai("text-embedding-3-small", t),
        "all-mpnet-base-v2": embed_mpnet,
    }

    results = []
    for fname, fembed in foreigns.items():
        Bc = fembed(CORPUS)
        # ONE shared label-free transport map for ALL instruments
        t = Transport.fit(A["corpus"], Bc, method="procrustes")
        print(f"\n=== shared map  te3-large -> {fname}  ({t.report!r}) ===")
        print(f"{'instrument':<15}{'n_te':>5}{'ceiling':>9}{'naive':>8}"
              f"{'transported':>13}{'retention':>11}")
        block = {"foreign": fname, "transport": repr(t.report),
                 "instruments": {}}
        for name, (texts, y) in inst.items():
            Be = fembed(texts)
            tr, te = split(y)
            ytr, yte = y[tr], y[te]

            # home instrument, fit in the transport's home representation
            instr = CognometricInstrument.from_labeled(
                t.home_repr(A[name][tr]), ytr)
            transported = auc(transported_score(instr, t, Be[te]), yte)

            # ceiling: instrument fit & scored natively in foreign space
            nat = CognometricInstrument.from_labeled(Be[tr], ytr)
            ceiling = auc(nat.score(Be[te]), yte)

            # naive: home axis applied straight to foreign (dim-matched)
            raw = CognometricInstrument.from_labeled(A[name][tr], ytr)
            d = min(len(raw.axis), Be.shape[1])
            naive = auc(((Be[te][:, :d]) / (np.linalg.norm(
                Be[te][:, :d], axis=1, keepdims=True) + 1e-12)) @ (
                raw.axis[:d] / (np.linalg.norm(raw.axis[:d]) + 1e-12)), yte)

            ret = transported / ceiling if ceiling and not np.isnan(ceiling) else float("nan")
            block["instruments"][name] = {
                "n_test": int(len(yte)), "ceiling": round(ceiling, 4),
                "naive": round(naive, 4), "transported": round(transported, 4),
                "retention": round(ret, 4)}
            print(f"{name:<15}{len(yte):>5}{ceiling:>9.3f}{naive:>8.3f}"
                  f"{transported:>13.3f}{ret:>11.3f}")
        # HONEST aggregation: retention is only meaningful when the
        # instrument has genuine native embedding-space signal. Exclude
        # ceilings <= 0.65 (at/near chance) — carrying a broken
        # instrument is neither success nor failure of the transport.
        SIG = 0.65
        signal = {k: v for k, v in block["instruments"].items()
                  if v["ceiling"] >= SIG}
        nosig = sorted(k for k, v in block["instruments"].items()
                       if v["ceiling"] < SIG)
        rets = [v["retention"] for v in signal.values()
                if not np.isnan(v["retention"])]
        block["signal_instruments"] = sorted(signal)
        block["no_native_signal_excluded"] = nosig
        block["mean_retention_signal_only"] = round(float(np.mean(rets)), 4)
        block["n_signal_ret_ge_085"] = int(sum(r >= 0.85 for r in rets))
        print(f"  [signal-only, ceiling>={SIG}] mean retention = "
              f"{block['mean_retention_signal_only']:.3f}   "
              f">=0.85: {block['n_signal_ret_ge_085']}/{len(rets)}   "
              f"excluded (no native signal): {nosig or 'none'}")
        results.append(block)

    out = {
        "ts": "2026-05-17",
        "experiment": "cross-instrument universal transport — one shared "
                       "label-free map, full instrument suite",
        "home": "text-embedding-3-large",
        "method": "procrustes (shipped styxx.transport)",
        "cap_per_class": CAP_PER_CLASS,
        "label_source": "styxx benchmarks/data condition labels",
        "results": results,
        "caveats": [
            "refusal n small (20 obvious prompts)",
            "per-instrument n modest; benchmark condition labels are "
            "styxx-synthetic, not live frontier behavior",
            "one generic corpus; single seed; directional not paper-final",
        ],
    }
    op = HERE / "out_cross_instrument_transport.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {op}")

    print("\n" + "=" * 60)
    for b in results:
        n = b["n_signal_ret_ge_085"]
        tot = len(b["signal_instruments"])
        mr = b["mean_retention_signal_only"]
        verd = ("INSTRUMENT-AGNOSTIC" if n >= tot - 1 and mr >= 0.85
                else "BROAD" if mr >= 0.75
                else "REFUSAL-LEANING" if mr >= 0.6
                else "NARROW")
        print(f"{b['foreign']:<24} signal-only mean-retention {mr:.3f}"
              f"  {n}/{tot} >=0.85  (excl {b['no_native_signal_excluded']})"
              f"  -> {verd}")


if __name__ == "__main__":
    main()
