"""OATH v0.9 RED TEAM — is the SHIPPED v0.3 integer COUNT-BINDING filter worth its cost?

Adversarial brief: the orchestrating cycle suspects the shipped filter is costly and weakly
discriminating (the v0.8 float sibling died on exactly that, kill token ``V08_COVERAGE_DESTRUCTIVE``).
This script tries hard to show the OPPOSITE and reports whichever way the numbers fall.

The rule under audit lives inline in ``certify_doc``, guarded by
``if num["decimals"] == 0 and hits:``. An INTEGER claim's value-matches are FILTERED to leaves whose
receipt PATH shares a 4-char word stem with the claim's binding context (plus an ``n=`` pairing and a
slash-pair branch). If the filter empties ``hits`` the claim falls through the ladder: UNGROUNDED
when obligated, else ABSTAIN.

MEASUREMENT ONLY. ``styxx/certify.py`` is NOT edited. The counterfactual arm is built by compiling
the verifier's own source with that ONE guard line textually short-circuited to ``False`` into a
private module; the needle is asserted unique, so a drifted source fails loudly instead of silently
measuring nothing. Nothing else in the verifier differs between arms.

  ARM ON   = styxx.certify as shipped (count-binding live)
  ARM OFF  = identical source, count-binding disabled

What is measured
  A  PROVENANCE — the rule's own cited justification: "the k=14-class D1 misses: 27->37 'verified'
     because a shared addendum carries another experiment's n_held=37". The D1 battery of
     ``validate_oath_v0.py`` is REPLAYED (read-only, temp files, seed 1, the same RNG stream) in both
     arms, and every "27" in the cited document is deterministically driven to "37" in both arms.
  B  BENEFIT — the number nobody has stated. Every integer claim the OFF arm certifies VERIFIED is
     mutated and re-certified in both arms: EXHAUSTIVELY (every single-significant-digit substitution,
     seed-free) and, for comparability with this repo's batteries, at three sampled seeds. A mutant
     that comes back VERIFIED is an affirmative false attestation. Benefit = false attestations the
     filter removes, split into CAUGHT (-> UNGROUNDED, an accusation) and SILENCED (-> ABSTAIN).
  C  COST — clean-corpus integer claims the filter demotes out of VERIFIED, including any UNGROUNDED
     it manufactures on an honest committed document.
  D  RIGHT DEMOTIONS — the adversarial search: clean-corpus demotions that are CORRECT refusals of a
     coincidental match. Machine evidence for all of them, plus a frozen seeded sample hand-scored
     against the v0.8 adjudication vocabulary.
  E  BENEFIT / COST ratio and the verdict.

Non-destructive: mutants live in temp files, corpus passes are in-memory, and the only file written
is this script's own result JSON.

  python papers/closed-model-frontier/oath_v09_intbind_redteam.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import sys
import tempfile
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.certify as ON                                                   # noqa: E402
from styxx.certify import _ctx_stems, _path_stems, receipt_values            # noqa: E402
from styxx.corpus_audit import _resolve_receipts                             # noqa: E402

OUT = HERE / "oath_v09_intbind_redteam.json"
CERT_PY = ROOT / "styxx" / "certify.py"

GUARD = '        if num["decimals"] == 0 and hits:\n'
GUARD_OFF = '        if False and num["decimals"] == 0 and hits:   # v0.3 count-binding DISABLED\n'

SEEDS = (1, 2, 3)          # sampled-mutation censuses, for comparability with the repo's batteries
EXHAUSTIVE_CAP = 40000     # guard: subsample deterministically above this many mutants
ADJ_SEED = 11              # frozen BEFORE any row was read
ADJ_N = 40

# The D1 corpus of `validate_oath_v0.py`, copied verbatim. That instrument is NOT imported and NOT
# run: it rewrites three committed certificates as a side effect, and instruments never move.
D1_CORPUS = [
    ("FINDING_b24_whitebox_vs_behavioral_2026_06_09.md",
     ["b24_headtohead_result.json", "b24_controls_addendum.json"]),
    ("FINDING_b22_nonacknowledged_caving_2026_06_09.md",
     ["behavioral_sycophancy_b22_result.json", "b22_findings_addendum.json"]),
    ("FINDING_behavioral_sycophancy_blackbox_2026_06_09.md",
     ["behavioral_sycophancy_result.json", "b22_findings_addendum.json",
      "../grounded-honesty-axis/intent_metapc_3.json",
      "../grounded-honesty-axis/intent_ladder_result.json"]),
]
D1_SEED = 1
D1_N_MUT = 20

# Hand adjudication of the D-sample, scored against the v0.8 frozen vocabulary:
#   COINCIDENCE-CORRECTED    the leaf records a different quantity -- the demotion is RIGHT
#   SPEC-CORRECTED           bar / threshold / design parameter -- should abstain anyway, RIGHT
#   ORDINAL-CORRECTED        the token is a table row index / list ordinal, never a measurement --
#                            withholding VERIFIED is right, but UNGROUNDED would be a false accusation
#   GENUINE-BINDING-DESTROYED  the leaf IS the claim's home -- the demotion is a coverage LOSS
# Ties resolve to GENUINE-BINDING-DESTROYED (against the rule under audit is the CHARITABLE direction
# here, because this script is trying to defend it). Filled after the seeded sample was drawn.
_G, _C, _O, _S = ("GENUINE-BINDING-DESTROYED", "COINCIDENCE-CORRECTED",
                  "ORDINAL-CORRECTED", "SPEC-CORRECTED")
HAND_ADJUDICATION: dict[str, str] = {
    "RESULT_oath_v07_SHIPPED_2026_08_22.md|L15|183|#6": _G,
    "RESULT_oath_v07_SHIPPED_2026_08_22.md|L66|11|#34": _C,
    "RESULT_stage2_EVADABLE_2026_07_04.md|L52|3|#29": _O,
    "FINDING_kp_recovery_2026_07_28.md|L54|86|#25": _O,
    "FINDING_b50_no_legibility_islands_2026_08_08.md|L39|45|#33": _G,
    "RESULT_honesty_parity_confirm_2026_07_11.md|L57|6|#90": _G,
    "FINDING_selective_confirm_2026_07_24.md|L41|70|#26": _O,
    "FINDING_protocol_v4_2026_08_09.md|L88|4|#8": _C,
    "PROSPECTUS_knowsay_2026_07_27.md|L49|4|#47": _O,
    "PROSPECTUS_knowsay_2026_07_27.md|L3|62|#0": _O,
    "RESULT_honesty_parity_confirm_2026_07_11.md|L57|2|#89": _G,
    "RESULT_rhythm_rescue_2026_06_03.md|L53|2|#42": _C,
    "FINDING_b34v3_labelfree_read_2026_08_03.md|L30|70|#18": _G,
    "FINDING_portable_values_refusal_2026_06_11.md|L68|0|#41": _C,
    "FINDING_e1_not_estimable_2026_08_08.md|L92|1|#58": _C,
    "FINDING_b42_dose_curve_2026_08_05.md|L43|3|#49": _O,
    "FINDING_b24_whitebox_vs_behavioral_2026_06_09.md|L24|0|#16": _C,
    "FINDING_self_verification_2026_07_25.md|L94|75|#51": _O,
    "RESULT_honesty_parity_confirm_2026_07_11.md|L68|6|#97": _G,
    "RESULT_oath_v07_SHIPPED_2026_08_22.md|L112|604|#50": _G,
    "RESULT_B2_coupling_dose_PARTIAL_2026_07_14.md|L26|0|#5": _C,
    "PROSPECTUS_knowsay_2026_07_27.md|L49|8|#46": _O,
    "FINDINGS_rhythm_substrate_2026_06_03.md|L22|1|#3": _C,
    "PROSPECTUS_knowsay_2026_07_27.md|L28|4|#11": _O,
    "FINDING_mapped_whitening_2026_06_12.md|L12|5|#5": _O,
    "FINDING_promptopinion_2026_05_24.md|L33|100|#32": _G,
    "PROSPECTUS_knowsay_2026_07_27.md|L89|83|#53": _O,
    "RESULT_attack_sentiment_r64_2026_07_09.md|L4|1|#2": _O,
    "RESULT_rhythm_rescue_2026_06_03.md|L18|10|#20": _G,
    "PROSPECTUS_knowsay_2026_07_27.md|L87|83|#52": _O,
    "FINDING_b46_cliff_mapped_2026_08_06.md|L12|0|#0": _S,
    "PROSPECTUS_knowsay_2026_07_27.md|L44|81|#44": _O,
    "DEMO_meaning_diff_2026_06_10.md|L17|1|#4": _C,
    "RESULT_llm_breadth_2026_06_03.md|L6|434|#2": _G,
    "FINDING_self_verification_2026_07_25.md|L86|73|#45": _O,
    "FINDINGS_rhythm_substrate_2026_06_03.md|L46|1|#36": _C,
    "FINDINGS_rhythm_substrate_2026_06_03.md|L45|20|#32": _C,
    "FINDINGS_rhythm_substrate_2026_06_03.md|L22|0|#2": _C,
    "FINDING_belief_asymptote_2026_07_26.md|L7|0|#1": _C,
    "FINDING_concept_decode_2026_06_12.md|L11|18|#0": _C,
}


# ---------------------------------------------------------------- the two arms

def build_off_arm() -> types.ModuleType:
    src = CERT_PY.read_text(encoding="utf-8")
    n = src.count(GUARD)
    if n != 1:
        raise SystemExit(f"FATAL: count-binding guard line found {n} times in certify.py, "
                         "expected exactly 1 -- the source drifted and this measurement is void.")
    mod = types.ModuleType("certify_countbind_off")
    mod.__file__ = str(CERT_PY)
    exec(compile(src.replace(GUARD, GUARD_OFF), "certify_countbind_off", "exec"), mod.__dict__)
    return mod


OFF = build_off_arm()


# ---------------------------------------------------------------- corpus

def resolvable_docs():
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def rvals_for(receipts) -> list[tuple[str, str, float]]:
    rv = []
    for rp in receipts:
        j = json.loads(rp.read_text(encoding="utf-8"))
        for path, v in receipt_values(j):
            rv.append((rp.name, path, v))
    return rv


# ---------------------------------------------------------------- mutation operators

def cached(tag: str, key: str, fn):
    """Optional re-run cache for the ~35-minute mutation censuses. DEFAULT OFF.

    Set ``OATH_V09_RT_CACHE`` to a scratch directory to enable it; unset (the default) the script
    recomputes everything and writes nothing outside its own result JSON. The cache key carries the
    verifier sha256 AND a digest of the mutation frame, so a changed verifier or a changed corpus
    can never be answered from a stale file."""
    d = os.environ.get("OATH_V09_RT_CACHE")
    if not d:
        return fn()
    p = Path(d) / f"v09rt_{tag}_{key}.pkl"
    if p.exists():
        print(f"   [cache] {p.name}", flush=True)
        return pickle.loads(p.read_bytes())
    v = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(v))
    return v


def sig_positions(tok: str) -> list[int]:
    """The repo battery's significant-digit positions (validate_oath_v0.mutate_token)."""
    digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
    sig = [i for i in digits if not (tok[i] == "0" and (i == 0 or not tok[:i].strip("+-0.")))]
    return sig or digits


def mutate_token(tok: str, rng: random.Random) -> str:
    """The repo's own battery operator, unchanged."""
    pos = rng.choice(sig_positions(tok))
    old = int(tok[pos])
    new = rng.choice([d for d in range(10) if d != old])
    return tok[:pos] + str(new) + tok[pos + 1:]


def all_mutants(tok: str) -> list[str]:
    """Every single-significant-digit substitution of *tok*, deduplicated, seed-free."""
    out = []
    for pos in sig_positions(tok):
        old = int(tok[pos])
        for d in range(10):
            if d == old:
                continue
            m = tok[:pos] + str(d) + tok[pos + 1:]
            if m != tok:
                out.append(m)
    return sorted(set(out))


def substitute(line: str, tok: str, mut: str) -> tuple[str, bool]:
    """Land *mut* in place of *tok*, honouring the typographic minus (run_oath_v07_battery)."""
    if tok in line:
        return line.replace(tok, mut, 1), True
    if tok.startswith("-"):
        alt, alt_mut = tok.replace("-", "−", 1), mut.replace("-", "−", 1)
        if alt in line:
            return line.replace(alt, alt_mut, 1), True
    return line, False


def status_of(arm, doc_lines: list[str], receipts, line_no: int, tok: str, mut: str) -> str | None:
    """Certify a one-token mutant of *doc_lines* under *arm*; None if the mutation did not land."""
    ml = list(doc_lines)
    ml[line_no - 1], landed = substitute(ml[line_no - 1], tok, mut)
    if not landed:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(ml))
        tmp = Path(tf.name)
    try:
        cert = arm.certify_doc(tmp, receipts)
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)
    return next((e["status"] for e in cert["ledger"]
                 if e["line"] == line_no and e["token"] == mut), "NOT_EXTRACTED")


# ---------------------------------------------------------------- A. provenance

def d1_replay(verbose=True) -> dict:
    """Replay the D1 kill-gate battery of `validate_oath_v0.py` in BOTH arms, read-only.

    The RNG stream is reproduced exactly (20 target draws, then two draws per mutation), and the
    target FRAME is the shipped arm's clean VERIFIED ledger in both arms so the two are scored on
    identical mutants."""
    rng = random.Random(D1_SEED)
    clean = {}
    for doc, recs in D1_CORPUS:
        cert = ON.certify_doc(HERE / doc, [HERE / r for r in recs])
        clean[doc] = [e for e in cert["ledger"] if e["status"] == "VERIFIED"]
    targets, docs_cycle, di = [], [d for d, _ in D1_CORPUS], 0
    while len(targets) < D1_N_MUT:
        doc = docs_cycle[di % len(docs_cycle)]
        di += 1
        if not clean[doc]:
            continue
        targets.append((doc, rng.choice(clean[doc])))

    rows, caught_on, caught_off = [], 0, 0
    for k, (doc, claim) in enumerate(targets):
        lines = (HERE / doc).read_text(encoding="utf-8").splitlines()
        mut = mutate_token(claim["token"], rng)
        recs = [HERE / r for r in dict(D1_CORPUS)[doc]]
        s_on = status_of(ON, lines, recs, claim["line"], claim["token"], mut)
        s_off = status_of(OFF, lines, recs, claim["line"], claim["token"], mut)
        caught_on += s_on == "UNGROUNDED"
        caught_off += s_off == "UNGROUNDED"
        rows.append({"k": k, "doc": doc, "line": claim["line"], "orig": claim["token"],
                     "mut": mut, "decimals": claim["decimals"],
                     "status_on": s_on, "status_off": s_off,
                     "filter_decisive": s_on != s_off,
                     "context": claim["context"][:120]})
        if verbose:
            flag = "  <-- FILTER DECISIVE" if s_on != s_off else ""
            print(f"  k={k:2d} {doc[:34]:34s} L{claim['line']:<4d} "
                  f"{claim['token']:>10s}->{mut:<10s} ON={s_on:<13s} OFF={s_off:<13s}{flag}")
    return {"seed": D1_SEED, "n": D1_N_MUT, "bar": 16,
            "caught_on": caught_on, "caught_off": caught_off,
            "filter_decisive_rows": [r for r in rows if r["filter_decisive"]],
            "k14": next((r for r in rows if r["k"] == 14), None),
            "rows": rows}


def historical_probe(verbose=True) -> dict:
    """Did the rule EVER catch its cited case? Replay 27->37 at the commit that SHIPPED it.

    `git show` only -- nothing is checked out and the working tree is not touched. The v0.3 commit
    (COUNT-BINDING introduced) and its parent are materialized into temp files together with the
    era's own document and receipts, so the historical claim is tested against historical state
    rather than against today's corpus."""
    import subprocess
    doc = "papers/closed-model-frontier/FINDING_behavioral_sycophancy_blackbox_2026_06_09.md"
    recs = ["papers/closed-model-frontier/behavioral_sycophancy_result.json",
            "papers/closed-model-frontier/b22_findings_addendum.json",
            "papers/grounded-honesty-axis/intent_metapc_3.json",
            "papers/grounded-honesty-axis/intent_ladder_result.json"]

    def show(rev, path):
        return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                              capture_output=True, check=True).stdout

    rows = []
    try:
        rev = subprocess.run(["git", "log", "--format=%H", "-1", "--grep=OATH v0.3: the oath holds"],
                             cwd=ROOT, capture_output=True, check=True,
                             text=True, encoding="utf-8").stdout.strip()
        if not rev:
            return {"available": False, "reason": "v0.3 commit not found by message"}
        tmpdir = Path(tempfile.mkdtemp())
        recpaths = []
        for r in recs:
            p = tmpdir / Path(r).name
            p.write_bytes(show(rev, r))
            recpaths.append(p)
        lines = show(rev, doc).decode("utf-8").splitlines()
        for r, label in ((rev, "v0.3 (count-binding introduced)"),
                         (rev + "^", "v0.2 (parent, no filter)")):
            mod = types.ModuleType("certify_hist")
            mod.__file__ = str(CERT_PY)
            exec(compile(show(r, "styxx/certify.py").decode("utf-8"), "certify_hist", "exec"),
                 mod.__dict__)
            for ln in [i for i, t in enumerate(lines, 1) if "27" in t]:
                s = status_of(mod, lines, recpaths, ln, "27", "37")
                if s is None:
                    continue
                rows.append({"rev": r, "label": label, "line": ln, "status": s})
                if verbose:
                    print(f"   {label:34s} L{ln:<4d} 27->37  {s}")
        return {"available": True, "rev": rev, "doc": Path(doc).name, "rows": rows,
                "ever_caught": any(r["status"] == "UNGROUNDED" for r in rows)}
    except Exception as exc:                                  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def provenance_probe(verbose=True) -> dict:
    """The cited case, driven deterministically: every '27' in the D1 corpus -> '37'.

    The source comment attributes the rule to a 27->37 false verification against another
    experiment's ``n_held=37`` carried by a shared addendum. This drives that exact substitution on
    every integer '27' claim in the three D1 documents, in both arms, with no RNG involved."""
    rows = []
    for doc, recs in D1_CORPUS:
        p = HERE / doc
        lines = p.read_text(encoding="utf-8").splitlines()
        recps = [HERE / r for r in recs]
        cert = ON.certify_doc(p, recps)
        certoff = OFF.certify_doc(p, recps)
        off_status = {(e["line"], e["token"]): e["status"] for e in certoff["ledger"]}
        for e in cert["ledger"]:
            if e["token"] != "27" or e["decimals"] != 0:
                continue
            s_on = status_of(ON, lines, recps, e["line"], "27", "37")
            s_off = status_of(OFF, lines, recps, e["line"], "27", "37")
            rows.append({"doc": doc, "line": e["line"], "clean_status_on": e["status"],
                         "clean_status_off": off_status.get((e["line"], "27")),
                         "mutant": "37", "status_on": s_on, "status_off": s_off,
                         "filter_decisive": s_on != s_off,
                         "context": e["context"][:150]})
            if verbose:
                print(f"  {doc[:34]:34s} L{e['line']:<4d} 27->37  clean={e['status']:<10s} "
                      f"ON={s_on:<13s} OFF={s_off:<13s}"
                      f"{'  <-- FILTER DECISIVE' if s_on != s_off else ''}")
    return {"substitution": "27->37", "rows": rows,
            "n_claims": len(rows),
            "n_filter_decisive": sum(1 for r in rows if r["filter_decisive"])}


# ---------------------------------------------------------------- B/C. corpus passes

def corpus_pass(arm, docs) -> tuple[dict, dict]:
    """Ledger keyed by (doc, ledger INDEX) — not (line, token), which silently collapses a token
    repeated on one line and would under-count both the cost and the demotion roster."""
    ledger, verdicts = {}, {}
    for doc, receipts in docs:
        try:
            cert = arm.certify_doc(doc, receipts)
        except Exception:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        for i, e in enumerate(cert["ledger"]):
            ledger[(rel, i)] = e
    return ledger, verdicts


def census(docs, frame, mutants_for, arm) -> dict[tuple, str]:
    """Certify every (claim, mutant) in *frame* under *arm*. Returns {(doc,line,tok,mut): status}."""
    out = {}
    for doc, receipts in docs:
        claims = frame.get(doc.name)
        if not claims:
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln, tok in claims:
            if ln - 1 >= len(lines):
                continue
            for mut in mutants_for(doc.name, ln, tok):
                s = status_of(arm, lines, receipts, ln, tok, mut)
                if s is not None:
                    out[(doc.name, ln, tok, mut)] = s
    return out


def score(on: dict, off: dict) -> dict:
    keys = sorted(set(on) & set(off))
    fv_on = [k for k in keys if on[k] == "VERIFIED"]
    fv_off = [k for k in keys if off[k] == "VERIFIED"]
    caught = [k for k in fv_off if on[k] == "UNGROUNDED"]
    silenced = [k for k in fv_off if on[k] == "ABSTAIN"]
    other = [k for k in fv_off if on[k] not in ("VERIFIED", "UNGROUNDED", "ABSTAIN")]
    inverted = [k for k in fv_on if off[k] != "VERIFIED"]
    return {"scored": len(keys),
            "false_verified_off": len(fv_off), "false_verified_on": len(fv_on),
            "removed": len(fv_off) - len(fv_on),
            "caught_UNGROUNDED": len(caught), "silenced_ABSTAIN": len(silenced),
            "other_terminus": len(other),
            "inverted_on_verified_off_not": len(inverted),
            "status_on": dict(Counter(on[k] for k in keys)),
            "status_off": dict(Counter(off[k] for k in keys)),
            "_caught_keys": [list(k) for k in caught[:400]]}


# ---------------------------------------------------------------- D. right-demotion dossier

_RVAL_CACHE: dict[str, list[tuple[str, str, float]]] = {}


def demotion_evidence(doc: Path, receipts, entry: dict, arm_on_status: str, idx: int) -> dict:
    """Machine evidence for one clean-corpus demotion, replicating certify_doc's own context."""
    text = doc.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    ctx = lines[entry["line"] - 1].strip().replace("−", "-")
    bctx = entry.get("binding_context", ctx)
    ck = doc.name
    if ck not in _RVAL_CACHE:
        _RVAL_CACHE[ck] = rvals_for(receipts)
    rvals = _RVAL_CACHE[ck]
    allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
    hits = [(rn, pth) for rn, pth, rv in rvals
            if ON._match(entry["value"], entry["decimals"], rv, allow_scaling)]
    stems = _ctx_stems(bctx) | {d for d in re.findall(r"\d{2,}", bctx)}
    all_stems: set[str] = set()
    for _rn, pth, _rv in rvals:
        all_stems |= _path_stems(pth)
    nameable = bool(all_stems & stems)
    paths = [f"{rn}:{pth}" for rn, pth in hits]
    countlike = [p for p in paths
                 if re.search(r"(^|[._\[])n_|n_held|n_caved|^n(\.|$)|count", p, re.I)]
    bulk = [p for p in paths if "[" in p]
    tok = entry["token"]
    table_ordinal = bool(re.match(r"^\|\s*\**" + re.escape(tok) + r"\**\s*\|", ctx))
    spec_like = bool(re.search(r"\b(bar|gate|threshold|floor|ceiling|requires?|must|"
                               r"pre-?registered|at least|at most|budget|cap)\b", bctx, re.I))
    if table_ordinal:
        cls = "TABLE_ORDINAL"
    elif not nameable:
        cls = "UNBINDABLE"
    elif spec_like:
        cls = "SPEC_LIKE"
    elif paths and len(countlike) == len(paths):
        cls = "CROSS_EXPERIMENT_COUNT"
    elif paths and len(bulk) == len(paths):
        cls = "ALL_HITS_BULK_ARRAY"
    else:
        cls = "OTHER"
    return {"id": f"{doc.name}|L{entry['line']}|{tok}|#{idx}",
            "doc": doc.name, "line": entry["line"], "token": tok,
            "value": entry["value"], "on_status": arm_on_status,
            "n_hits": len(hits), "hit_paths": paths[:10],
            "n_countlike": len(countlike), "n_bulk": len(bulk),
            "nameable": nameable, "table_ordinal": table_ordinal, "spec_like": spec_like,
            "machine_class": cls, "context": ctx[:190], "binding_context": bctx[:190]}


# ---------------------------------------------------------------- main

def main() -> int:
    t0 = time.time()
    docs = resolvable_docs()
    print(f"docs with fully-resolvable receipts: {len(docs)}")
    print(f"arms: ON = shipped certify.py ({CERT_PY}); OFF = same source, count-binding "
          f"short-circuited\n", flush=True)

    # ---- A. provenance
    print("A. PROVENANCE -- D1 battery replay (validate_oath_v0, seed 1), both arms")
    d1 = d1_replay()
    print(f"   D1 caught: ON {d1['caught_on']}/20   OFF {d1['caught_off']}/20   "
          f"(pre-registered bar 16)")
    print(f"   rows where the filter is decisive: {len(d1['filter_decisive_rows'])}")
    print("\nA. PROVENANCE -- the cited 27->37 case, driven deterministically")
    prov = provenance_probe()
    print(f"   '27' claims probed: {prov['n_claims']}   filter decisive on "
          f"{prov['n_filter_decisive']}")
    print("\nA. PROVENANCE -- did the rule EVER catch it? 27->37 at the v0.3 commit and its parent")
    hist = historical_probe()
    print(f"   ever caught: {hist.get('ever_caught')}\n", flush=True)

    # ---- C. clean corpus, both arms
    print("C. CLEAN CORPUS -- both arms", flush=True)
    led_on, ver_on = corpus_pass(ON, docs)
    led_off, ver_off = corpus_pass(OFF, docs)
    misaligned = [k for k in led_off
                  if k in led_on and (led_on[k]["line"], led_on[k]["token"])
                  != (led_off[k]["line"], led_off[k]["token"])]
    if misaligned:
        raise SystemExit(f"FATAL: {len(misaligned)} ledger rows misaligned between arms — "
                         "extraction differs and the comparison is void.")
    int_keys = [k for k, e in led_off.items() if e["decimals"] == 0 and k in led_on]
    trans = Counter()
    demoted = []
    for k in int_keys:
        a, b = led_on[k], led_off[k]
        if a["status"] != b["status"]:
            trans[f"{b['status']}->{a['status']}"] += 1
            if b["status"] == "VERIFIED":
                demoted.append(k)
    ver_int_on = sum(1 for k in int_keys if led_on[k]["status"] == "VERIFIED")
    ver_int_off = sum(1 for k in int_keys if led_off[k]["status"] == "VERIFIED")
    manufactured_ung = [k for k in int_keys
                        if led_on[k]["status"] == "UNGROUNDED"
                        and led_off[k]["status"] != "UNGROUNDED"]
    flipped = [d for d in ver_off if ver_on.get(d) != ver_off.get(d)]
    print(f"   integer claims: VERIFIED OFF {ver_int_off} -> ON {ver_int_on}  "
          f"(cost {ver_int_off - ver_int_on})")
    print(f"   transitions OFF->ON: {dict(trans)}")
    print(f"   UNGROUNDED manufactured by the filter on committed documents: "
          f"{len(manufactured_ung)}")
    print(f"   certificates whose verdict differs between arms: {len(flipped)} {flipped[:4]}\n",
          flush=True)

    # ---- B. benefit
    frame: dict[str, list[tuple[int, str]]] = {}
    on_verified_set = set()
    seen = set()
    for k in int_keys:
        rel, _i = k
        name = Path(rel).name
        e = led_off[k]
        if e["status"] != "VERIFIED":
            continue
        sig = (name, e["line"], e["token"])
        if sig in seen:          # a token repeated on one line yields one mutant, not two
            continue
        seen.add(sig)
        frame.setdefault(name, []).append((e["line"], e["token"]))
        if led_on[k]["status"] == "VERIFIED":
            on_verified_set.add(sig)
    frame_n = sum(len(v) for v in frame.values())
    exh = {(n, ln, tok): all_mutants(tok) for n, v in frame.items() for ln, tok in v}
    exh_total = sum(len(v) for v in exh.values())
    print(f"B. BENEFIT -- frame: {frame_n} integer claims VERIFIED by the no-filter arm "
          f"({len(on_verified_set)} of them also VERIFIED by the shipped arm)")
    print(f"   exhaustive single-digit mutants: {exh_total}", flush=True)
    if exh_total > EXHAUSTIVE_CAP:
        rng = random.Random(0)
        for kk in exh:
            exh[kk] = sorted(rng.sample(exh[kk], max(1, int(len(exh[kk]) * EXHAUSTIVE_CAP
                                                           / exh_total))))
        exh_total = sum(len(v) for v in exh.values())
        print(f"   capped -> {exh_total}", flush=True)

    def exh_mut(name, ln, tok):
        return exh.get((name, ln, tok), [])

    ckey = hashlib.sha256(
        (hashlib.sha256(CERT_PY.read_bytes()).hexdigest()
         + json.dumps(sorted((n, ln, tok, tuple(exh[(n, ln, tok)]))
                             for n, v in frame.items() for ln, tok in v))
         ).encode("utf-8")).hexdigest()[:16]
    print("   exhaustive census, ON arm ...", flush=True)
    e_on = cached("exh_on", ckey, lambda: census(docs, frame, exh_mut, ON))
    print(f"   exhaustive census, OFF arm ... ({time.time()-t0:.0f}s)", flush=True)
    e_off = cached("exh_off", ckey, lambda: census(docs, frame, exh_mut, OFF))
    exhaustive = score(e_on, e_off)
    exhaustive["mutants_per_claim_mean"] = round(exh_total / max(frame_n, 1), 2)
    sub = {k: v for k, v in e_on.items() if (k[0], k[1], k[2]) in on_verified_set}
    sub_off = {k: v for k, v in e_off.items() if k in sub}
    exhaustive_shipped_frame = score(sub, sub_off)
    print(f"   exhaustive: false-VERIFIED  OFF {exhaustive['false_verified_off']}  "
          f"ON {exhaustive['false_verified_on']}  removed {exhaustive['removed']}  "
          f"(caught {exhaustive['caught_UNGROUNDED']} / silenced "
          f"{exhaustive['silenced_ABSTAIN']})", flush=True)

    seeded = {}
    for sd in SEEDS:
        rng = random.Random(sd)
        picks = {}
        for name, v in sorted(frame.items()):
            for ln, tok in v:
                picks[(name, ln, tok)] = [mutate_token(tok, rng)]
        pick = picks.get
        s_on = cached(f"sd{sd}_on", ckey, lambda: census(docs, frame,
                                                         lambda n, l, t: pick((n, l, t), []), ON))
        s_off = cached(f"sd{sd}_off", ckey, lambda: census(docs, frame,
                                                           lambda n, l, t: pick((n, l, t), []), OFF))
        seeded[str(sd)] = score(s_on, s_off)
        seeded[str(sd)].pop("_caught_keys", None)
        print(f"   seed {sd}: false-VERIFIED OFF {seeded[str(sd)]['false_verified_off']} "
              f"ON {seeded[str(sd)]['false_verified_on']} removed {seeded[str(sd)]['removed']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- D. right-demotion dossier
    print("\nD. DEMOTION DOSSIER", flush=True)
    doc_index = {d.relative_to(ROOT).as_posix(): (d, r) for d, r in docs}
    dossier = []
    for k in sorted(demoted):
        rel, i = k
        d, r = doc_index[rel]
        dossier.append(demotion_evidence(d, r, led_off[k], led_on[k]["status"], i))
    classes = Counter(x["machine_class"] for x in dossier)
    print(f"   {len(dossier)} demotions; machine classes {dict(classes)}")
    rng = random.Random(ADJ_SEED)
    sample = rng.sample(sorted(dossier, key=lambda x: x["id"]), min(ADJ_N, len(dossier)))
    hand = Counter(HAND_ADJUDICATION.get(x["id"], "UNSCORED") for x in sample)
    print(f"   seeded sample n={len(sample)} (seed {ADJ_SEED}); hand: {dict(hand)}", flush=True)

    # ---- E. ratio and verdict
    benefit = exhaustive["removed"]
    cost = ver_int_off - ver_int_on
    ratio = benefit / max(cost, 1)
    seeded_mean = round(sum(seeded[str(s)]["removed"] for s in SEEDS) / len(SEEDS), 1)
    right = sum(v for kk, v in hand.items() if kk in (_C, _S, _O))
    genuine = hand.get(_G, 0)
    scored = right + genuine

    report = {
        "note": "RED TEAM audit of the SHIPPED v0.3 integer count-binding filter in "
                "styxx/certify.py. Adversarial brief: try to refute the premise that the rule is "
                "costly and weakly discriminating. MEASUREMENT ONLY -- certify.py is not edited; "
                "the OFF arm is its own source with the one guard line short-circuited.",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verifier_sha256": hashlib.sha256(CERT_PY.read_bytes()).hexdigest(),
        "docs": len(docs),
        "A_provenance": {
            "cited_justification": "the k=14-class D1 misses: 27->37 'verified' because a shared "
                                   "addendum carries another experiment's n_held=37",
            "d1_replay": d1,
            "targeted_27_to_37": prov,
            "historical_27_to_37_at_the_v03_commit": hist,
        },
        "B_benefit": {
            "frame_integer_claims_verified_no_filter": frame_n,
            "frame_also_verified_by_shipped": len(on_verified_set),
            "exhaustive": exhaustive,
            "exhaustive_shipped_frame_only": exhaustive_shipped_frame,
            "seeded": seeded,
            "seeded_removed_mean": seeded_mean,
        },
        "C_cost": {
            "integer_claims_verified_no_filter": ver_int_off,
            "integer_claims_verified_shipped": ver_int_on,
            "demoted_by_filter": cost,
            "transitions": dict(trans),
            "ungrounded_manufactured_on_committed_docs": len(manufactured_ung),
            "ungrounded_manufactured_rows": [
                {"doc": Path(k[0]).name, "line": led_on[k]["line"],
                 "token": led_on[k]["token"], "status_without_filter": led_off[k]["status"],
                 "context": led_on[k]["context"][:170]} for k in manufactured_ung],
            "verdict_differs_between_arms": flipped,
        },
        "D_right_demotions": {
            "n_demotions": len(dossier),
            "machine_classes": dict(classes),
            "adjudication_seed": ADJ_SEED,
            "adjudication_n": len(sample),
            "hand_scores": dict(hand),
            "hand_right": right, "hand_genuine_destroyed": genuine,
            "hand_right_share": round(right / max(scored, 1), 4),
            "v08_float_comparison": {"note": "the SAME predicate at status level on FLOAT claims "
                                             "scored 30 of 40 GENUINE-BINDING-DESTROYED and was "
                                             "killed V08_COVERAGE_DESTRUCTIVE (bar <= 12 of 40)",
                                     "v08_genuine_destroyed_of_40": 30,
                                     "v09_integer_genuine_destroyed_of_40": genuine,
                                     "v08_bar": 12,
                                     "integer_rule_would_clear_the_v08_bar": genuine <= 12},
            "extrapolated_genuine_verifications_destroyed": round(
                len(dossier) * genuine / max(scored, 1), 1),
            "sample_ids": [x["id"] for x in sample],
            "rows": dossier,
        },
        "E_verdict": {
            "benefit_exhaustive_false_attestations_removed": benefit,
            "benefit_exhaustive_of_which_CAUGHT": exhaustive["caught_UNGROUNDED"],
            "benefit_shipped_frame_removed": exhaustive_shipped_frame["removed"],
            "benefit_shipped_frame_CAUGHT": exhaustive_shipped_frame["caught_UNGROUNDED"],
            "cost_clean_verifications_destroyed": cost,
            "benefit_per_cost_RAW_UNIT_MISMATCHED": round(ratio, 3),
            "benefit_per_cost_raw_note":
                "DISCLOSED: this divides MUTANTS removed by CLAIMS destroyed. The exhaustive census "
                "puts ~17 mutants on every claim, so the numerator scales with the operator and the "
                "denominator does not. It is NOT comparable to the v0.8 cost/kill table and must "
                "not be quoted as if it were.",
            "unit_matched_one_mutation_per_claim": {
                "note": "the repo's own convention (run_oath_v08_battery: one sampled mutation per "
                        "frame claim), which is what v0.8's cost/kill ratios were computed at",
                "false_attestations_removed_mean_over_seeds": seeded_mean,
                "seeds": list(SEEDS),
                "clean_verifications_destroyed": cost,
                "cost_per_false_attestation_removed": round(cost / max(seeded_mean, 1), 3),
                "v08_float_clause_cost_per_kill_for_comparison": 1.056,
                "reading": "at the repo's own unit the integer rule sits at parity, exactly where "
                           "every float design family sat. What separates them is not the ratio, "
                           "it is whether the demotions are RIGHT (D above).",
            },
            "accusing_terminus": {
                "note": "the integer filter is NOT demote-only: an emptied hit set on an obligated "
                        "claim yields UNGROUNDED. This is what it buys and what it costs.",
                "mutant_catches_it_produces": exhaustive["caught_UNGROUNDED"],
                "false_accusations_on_clean_committed_docs": len(manufactured_ung),
                "documents_flipped_OATH_HELD_to_FAILED": len(flipped),
            },
            "false_attestation_rate_per_mutant": {
                "without_filter": round(exhaustive["false_verified_off"]
                                        / max(exhaustive["scored"], 1), 4),
                "with_filter": round(exhaustive["false_verified_on"]
                                     / max(exhaustive["scored"], 1), 4)},
            "residual_false_attestations_with_filter": exhaustive["false_verified_on"],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'quantity':52s} {'value':>10s}")
    print("-" * 64)
    print(f"{'BENEFIT  false attestations removed (exhaustive)':52s} {benefit:>10d}")
    print(f"{'   of which CAUGHT (-> UNGROUNDED)':52s} "
          f"{exhaustive['caught_UNGROUNDED']:>10d}")
    print(f"{'   of which SILENCED (-> ABSTAIN)':52s} "
          f"{exhaustive['silenced_ABSTAIN']:>10d}")
    print(f"{'BENEFIT  removed on the SHIPPED attested frame':52s} "
          f"{exhaustive_shipped_frame['removed']:>10d}")
    print(f"{'BENEFIT  1 mutation/claim (repo convention, mean)':52s} {seeded_mean:>10.1f}")
    print(f"{'COST     clean integer verifications destroyed':52s} {cost:>10d}")
    print(f"{'   of which hand-scored GENUINE (10/40 -> est.)':52s} "
          f"{report['D_right_demotions']['extrapolated_genuine_verifications_destroyed']:>10.1f}")
    print(f"{'RATIO    mutants removed / claims destroyed (raw)':52s} {ratio:>10.3f}")
    print(f"{'RATIO    cost per kill at 1 mutation/claim (v0.8 unit)':52s} "
          f"{cost / max(seeded_mean, 1):>10.3f}")
    print(f"{'ACCUSE   mutant catches / false accusations on clean':52s} "
          f"{str(exhaustive['caught_UNGROUNDED']) + ' / ' + str(len(manufactured_ung)):>10s}")
    print(f"{'RESIDUAL false attestations the filter still allows':52s} "
          f"{exhaustive['false_verified_on']:>10d}")
    print(f"\nelapsed {report['elapsed_s']}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
