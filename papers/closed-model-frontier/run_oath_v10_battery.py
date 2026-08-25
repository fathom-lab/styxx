"""OATH v0.10 battery — G0..G6 per PREREG_oath_v10_token_column_2026_08_23.

Under test: two severable clauses in ``styxx.certify``.

  V10_TOKEN_COLUMN           `extract_numbers` records the column its match was found at and
                             `certify_doc` anchors `pre`/`post` there, instead of re-finding the
                             token STRING with `ctx.find` and landing on the first occurrence.
                             (primary)
  V10_SLASHPAIR_RANGE_GUARD  the v0.3 range-sanity rule does not fire on a slash-pair numerator.
                             (companion; exists for the one latent false accusation the primary
                             un-masks, and provably carries no behaviour of its own)

Every gate runs in BOTH arms on the identical sample with the identical seed. G1's ON arm must
strictly improve on its OFF arm or the run is VOID — a battery reading the same number in both
arms is not measuring the repair.

G6 is declared NO-CREDIT in the prereg: this cycle claims no tamper improvement and gates only
against a tamper REGRESSION. Its false-attestation column RISES and is published as a cost.

Non-destructive: mutants live in temp files, corpus passes are in-memory, and the only file
written is this battery's own result JSON.

  python papers/closed-model-frontier/run_oath_v10_battery.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import importlib                                                           # noqa: E402

# importlib, not `import styxx.certify as C`: the package attribute `styxx.certify` is the
# provenance FUNCTION by convention, and `import a.b as c` resolves getattr(a, 'b') BEFORE
# sys.modules. The plain form binds the function whenever that attribute has been touched, and
# every flag write in this harness would then land on the wrong object -- both arms would silently
# run the shipped verifier and the battery would report a real-looking null.
C = importlib.import_module("styxx.certify")
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

CENSUS = HERE / "oath_v10_column_census.json"
BASELINE = HERE / "oath_v10_baseline_ledger.json"
OUT = HERE / "oath_v10_battery_result.json"

G1_BAR_ON, G1_BAR_OFF = 0, 349
G2_BAR = 0
G3_BAR = 0
G4A_CORRECT_BAR, G4A_WRONG_BAR, G4B_BAR = 20, 0, 10
G5_BAR = 0
G6A_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
G6B_SEEDS = (1, 2, 3, 4, 5)
G6B_TOLERANCE = 10

# Frozen shadow-sweep expectations (PREREG "Disclosed method"). G0 demands the real verifier
# reproduce them exactly; a disagreement means the shipped edit is not the swept edit.
SWEEP = {
    "col=off,guard=off": {"diffs": 0, "flips": 0},
    "col=off,guard=on": {"diffs": 0, "flips": 0},
    "col=on,guard=off": {"diffs": 45, "flips": 1},
    "col=on,guard=on": {"diffs": 44, "flips": 0},
}
SWEEP_SPLIT = {"ABSTAIN->VERIFIED": 27, "VERIFIED->ABSTAIN": 17}

# ---- hand adjudication of all 44 transitions against the prereg's frozen definition, as amended
# there BEFORE this run. VERIFIED->ABSTAIN: DESTRUCTIVE iff the token is a MEASUREMENT whose
# shipped receipt_ref named a leaf genuinely holding it; CORRECT iff the abstention is what the
# shipped rules prescribe for the text that actually surrounds it. ABSTAIN->VERIFIED: CORRECT iff
# a measurement grounded in a related leaf, or a bar grounded in a leaf that names it as a bar;
# QUESTIONABLE iff grounded in an unrelated leaf (the v0.4 coincidence channel, closed NEGATIVE by
# v0.8); WRONG iff a notation artifact or historical quotation is now being sworn to. Ties resolve
# against the change. There is no sample and no seed: 44 is small enough to score exhaustively.
ADJUDICATION = {
    # --- VERIFIED -> ABSTAIN, CORRECT (8) -------------------------------------------------
    # `S_frame@20` -- v0.5 class D @-glued parameter; the shipped VERIFIED read the OTHER `20`.
    ("FINDING_combined_signal_2026_07_26.md", 26, "20", 13): "CORRECT",
    ("FINDING_combined_signal_2026_07_26.md", 89, "20", 60): "CORRECT",
    # "against a 0.05 bar" -- a BAR; the shipped verifier was swearing it to a coincidental leaf.
    ("FINDING_scale_channel_2026_07_24.md", 30, "0.05", 15): "CORRECT",
    # "Cycle 74" -- a cycle ordinal, not a measurement. The shipped VERIFIED read the `74` in
    # "cycles 74/75" and grounded it in mmlu_mc_cot.n_correct.
    ("FINDING_self_verification_2026_07_25.md", 94, "74", 52): "CORRECT",
    # both inside "originally printed [...]" -- is_hist. The LIVE copies of these numbers earlier
    # on the same line keep their VERIFIED status; only the quoted historical ones move.
    ("FINDING_b22_nonacknowledged_caving_2026_06_09.md", 24, "0.500", 28): "CORRECT",
    ("FINDING_b22_nonacknowledged_caving_2026_06_09.md", 24, "0.284", 33): "CORRECT",
    # `V07_PRECISION_DIGITS` = 7 -- a frozen flag value, i.e. a spec constant.
    ("RESULT_oath_v07_SHIPPED_2026_08_22.md", 30, "7", 9): "CORRECT",
    # "| **0.80** | >= 0.80 | PASS |" -- the BAR column; the shipped verifier read the observed
    # column's text and swore the bar to ref_knowledge_acc.
    ("RESULT_foundation_2026_07_04.md", 16, "0.80", 7): "CORRECT",
    # --- VERIFIED -> ABSTAIN, DESTRUCTIVE (9) ---------------------------------------------
    # every one of these has `pre` ending in a bare `=` (gate G4c checks it mechanically):
    # V10_EQUALS_SPEC_OVERREACH, the named residual handed to a successor prereg.
    ("FINDING_conscience_loop_2026_07_24.md", 60, "0.0854", 47): "DESTRUCTIVE",
    ("FINDING_protocol_metric_identity_2026_08_07.md", 21, "1", 7): "DESTRUCTIVE",
    ("RESULT_B2_adaptive_erasure_SURVIVES_2026_07_13.md", 7, "5", 2): "DESTRUCTIVE",
    ("RESULT_B2_coupling_confirm_VOID_2026_07_16.md", 13, "5", 0): "DESTRUCTIVE",
    ("RESULT_honesty_parity_confirm_2026_07_11.md", 48, "1", 87): "DESTRUCTIVE",
    ("FINDING_b22_nonacknowledged_caving_2026_06_09.md", 75, "1.0", 65): "DESTRUCTIVE",
    ("FINDING_b34v3_labelfree_read_2026_08_03.md", 42, "2", 31): "DESTRUCTIVE",
    ("FINDING_b42_dose_curve_2026_08_05.md", 41, "1", 42): "DESTRUCTIVE",
    ("FINDING_portable_conscience_v0_2026_06_10.md", 29, "1.000", 11): "DESTRUCTIVE",
    # --- ABSTAIN -> VERIFIED, CORRECT (23) ------------------------------------------------
    # a bar grounded in the leaf that RECORDS it as a bar.
    ("FINDING_verifier_at_7b_2026_07_27.md", 22, "0.75", 5): "CORRECT",
    ("FINDING_p1_third_quarantine_2026_08_08.md", 26, "0.05", 9): "CORRECT",
    # the OBSERVED column of a markdown gate table. The shipped windows anchored every token on
    # the row at the BAR, so the measured result was abstained as if it were its own threshold --
    # certification by omission, on the exact column a gate table exists to report.
    ("FINDING_protocol_metric_identity_2026_08_07.md", 11, "0", 1): "CORRECT",
    ("FINDING_protocol_metric_identity_2026_08_07.md", 12, "1", 3): "CORRECT",
    ("FINDING_protocol_metric_identity_2026_08_07.md", 13, "1", 5): "CORRECT",
    ("FINDING_protocol_power_basis_invalid_2026_08_07.md", 12, "1", 5): "CORRECT",
    ("FINDING_protocol_power_basis_invalid_2026_08_07.md", 13, "0", 7): "CORRECT",
    ("FINDING_b45_frame_geometry_2026_08_06.md", 14, "5", 4): "CORRECT",
    ("FINDING_b48_invalid_null_bar_2026_08_06.md", 20, "45", 4): "CORRECT",
    ("FINDING_b50_no_legibility_islands_2026_08_08.md", 12, "45", 1): "CORRECT",
    ("FINDING_c1_instrument_blind_to_isc_2026_08_06.md", 13, "21", 1): "CORRECT",
    ("FINDING_h1a_human_single_clique_2026_08_06.md", 13, "8", 1): "CORRECT",
    ("FINDING_h1b_no_unreadable_minds_2026_08_06.md", 13, "8", 2): "CORRECT",
    ("RESULT_E3_adaptive_STANDS_2026_07_04.md", 21, "1", 20): "CORRECT",
    # seed identifiers grounded in the receipt's own seeds array.
    ("RESULT_attack_sentiment_r64_2026_07_09.md", 28, "0", 39): "CORRECT",
    ("RESULT_B2_subspace_erasure_SURVIVES_2026_07_12.md", 75, "1", 88): "CORRECT",
    ("RESULT_E2_strong_attacker_2026_07_07.md", 65, "1", 49): "CORRECT",
    ("RESULT_E3PRIME_bite_verification_2026_07_07.md", 60, "1", 35): "CORRECT",
    # measurements grounded in the leaf that names them.
    ("RESULT_honesty_parity_confirm_llama_2026_07_12.md", 45, "0", 75): "CORRECT",
    ("FINDINGS_rhythm_substrate_2026_06_03.md", 40, "20", 28): "CORRECT",
    ("FINDING_fame_vs_truth_2026_05_25.md", 18, "1.00", 8): "CORRECT",
    ("FINDING_fame_vs_truth_2026_05_25.md", 18, "1.00", 12): "CORRECT",
    ("FINDING_p1_third_quarantine_2026_08_08.md", 84, "24", 49): "CORRECT",
    # --- ABSTAIN -> VERIFIED, QUESTIONABLE (4) --------------------------------------------
    # bars grounded in leaves that do NOT name them as bars, and two counts grounded in leaves
    # whose path only loosely relates: the standing v0.4 coincidence channel, credited to nobody.
    ("RESULT_attack_sentiment_r64_2026_07_09.md", 28, "0.65", 40): "QUESTIONABLE",
    ("RESULT_attack_sentiment_wholestack_2026_07_09.md", 28, "0.60", 34): "QUESTIONABLE",
    ("FINDING_b45_frame_geometry_2026_08_06.md", 14, "5", 3): "QUESTIONABLE",
    ("FINDING_b24_whitebox_vs_behavioral_2026_06_09.md", 45, "0", 33): "QUESTIONABLE",
}


def set_flags(col: bool, guard: bool) -> None:
    C.V10_TOKEN_COLUMN = col
    C.V10_SLASHPAIR_RANGE_GUARD = guard


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
        except Exception:                                        # pragma: no cover - defensive
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def corpus_pass(docs, col: bool, guard: bool) -> dict:
    set_flags(col, guard)
    ledger, verdicts, entries = {}, {}, {}
    for doc, receipts in docs:
        try:
            cert = C.certify_doc(doc, receipts)
        except Exception:                                        # pragma: no cover - defensive
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        for i, e in enumerate(cert["ledger"]):
            ledger[f"{rel}|L{e['line']}|{e['token']}|#{i}"] = e["status"]
            entries[(rel, i)] = e
    return {"ledger": ledger, "verdicts": verdicts, "entries": entries}


def diff_vs_baseline(passed, base) -> tuple[int, int, list]:
    keys = set(passed["ledger"]) | set(base["ledger"])
    diffs = sorted(k for k in keys if passed["ledger"].get(k) != base["ledger"].get(k))
    flips = sorted(d for d, v in base["verdicts"].items() if passed["verdicts"].get(d) != v)
    return len(diffs), len(flips), diffs


def anchoring(docs) -> dict:
    """G1, both arms in one pass: where does each arm anchor, versus where the token IS?

    The naive check "does `ctx[at:at+len]` equal the token" cannot fail for the OFF arm — `str.find`
    only ever returns a position where that string is present. The defect is not that the OFF arm
    lands on a non-token, it is that it lands on a DIFFERENT token, so the comparison has to be
    against the extraction column, which is what the ON arm records. The ON arm's number is an
    identity (invariant I2) and is reported as one; the OFF arm's is the measurement.
    """
    out = {}
    for arm in ("on", "off"):
        set_flags(arm == "on", arm == "on")
        misplaced, total, not_on_token, examples = 0, 0, 0, []
        for doc, _rc in docs:
            text = doc.read_text(encoding="utf-8")
            lines = text.splitlines()
            set_flags(True, True)
            true_cols = [e["col"] for e in C.extract_numbers(text)]
            set_flags(arm == "on", arm == "on")
            for num, tcol in zip(C.extract_numbers(text), true_cols):
                raw = lines[num["line"] - 1].replace("−", "-")
                lead = len(raw) - len(raw.lstrip())
                ctx = raw.strip()
                at = (num["col"] - lead) if "col" in num else ctx.find(num["token"])
                total += 1
                not_on_token += ctx[at:at + len(num["token"])] != num["token"]
                if at != tcol - lead:
                    misplaced += 1
                    if len(examples) < 20:
                        examples.append(f"{doc.name}|L{num['line']}|{num['token']}|"
                                        f"at={at}|true={tcol - lead}")
        out[arm] = {"tokens": total, "misplaced": misplaced,
                    "anchor_not_on_the_token_string": not_on_token, "examples": examples}
    return out


def extraction_signature(paths, col: bool, guard: bool) -> dict:
    """G2: the ordered (line, token) list per document. The repair must not move extraction."""
    set_flags(col, guard)
    sig = {}
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:                                        # pragma: no cover - defensive
            continue
        sig[p.relative_to(ROOT).as_posix()] = [(e["line"], e["token"])
                                               for e in C.extract_numbers(text)]
    return sig


def mutate_sig(tok: str, rng: random.Random) -> str:
    if "." in tok:
        frac_at = tok.index(".") + 1
        frac = tok[frac_at:]
        sig = [i for i in range(min(6, len(frac)))
               if not (frac[i] == "0" and set(frac[:i]) <= {"0"})]
        pos = frac_at + rng.choice(sig or [0])
    else:
        pos = rng.choice([i for i, ch in enumerate(tok) if ch.isdigit()])
    old = int(tok[pos])
    return tok[:pos] + str(rng.choice([d for d in range(10) if d != old])) + tok[pos + 1:]


def _certify_mutant(doc_lines, line_no, new_line, receipts, idx, mut, col, guard):
    set_flags(col, guard)
    ml = list(doc_lines)
    ml[line_no - 1] = new_line
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(ml))
        tmp = Path(tf.name)
    try:
        led = C.certify_doc(tmp, receipts)["ledger"]
    finally:
        tmp.unlink(missing_ok=True)
    if idx < len(led) and led[idx]["token"] == mut:
        return led[idx]["status"]
    e = next((x for x in led if x["line"] == line_no and x["token"] == mut), None)
    return e["status"] if e else "NOT_EXTRACTED"


def token_columns(doc: Path) -> list[dict]:
    """Every extracted token with its true column, read off the verifier with the flag ON."""
    set_flags(True, True)
    text = doc.read_text(encoding="utf-8")
    return list(C.extract_numbers(text))


def g6_collision(docs, base) -> dict:
    """G6a — the instance-2 channel: mutants that CREATE a collision the clean document lacked.

    Substitution is at the token's KNOWN column. A first-occurrence `line.replace(tok, mut, 1)` --
    what run_oath_v07/v09_battery.py and corpus_audit.audit_document both do -- lands on the wrong
    occurrence for exactly the population under test and would make this leg meaningless.
    """
    pop = []
    for doc, rc in docs:
        rel = doc.relative_to(ROOT).as_posix()
        for i, e in enumerate(token_columns(doc)):
            if base["ledger"].get(f"{rel}|L{e['line']}|{e['token']}|#{i}") == "VERIFIED":
                pop.append({"doc": doc, "receipts": rc, "rel": rel, "line": e["line"],
                            "token": e["token"], "idx": i, "col": e["col"]})
    cache = {d.relative_to(ROOT).as_posix(): d.read_text(encoding="utf-8").splitlines()
             for d, _ in docs}
    per_seed, pooled = [], collections.Counter()
    for seed in G6A_SEEDS:
        rng = random.Random(seed)
        collide = []
        for r in pop:
            mut = mutate_sig(r["token"], rng)
            lines = cache[r["rel"]]
            norm = lines[r["line"] - 1].replace("−", "-")
            if norm[r["col"]:r["col"] + len(r["token"])] != r["token"]:
                continue
            new_line = norm[:r["col"]] + mut + norm[r["col"] + len(r["token"]):]
            lead = len(new_line) - len(new_line.lstrip())
            clean = norm.strip()
            if new_line.strip().find(mut) != r["col"] - lead \
                    and clean.find(r["token"]) == r["col"] - (len(norm) - len(norm.lstrip())):
                collide.append((r, mut, lines, new_line))
        off, on = collections.Counter(), collections.Counter()
        for r, mut, lines, new_line in collide:
            off[_certify_mutant(lines, r["line"], new_line, r["receipts"], r["idx"], mut,
                                False, False)] += 1
            on[_certify_mutant(lines, r["line"], new_line, r["receipts"], r["idx"], mut,
                               True, True)] += 1
        per_seed.append({"seed": seed, "n_collide": len(collide),
                         "off": dict(off), "on": dict(on)})
        pooled["n"] += len(collide)
        pooled["off_caught"] += off["UNGROUNDED"]
        pooled["on_caught"] += on["UNGROUNDED"]
        pooled["off_false"] += off["VERIFIED"]
        pooled["on_false"] += on["VERIFIED"]
        pooled["off_abstain"] += off["ABSTAIN"]
        pooled["on_abstain"] += on["ABSTAIN"]
        print(f"  seed {seed:<3d} collisions {len(collide):3d}   caught OFF {off['UNGROUNDED']:3d} "
              f"-> ON {on['UNGROUNDED']:3d}   false-attested OFF {off['VERIFIED']:3d} -> ON "
              f"{on['VERIFIED']:3d}   abstained OFF {off['ABSTAIN']:3d} -> ON {on['ABSTAIN']:3d}",
              flush=True)
    return {"per_seed": per_seed, "pooled": dict(pooled),
            "pass": pooled["on_caught"] >= pooled["off_caught"]}


def g6_clean(docs, census) -> dict:
    """G6b — the misplaced-token roster from the clean corpus. NO-CREDIT: regression gate only."""
    by_rel = {d.relative_to(ROOT).as_posix(): (d, rc) for d, rc in docs}
    roster = [r for r in census["certified_corpus"]["roster"] if r["rel"] in by_rel]
    cache = {rel: by_rel[rel][0].read_text(encoding="utf-8").splitlines() for rel in by_rel}
    per_seed, pooled = [], collections.Counter()
    for seed in G6B_SEEDS:
        rng = random.Random(seed)
        off, on = collections.Counter(), collections.Counter()
        for r in roster:
            doc, rc = by_rel[r["rel"]]
            lines = cache[r["rel"]]
            norm = lines[r["line"] - 1].replace("−", "-")
            lead = len(norm) - len(norm.lstrip())
            col = r["true_at"] + lead
            mut = mutate_sig(r["token"], rng)
            if norm[col:col + len(r["token"])] != r["token"]:
                off["DID_NOT_LAND"] += 1
                on["DID_NOT_LAND"] += 1
                continue
            new_line = norm[:col] + mut + norm[col + len(r["token"]):]
            off[_certify_mutant(lines, r["line"], new_line, rc, r["ledger_index"], mut,
                                False, False)] += 1
            on[_certify_mutant(lines, r["line"], new_line, rc, r["ledger_index"], mut,
                               True, True)] += 1
        per_seed.append({"seed": seed, "off": dict(off), "on": dict(on)})
        pooled["off_caught"] += off["UNGROUNDED"]
        pooled["on_caught"] += on["UNGROUNDED"]
        pooled["off_false"] += off["VERIFIED"]
        pooled["on_false"] += on["VERIFIED"]
        print(f"  seed {seed:<3d} caught OFF {off['UNGROUNDED']:3d} -> ON {on['UNGROUNDED']:3d}   "
              f"false-attested OFF {off['VERIFIED']:3d} -> ON {on['VERIFIED']:3d}", flush=True)
    delta = abs(pooled["on_caught"] - pooled["off_caught"])
    return {"n_roster": len(roster), "per_seed": per_seed, "pooled": dict(pooled),
            "pooled_abs_delta": delta, "tolerance": G6B_TOLERANCE,
            "pass": delta <= G6B_TOLERANCE}


def main() -> int:                                                   # noqa: C901 - one report
    t0 = time.time()
    for flag in ("V10_TOKEN_COLUMN", "V10_SLASHPAIR_RANGE_GUARD"):
        if not hasattr(C, flag):
            print(f"FATAL: styxx.certify has no {flag} — nothing to test.")
            return 2
    for p in (CENSUS, BASELINE):
        if not p.exists():
            print(f"FATAL: {p.name} missing — the pre-fix measurement cannot be re-read.")
            return 2
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    original = (C.V10_TOKEN_COLUMN, C.V10_SLASHPAIR_RANGE_GUARD)

    all_docs = resolvable_docs()
    base_docs = set(base["verdicts"])
    docs = [(d, rc) for d, rc in all_docs if d.relative_to(ROOT).as_posix() in base_docs]
    added = sorted(d.relative_to(ROOT).as_posix() for d, _ in all_docs
                   if d.relative_to(ROOT).as_posix() not in base_docs)
    print(f"documents with resolvable receipts: {len(all_docs)}   baseline frame: {len(docs)}   "
          f"added since baseline: {len(added)}")

    # ---- G0 sweep fidelity -----------------------------------------------------------------
    print("\nG0 sweep fidelity (real verifier must reproduce the frozen shadow sweep):")
    sweep_obs, sweep_ok, on_diffs = {}, True, []
    for col, guard in ((False, False), (False, True), (True, False), (True, True)):
        p = corpus_pass(docs, col, guard)
        n_d, n_f, diffs = diff_vs_baseline(p, base)
        key = f"col={'on' if col else 'off'},guard={'on' if guard else 'off'}"
        sweep_obs[key] = {"diffs": n_d, "flips": n_f}
        ok = SWEEP[key]["diffs"] == n_d and SWEEP[key]["flips"] == n_f
        sweep_ok &= ok
        if col and guard:
            on_pass, on_diffs = p, diffs
        print(f"  {key:<22s} diffs {n_d:3d} (frozen {SWEEP[key]['diffs']:3d})   "
              f"flips {n_f} (frozen {SWEEP[key]['flips']})   {'OK' if ok else 'MISMATCH'}")

    split = collections.Counter(f"{base['ledger'].get(k)}->{on_pass['ledger'].get(k)}"
                                for k in on_diffs)
    split_ok = dict(split) == SWEEP_SPLIT
    print(f"  transition split {dict(split)} (frozen {SWEEP_SPLIT})   "
          f"{'OK' if split_ok else 'MISMATCH'}")
    g0_pass = sweep_ok and split_ok

    # ---- G1 anchoring, two-armed -----------------------------------------------------------
    anch = anchoring(docs)
    a_on, a_off = anch["on"], anch["off"]
    g1_improves = a_on["misplaced"] < a_off["misplaced"]
    print(f"\nG1 anchoring ON  : {a_on['misplaced']}/{a_on['tokens']} misplaced "
          f"(bar == {G1_BAR_ON})")
    print(f"G1 anchoring OFF : {a_off['misplaced']}/{a_off['tokens']} misplaced "
          f"(frozen {G1_BAR_OFF}; ON must improve)")

    # ---- G2 extraction invariance ----------------------------------------------------------
    md = [p for p in sorted(ROOT.glob("papers/**/*.md")) if "anc" not in p.parts]
    sig_on = extraction_signature(md, True, True)
    sig_off = extraction_signature(md, False, False)
    moved = sorted(k for k in set(sig_on) | set(sig_off) if sig_on.get(k) != sig_off.get(k))
    print(f"\nG2 extraction invariance: {len(moved)} of {len(md)} documents extract differently "
          f"(bar {G2_BAR})")

    # ---- G3 no new accusation --------------------------------------------------------------
    new_ung = [k for k, s in on_pass["ledger"].items()
               if s == "UNGROUNDED" and base["ledger"].get(k) != "UNGROUNDED"]
    failed_before = {d for d, v in base["verdicts"].items() if v != "OATH-HELD"}
    now_failed = {d for d, v in on_pass["verdicts"].items() if v != "OATH-HELD"}
    new_failed = sorted(now_failed - failed_before)
    print(f"G3 new UNGROUNDED : {len(new_ung)} (bar {G3_BAR}) | "
          f"new OATH-FAILED documents: {len(new_failed)} (bar {G3_BAR})")

    # ---- G4 adjudicated coverage ------------------------------------------------------------
    set_flags(True, True)
    rows, unadjudicated = [], []
    for k in on_diffs:
        rel, ln, tok, ordinal = k.rsplit("|", 3)
        line_no, idx = int(ln[1:]), int(ordinal[1:])
        e = on_pass["entries"][(rel, idx)]
        raw = (ROOT / rel).read_text(encoding="utf-8").splitlines()[line_no - 1]
        raw = raw.replace("−", "-")
        ctx, lead = raw.strip(), len(raw) - len(raw.lstrip())
        at = e["col"] - lead
        pre = ctx[max(0, at - 18):at]
        key = (Path(rel).name, line_no, tok, idx)
        verdict = ADJUDICATION.get(key)
        if verdict is None:
            unadjudicated.append("|".join(map(str, key)))
            continue
        rows.append({"key": "|".join(map(str, key)),
                     "transition": f"{base['ledger'].get(k)}->{e['status']}",
                     "adjudication": verdict, "pre_ends_equals": pre.rstrip().endswith("="),
                     "receipt_ref": e["receipt_ref"]})
    restorations = [r for r in rows if r["transition"] == "ABSTAIN->VERIFIED"]
    abstentions = [r for r in rows if r["transition"] == "VERIFIED->ABSTAIN"]
    n_correct = sum(1 for r in restorations if r["adjudication"] == "CORRECT")
    n_wrong = sum(1 for r in restorations if r["adjudication"] == "WRONG")
    destructive = [r for r in abstentions if r["adjudication"] == "DESTRUCTIVE"]
    not_equals = [r["key"] for r in destructive if not r["pre_ends_equals"]]
    print(f"\nG4a restorations  : {n_correct}/{len(restorations)} CORRECT (bar >= "
          f"{G4A_CORRECT_BAR}), WRONG {n_wrong} (bar {G4A_WRONG_BAR})")
    print(f"G4b abstentions   : {len(destructive)}/{len(abstentions)} DESTRUCTIVE "
          f"(bar <= {G4B_BAR})")
    print(f"G4c residual      : {len(destructive) - len(not_equals)}/{len(destructive)} "
          f"destructive abstentions have `pre` ending in a bare '=' (bar: all of them)")

    # ---- G5 severability --------------------------------------------------------------------
    off_off = sweep_obs["col=off,guard=off"]["diffs"]
    off_on = sweep_obs["col=off,guard=on"]["diffs"]
    print(f"\nG5 severability   : both OFF {off_off} diffs (bar {G5_BAR}); "
          f"guard-only {off_on} diffs (bar {G5_BAR})")

    # ---- G6 tamper, NO-CREDIT ---------------------------------------------------------------
    print(f"\nG6a collision channel (mutation CREATES the collision), seeds {G6A_SEEDS}:",
          flush=True)
    g6a = g6_collision(docs, base)
    pa = g6a["pooled"]
    print(f"  pooled: {pa['n']} mutants  caught {pa['off_caught']} -> {pa['on_caught']}  "
          f"false-attested {pa['off_false']} -> {pa['on_false']}  "
          f"abstained {pa['off_abstain']} -> {pa['on_abstain']}")
    print(f"\nG6b clean misplaced roster, seeds {G6B_SEEDS}:", flush=True)
    g6b = g6_clean(docs, census)
    pb = g6b["pooled"]
    print(f"  pooled: caught {pb['off_caught']} -> {pb['on_caught']} "
          f"(|delta| {g6b['pooled_abs_delta']} <= {G6B_TOLERANCE})  "
          f"false-attested {pb['off_false']} -> {pb['on_false']}")

    C.V10_TOKEN_COLUMN, C.V10_SLASHPAIR_RANGE_GUARD = original

    gates = {
        "G0_sweep_fidelity": {
            "role": "VOID CHECK — the shipped edit must be the swept edit",
            "frozen": SWEEP, "observed": sweep_obs,
            "frozen_split": SWEEP_SPLIT, "observed_split": dict(split),
            "pass": g0_pass},
        "G1_anchoring": {
            "role": "ANCHORING (gated, two-armed)", "on": a_on, "off": a_off,
            "bar_on": G1_BAR_ON, "frozen_off": G1_BAR_OFF, "on_improves_off": g1_improves,
            "pass": a_on["misplaced"] == G1_BAR_ON and g1_improves},
        "G2_extraction_invariance": {
            "role": "INVARIANCE (gated, two-armed) — the repair may move no token",
            "documents": len(md), "documents_extracting_differently": len(moved),
            "examples": moved[:20], "bar": G2_BAR, "pass": len(moved) == G2_BAR},
        "G3_no_new_accusation": {
            "role": "SAFETY (gated)", "new_ungrounded": new_ung,
            "new_oath_failed_documents": new_failed, "bar": G3_BAR,
            "pass": not new_ung and not new_failed},
        "G4_adjudicated_coverage": {
            "role": "COVERAGE (gated, hand-adjudicated, ties against the change)",
            "restorations": len(restorations), "restorations_correct": n_correct,
            "restorations_wrong": n_wrong, "bar_correct": G4A_CORRECT_BAR,
            "bar_wrong": G4A_WRONG_BAR,
            "abstentions": len(abstentions), "abstentions_destructive": len(destructive),
            "bar_destructive": G4B_BAR,
            "destructive_not_explained_by_equals_clause": not_equals,
            "unadjudicated": unadjudicated, "rows": rows,
            "pass": (n_correct >= G4A_CORRECT_BAR and n_wrong == G4A_WRONG_BAR
                     and len(destructive) <= G4B_BAR and not not_equals
                     and not unadjudicated)},
        "G5_severability": {
            "role": "SEVERABILITY (gated, two bars)",
            "both_off_ledger_differences": off_off,
            "guard_only_ledger_differences": off_on, "bar": G5_BAR,
            "pass": off_off == G5_BAR and off_on == G5_BAR},
        "G6a_tamper_collision_channel": {
            "role": "TAMPER, NO-CREDIT (gated against regression only)", **g6a},
        "G6b_tamper_clean_roster": {
            "role": "TAMPER, NO-CREDIT (gated against regression only)", **g6b},
    }
    report = {
        "prereg": "PREREG_oath_v10_token_column_2026_08_23.md",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "baseline_verifier_sha256": base["verifier_sha256"],
        "census_verifier_sha256": census["generated_at_verifier_sha256"],
        "shipped_flags": {"V10_TOKEN_COLUMN": original[0],
                          "V10_SLASHPAIR_RANGE_GUARD": original[1]},
        "documents_resolvable_now": len(all_docs),
        "documents_in_baseline_frame": len(docs),
        "documents_added_since_baseline": added,
        "gates": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                  for k, v in gates.items()},
        "tamper_per_seed": {"G6a": g6a["per_seed"], "G6b": g6b["per_seed"]},
        "asserted_invariants_not_gated": {
            "I1": "`col` is a new key on each ledger entry; no existing key changes value or "
                  "type. seal.py, corpus_audit.py and the suite read status/line/token only.",
            "I2": "anchoring is exact by construction — `col` is the offset the match was found "
                  "at, in a string the same length as the source line. G1 is an identity check "
                  "and is reported as one, not credited as a measurement.",
        },
        "declared_no_credit": "G6a/G6b. This cycle claims NO tamper improvement. The "
                              "false-attestation column rises (see pooled figures) and is "
                              "published as the cost of pointing the windows at the right text; "
                              "the abstentions it replaces were produced by windows aimed at "
                              "another token and were never a safety property.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\ngates: " + "  ".join(f"{k.split('_')[0]}={'PASS' if v['pass'] else 'FAIL'}"
                                  for k, v in gates.items()))
    allp = all(v["pass"] for v in gates.values())
    print(f"ALL GATES: {'PASS' if allp else 'MISSED A BAR'}")
    print(f"elapsed {report['elapsed_s']}s -> {OUT.name}")
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())
