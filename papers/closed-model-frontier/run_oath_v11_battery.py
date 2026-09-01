"""OATH v0.11 battery — G1..G8 per PREREG_oath_v11_row_ordinal_retraction_2026_08_25.

Under test: ONE flag-gated clause in ``styxx.certify``.

  V11_ORDINAL_LABEL   a status-level demotion to ABSTAIN with the machine-readable reason
                      `row_ordinal_label`, at the `is_spec` tier — before any obligation or
                      match is consulted. Fires iff the token's recorded column lies in the
                      FIRST cell of a markdown table DATA row, the header row's first cell is
                      an exact member of the 9-entry vocabulary frozen in the prereg, and the
                      cell is entirely a bare non-negative integer <= 100.

This cycle performs, once, by design, the event v0.9's G4 existed to forbid: four
UNGROUNDED->ABSTAIN conversions and one FAILED->HELD flip. The license is structural — the
retraction targets the accusation's PRESUPPOSITION (claimhood), never its verdict — and the
whole point of this battery is that the license is CHECKED rather than argued.

Every gate runs two-armed (flag OFF / flag ON) at the ship-candidate verifier, on the frame as
frozen in the prereg. Mutation operators are IMPORTED from the v0.9 battery, never copied.
Non-destructive: mutants live in temp files, corpus passes are in-memory, and the only file
written is this battery's own result JSON.

What is NOT here, and why: G7 (suite closure) and G9 (boundary disclosure) are properties of
the ship commit and the RESULT document, not of a harness run. G9's re-derived surfaces ARE
computed here, under `boundary_surfaces`, so the RESULT can cite a receipt rather than a
memory.

  python papers/closed-model-frontier/run_oath_v11_battery.py
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
# every flag write in this harness would then land on the wrong object -- both arms would
# silently run the shipped verifier and the battery would report a real-looking null.
C = importlib.import_module("styxx.certify")
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

# Mutation operators IMPORTED from the v0.9 battery (prereg: "imported ... never copied").
# `mutate_sig` perturbs one significant digit keeping format; `substitute` is the sign-aware
# replacement that stops a U+2212-signed claim from silently no-oping into a fake verifier miss.
sys.path.insert(0, str(HERE))
_v09 = importlib.import_module("run_oath_v09_battery")
mutate_sig = _v09.mutate_sig
substitute = _v09.substitute

BASELINE = HERE / "oath_v11_baseline_ledger.json"
CENSUS = HERE / "oath_v10_ordinal_census.json"
PANEL = HERE / "oath_v11_panel_recheck.json"
OUT = HERE / "oath_v11_battery_result.json"

REASON = "row_ordinal_label"
PROSPECTUS = "papers/agent-conscience/PROSPECTUS_knowsay_2026_07_27.md"

# ---------------------------------------------------------------- frozen expectations
# Everything below is quoted from the prereg, which was frozen on commit cbd2864 BEFORE
# styxx/certify.py was touched. No bar here moves.

# The 11-coordinate roster: (line, token, OFF-arm status). V 5 / U 4 / A 2.
ROSTER = (
    (25, "1", "VERIFIED"), (26, "2", "VERIFIED"), (27, "3", "UNGROUNDED"),
    (28, "4", "UNGROUNDED"), (29, "5", "UNGROUNDED"), (30, "6", "ABSTAIN"),
    (31, "7", "ABSTAIN"), (32, "8", "UNGROUNDED"), (33, "9", "VERIFIED"),
    (34, "10", "VERIFIED"), (35, "11", "VERIFIED"),
)
# The enumerated retraction whitelist. Permitted UNGROUNDED->ABSTAIN: exactly these four.
TARGETS = frozenset({(27, "3"), (28, "4"), (29, "5"), (32, "8")})
# Expected collateral, enumerated before data. Permitted VERIFIED->ABSTAIN: exactly these five.
COLLATERAL = frozenset({(25, "1"), (26, "2"), (33, "9"), (34, "10"), (35, "11")})

G1_FRAME = {"documents": 140, "tokens": 5681,
            "VERIFIED": 4196, "ABSTAIN": 1481, "UNGROUNDED": 4}
G3_POST = {"VERIFIED": 4191, "ABSTAIN": 1490, "UNGROUNDED": 0}
G4A_ROSTER = {"VERIFIED": 5, "UNGROUNDED": 4, "ABSTAIN": 2}
G4B_FRESH_N = 10                       # "a fresh draw of 10 non-PROSPECTUS tokens"
G5_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
G5_MIN_VALID_SEEDS = 8
G5_COMPARISON_BAND = (22, 36)          # severable: the broad-class contrast, mean ~28.1
G6_MUTANTS = 117
G6_SWEEP = {"UNGROUNDED": 46, "VERIFIED": 50, "ABSTAIN": 21}
G8_TOKENS = 5681                       # non-extraction would read 5,670

# The census's wider 16-entry vocabulary, quoted from oath_v10_ordinal_census.json, used ONLY to
# re-derive the REGENERATION surface for G9. It is not the clause's vocabulary and never gates.
CENSUS_VOCAB = frozenset({"", "#", "-", "claim", "id", "idx", "index", "item", "line",
                          "n", "no", "no.", "nr", "num", "rank", "row"})


def set_v11(on: bool) -> None:
    C.V11_ORDINAL_LABEL = on


# ---------------------------------------------------------------- frame

def resolvable_docs() -> list[tuple[Path, list[Path]]]:
    """The certified frame — identical definition to the census's `frame.certified` and to
    `_resolvable()` in tests/test_certificate_reproduces.py."""
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


def frame_pass(docs, on: bool) -> dict:
    """Re-certify the whole frame in one arm. Returns per-document certificates."""
    set_v11(on)
    certs = {}
    for doc, receipts in docs:
        certs[doc.relative_to(ROOT).as_posix()] = C.certify_doc(doc, receipts)
    return certs


def firings(cert) -> set:
    return {(e["line"], e.get("col"), e["token"])
            for e in cert["ledger"] if e["receipt_ref"] == REASON}


# ---------------------------------------------------------------- G1: instrument validity

def gate_g1(off_certs) -> dict:
    """VOID-producing. Frame drift is a property of the tree, not of the clause.

    Two legs. (a) Extractor replication: the pre-change extraction snapshot must reproduce
    token-for-token at this verifier, repo-wide — this cycle refactored the table-header walk
    that `extract_numbers` uses, so a moved token would invalidate every number downstream.
    (b) Frame at run == frame at freeze.
    """
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    mism = []
    for rel, rec in base["extraction"].items():
        toks = C.extract_numbers((ROOT / rel).read_text(encoding="utf-8"))
        blob = json.dumps(toks, sort_keys=True, ensure_ascii=False)
        if hashlib.sha256(blob.encode("utf-8")).hexdigest() != rec["sha256"]:
            mism.append(rel)

    counts = collections.Counter()
    for c in off_certs.values():
        counts.update(c["counts"])
    frame = {"documents": len(off_certs), "tokens": sum(len(c["ledger"]) for c in off_certs.values()),
             "VERIFIED": counts["VERIFIED"], "ABSTAIN": counts["ABSTAIN"],
             "UNGROUNDED": counts["UNGROUNDED"]}

    # AUDIT, not a bar: per-token OFF-arm identity against the pre-change baseline, statuses AND
    # reasons. Strictly stronger than the aggregate leg the prereg gates on, and reported rather
    # than gated because no bar moves after the freeze — including in the strict direction.
    off_ledger = {}
    for rel, c in off_certs.items():
        for i, e in enumerate(c["ledger"]):
            off_ledger[f"{rel}|L{e['line']}|{e['token']}|#{i}"] = [e["status"], e["receipt_ref"]]
    sev = [k for k, v in base["ledger"].items() if off_ledger.get(k) != v]

    # The same omission guard bars (ii)/(iii) of G4'b were missing: "mismatches = 0 over all
    # documents" is a negative existence test, and an emptied or truncated baseline map would
    # satisfy it over ZERO documents. The population must be non-empty AND must be the one the
    # baseline recorded for itself. Implements the frozen bar; does not move it.
    replicated = len(base["extraction"])
    population_ok = (replicated > 0
                     and replicated == base["extraction_repo_wide"]["documents"])
    ok = population_ok and not mism and frame == G1_FRAME
    return {"gate": "G1", "name": "INSTRUMENT VALIDITY (VOID-producing)",
            # The outcome token is carried in the verdict, as every other gate does. A bare
            # "VOID" here crashed `main` on the ONE path a VOID-producing gate exists to take —
            # so the pre-committed V11_BATTERY_VOID outcome could never have been written.
            # Found by the adversarial audit of this battery, then reproduced by a real VOID.
            "verdict": "PASS" if ok else "VOID:V11_BATTERY_VOID",
            "extractor_replication_documents": replicated,
            "extractor_replication_population_ok": population_ok,
            "extractor_mismatches": len(mism), "extractor_mismatch_roster": mism[:20],
            "frame_at_freeze": G1_FRAME, "frame_at_run": frame,
            "frame_matches": frame == G1_FRAME,
            "audit_off_arm_severability_mismatches": len(sev),
            "audit_off_arm_severability_roster": sev[:20],
            "audit_note": "Per-token OFF-arm identity (status AND reason) against the "
                          "pre-change baseline. An AUDIT, not a bar: the prereg gates G1 on the "
                          "aggregate frame, and adding a bar after the freeze is bar-moving even "
                          "when it makes the run harder to pass."}


# ---------------------------------------------------------------- G2: firing surface

def gate_g2(on_certs) -> dict:
    """Exact, both directions. A 12th firing is over-reach; a missed roster token is under-reach."""
    fired = {rel: firings(c) for rel, c in on_certs.items()}
    in_prospectus = fired.get(PROSPECTUS, set())
    elsewhere = {rel: sorted(f) for rel, f in fired.items() if f and rel != PROSPECTUS}
    got = {(ln, tok) for ln, _col, tok in in_prospectus}
    want = {(ln, tok) for ln, tok, _st in ROSTER}
    over = sorted(got - want)
    under = sorted(want - got)
    ok = not over and not under and not elsewhere
    return {"gate": "G2", "name": "FIRING-SURFACE EXACTNESS",
            "verdict": "PASS" if ok else ("FAIL:V11_OVERREACH" if over or elsewhere
                                          else "FAIL:V11_UNDERREACH"),
            "firings_in_frame": sum(len(f) for f in fired.values()),
            "firing_documents": sorted(rel for rel, f in fired.items() if f),
            "prospectus_firings": len(in_prospectus),
            "over_reach": over, "under_reach": under,
            "firings_outside_prospectus": elsewhere,
            "coordinates": sorted(in_prospectus)}


# ---------------------------------------------------------------- G3: retraction ledger audit

def gate_g3(off_certs, on_certs) -> dict:
    """Whole-frame re-certification, ON vs OFF.

    Stated honestly, as the prereg does: given G1's frozen statuses and G2's exact roster, every
    equality here follows ARITHMETICALLY. G3 is an end-to-end implementation audit of the certify
    pipeline — gated because an implementation can fail where arithmetic cannot — and it is NOT
    independent evidence for the retraction. That evidence lives in G4' and G5.

    Two legs are demoted outright and do not appear in the bar list: A1 (0 conversions elsewhere
    in frame) is true by construction, because the frame's four UNGROUNDED ARE the target; and
    I2 (HELD->FAILED = 0) is an identity of an abstain-only clause. Both are audited below.
    """
    conversions = collections.Counter()
    ung_to_abs, ver_to_abs, held_to_failed, failed_to_held, misaligned = [], [], [], [], []
    for rel, off in off_certs.items():
        on = on_certs[rel]
        if len(off["ledger"]) != len(on["ledger"]):
            misaligned.append(rel)
            continue
        for a, b in zip(off["ledger"], on["ledger"]):
            # Index alignment IS the mechanism proof: the clause must not move extraction.
            if (a["line"], a["token"], a.get("col")) != (b["line"], b["token"], b.get("col")):
                misaligned.append(f"{rel}|L{a['line']}|{a['token']}")
                continue
            if a["status"] == b["status"]:
                continue
            conversions[f"{a['status']}->{b['status']}"] += 1
            coord = (rel, a["line"], a["token"])
            if a["status"] == "UNGROUNDED" and b["status"] == "ABSTAIN":
                ung_to_abs.append(coord)
            elif a["status"] == "VERIFIED" and b["status"] == "ABSTAIN":
                ver_to_abs.append(coord)
        if off["verdict"] == "OATH-HELD" and on["verdict"] != "OATH-HELD":
            held_to_failed.append(rel)
        if off["verdict"] != "OATH-HELD" and on["verdict"] == "OATH-HELD":
            failed_to_held.append(rel)

    post = collections.Counter()
    for c in on_certs.values():
        post.update(c["counts"])
    post_frame = {k: post[k] for k in ("VERIFIED", "ABSTAIN", "UNGROUNDED")}

    got_u = {(ln, tok) for rel, ln, tok in ung_to_abs if rel == PROSPECTUS}
    got_v = {(ln, tok) for rel, ln, tok in ver_to_abs if rel == PROSPECTUS}
    off_target = [c for c in ung_to_abs + ver_to_abs if c[0] != PROSPECTUS]

    ok = (got_u == set(TARGETS) and got_v == set(COLLATERAL) and not off_target
          and failed_to_held == [PROSPECTUS] and post_frame == G3_POST and not misaligned
          and not held_to_failed)
    return {"gate": "G3", "name": "THE RETRACTION LEDGER AUDIT",
            "verdict": "PASS" if ok else "FAIL:V11_RETRACTION_MISCOUNT",
            "transitions": dict(conversions),
            "ungrounded_to_abstain": sorted(ung_to_abs),
            "ungrounded_to_abstain_matches_whitelist": got_u == set(TARGETS),
            "verified_to_abstain": sorted(ver_to_abs),
            "verified_to_abstain_matches_enumeration": got_v == set(COLLATERAL),
            "failed_to_held": failed_to_held,
            "off_target_conversions": off_target,
            "post_clause_frame_expected": G3_POST, "post_clause_frame": post_frame,
            "ledger_misalignments": misaligned,
            "audit_A1_conversions_elsewhere_in_frame": len(off_target),
            "audit_A1_note": "Vacuous in-frame and never sold as a passed bar: the frame's four "
                             "UNGROUNDED ARE the target.",
            "audit_I2_held_to_failed": len(held_to_failed),
            "audit_I2_note": "Zero by construction of an abstain-only clause. Audited and "
                             "reported; deliberately absent from this gate's bar list.",
            "genuine_verifications_destroyed": 0 if got_v == set(COLLATERAL) else None,
            "genuine_verifications_note": "All five VERIFIED->ABSTAIN were hand-adjudicated as "
                                          "false attestations (a rate coincidence, three index "
                                          "leaves, one unrelated count) and re-adjudicated blind "
                                          "at G4'b."}


# ---------------------------------------------------------------- G4'a: mechanical warrant

def gate_g4a(off_certs) -> dict:
    """The 11-token roster's OFF-arm statuses reproduce exactly at the ship-candidate verifier."""
    ledger = {(e["line"], e["token"]): e["status"]
              for e in off_certs[PROSPECTUS]["ledger"]}
    got, drift = collections.Counter(), []
    for ln, tok, want in ROSTER:
        have = ledger.get((ln, tok))
        got[have] += 1
        if have != want:
            drift.append({"line": ln, "token": tok, "expected": want, "observed": have})
    ok = not drift and {k: v for k, v in got.items()} == G4A_ROSTER
    return {"gate": "G4'a", "name": "WARRANT — mechanical leg",
            "verdict": "PASS" if ok else "FAIL:V11_WARRANT_FAILED",
            "expected": G4A_ROSTER, "observed": dict(got), "drift": drift}


def gate_g4b() -> dict:
    """The human leg, adjudicated out of battery. Read here, scored here, never scored by here.

    The battery does not adjudicate; it reads the artifact and applies the prereg's three bars:
      (i)   all four targets adjudicate LABEL
      (ii)  zero LABELs among the fresh non-PROSPECTUS draw
      (iii) zero CLAIM calls on the seven non-target roster tokens
    """
    if not PANEL.exists():
        return {"gate": "G4'b", "name": "WARRANT — second blind adjudicator",
                "verdict": "ABSENT", "note": f"{PANEL.name} not present"}
    p = json.loads(PANEL.read_text(encoding="utf-8"))
    calls = {(c["rel"], c["line"], str(c["token"])): c["call"] for c in p["cases"]}
    tgt = [{"line": ln, "token": tok, "call": calls.get((PROSPECTUS, ln, tok))}
           for ln, tok in sorted(TARGETS)]
    non_target = [{"line": ln, "token": tok, "call": calls.get((PROSPECTUS, ln, tok))}
                  for ln, tok, _s in ROSTER if (ln, tok) not in TARGETS]
    fresh = [{"rel": rel, "line": ln, "token": tok, "call": call}
             for (rel, ln, tok), call in calls.items() if rel != PROSPECTUS]

    # COVERAGE, checked before any bar is read.
    #
    # Bars (ii) and (iii) are NEGATIVE existence tests — "zero LABELs in the fresh draw", "zero
    # CLAIM calls on the seven non-targets". `all()` over an empty sequence is True, and an
    # absent case yields `call = None`, for which `None != "LABEL"` is also True. Without this
    # block a TRUNCATED panel — no fresh draw at all, or no non-target roster cases — clears both
    # bars by OMISSION, and an adjudicator who examined nothing would license the retraction.
    #
    # A gate satisfied by omission is precisely the defect this cycle exists to retract, one
    # level up: G8 names it certified-by-omission, "the inverse of the oath", and the Retraction
    # Protocol's fifth clause is "silence loud, never omission". Found by the adversarial audit
    # of this battery, not by its author.
    #
    # This IMPLEMENTS the frozen bar rather than moving it. The prereg sizes both populations
    # exactly — "a fresh draw of 10 non-PROSPECTUS tokens", "the seven non-target roster tokens"
    # — so a gate that passes on zero of them was never enforcing what was frozen. An incomplete
    # artifact yields INCOMPLETE, never PASS and never FAIL: the battery refuses to score rather
    # than inventing a verdict the outcome table does not name.
    scored = ("CLAIM", "LABEL")
    coverage = {
        "targets_adjudicated": sum(1 for c in tgt if c["call"] in scored),
        "targets_expected": len(TARGETS),
        "non_target_roster_adjudicated": sum(1 for c in non_target if c["call"] in scored),
        "non_target_roster_expected": len(ROSTER) - len(TARGETS),
        "fresh_draw_adjudicated": sum(1 for c in fresh if c["call"] in scored),
        "fresh_draw_expected": G4B_FRESH_N,
        "reported_totals_reconcile_with_cases":
            p.get("totals", {}).get("examined") == len(p["cases"]),
    }
    complete = (coverage["targets_adjudicated"] == coverage["targets_expected"]
                and coverage["non_target_roster_adjudicated"]
                == coverage["non_target_roster_expected"]
                and coverage["fresh_draw_adjudicated"] == coverage["fresh_draw_expected"]
                and coverage["reported_totals_reconcile_with_cases"])
    if not complete:
        return {"gate": "G4'b", "name": "WARRANT — second blind adjudicator",
                "verdict": "INCOMPLETE", "artifact": PANEL.name, "coverage": coverage,
                "note": "The panel artifact does not cover the populations the prereg sizes. "
                        "Bars (ii) and (iii) are negative existence tests and would pass "
                        "vacuously; the battery refuses to score them."}

    bar_i = all(c["call"] == "LABEL" for c in tgt) and len(tgt) == 4
    bar_ii = all(c["call"] != "LABEL" for c in fresh)
    bar_iii = all(c["call"] != "CLAIM" for c in non_target)
    if bar_i and bar_ii and bar_iii:
        verdict = "PASS"
    elif not bar_i:
        verdict = "FAIL:V11_WARRANT_FAILED"
    elif not bar_ii:
        verdict = "FAIL:V11_CLASS_UNDERENUMERATED"
    else:
        verdict = "FAIL:V11_COLLATERAL_CONTESTED"
    return {"gate": "G4'b", "name": "WARRANT — second blind adjudicator",
            "verdict": verdict, "artifact": PANEL.name, "coverage": coverage,
            "verifier_pin": p.get("verifier_pin"), "totals": p.get("totals"),
            "bar_i_all_four_targets_LABEL": bar_i, "targets": tgt,
            "bar_ii_zero_LABEL_in_fresh_draw": bar_ii,
            "fresh_draw_examined": len(fresh),
            "fresh_draw_LABELs": [c for c in fresh if c["call"] == "LABEL"],
            "bar_iii_zero_CLAIM_on_non_targets": bar_iii, "non_targets": non_target}


# ---------------------------------------------------------------- mutation machinery

def _mutant_cert(doc: Path, receipts, line_no: int, tok: str, mut: str):
    """Certify a one-token mutant of *doc* in a temp file. Returns (cert, landed)."""
    lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
    ml = list(lines)
    ml[line_no - 1], landed = substitute(ml[line_no - 1], tok, mut)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(ml))
        tmp = Path(tf.name)
    try:
        return C.certify_doc(tmp, receipts), landed
    finally:
        tmp.unlink(missing_ok=True)


def _row_at(cert, line_no: int, col: int):
    """The ledger row at a COORDINATE, not at a token string.

    Keying by (line, token) — the v0.9 battery's key — collides whenever a mutation happens to
    produce a digit already present on the line. `col` is length-preserved by `mutate_sig`, so
    the coordinate survives the mutation and the lookup is exact.
    """
    return next((e for e in cert["ledger"]
                 if e["line"] == line_no and e.get("col") == col), None)


# ---------------------------------------------------------------- G5: catch decomposition

def gate_g5(docs, off_certs, on_certs) -> dict:
    """Ten seeds. Raw ON-caught is NOT a gate (I1: for a structure-reading clause ON = 0 is an
    identity, and reporting it as a tamper result would launder an identity as a finding).
    `catch_surfacing_in_verdict` = 0 is NOT a gate either (I3: it cannot fail while G2 holds).
    Both are audited. The gates are the positive control, the identity audit, and collateral.
    """
    by_rel = {d.relative_to(ROOT).as_posix(): (d, r) for d, r in docs}
    doc, receipts = by_rel[PROSPECTUS]
    roster_cols = {(e["line"], e["token"]): e.get("col")
                   for e in off_certs[PROSPECTUS]["ledger"]}
    on_base_fire = firings(on_certs[PROSPECTUS])
    off_base_held = off_certs[PROSPECTUS]["verdict"] == "OATH-HELD"
    on_base_held = on_certs[PROSPECTUS]["verdict"] == "OATH-HELD"

    # The broad 123-token comparison class: every sole-token first cell in frame, which is what
    # the REJECTED positional rule would have silenced.
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    broad = [r for r in census["roster"] if r.get("sole_token_in_cell")]

    seeds = []
    for seed in G5_SEEDS:
        # ONE rng stream per seed, consumed before either arm runs, so both arms see the SAME
        # mutants. Generating them inside an arm would make the arms incomparable.
        rng = random.Random(seed)
        muts = [(ln, tok, mutate_sig(tok, rng)) for ln, tok, _s in ROSTER]

        set_v11(False)
        off_caught = off_surface = off_unlanded = 0
        for ln, tok, mut in muts:
            cert, landed = _mutant_cert(doc, receipts, ln, tok, mut)
            off_unlanded += not landed
            row = _row_at(cert, ln, roster_cols[(ln, tok)])
            if row and row["status"] == "UNGROUNDED":
                off_caught += 1
            if off_base_held and cert["verdict"] != "OATH-HELD":
                off_surface += 1

        set_v11(True)
        override_missed = collateral = on_surface = 0
        for ln, tok, mut in muts:
            cert, _ = _mutant_cert(doc, receipts, ln, tok, mut)
            row = _row_at(cert, ln, roster_cols[(ln, tok)])
            # The clause must still fire ON THE MUTATED TOKEN. If it does not, it has become
            # value-reading -- a fuse that is absent on exactly the input it exists to handle.
            if not (row and row["receipt_ref"] == REASON):
                override_missed += 1
            # Sibling abstentions must survive. The rejected 1..N run detectors lost 110 (and 90)
            # per doctored digit, because breaking the run silenced the whole column.
            expect = {f for f in on_base_fire if f[0] != ln}
            if len(expect - {f for f in firings(cert) if f[0] != ln}) > 0:
                collateral += len(expect - {f for f in firings(cert) if f[0] != ln})
            if on_base_held and cert["verdict"] != "OATH-HELD":
                on_surface += 1

        # Severable comparison arm: the broad class's reader-visible catch cost, measured at the
        # OFF arm. A mutation that flips its document OATH-HELD -> OATH-FAILED is a catch a
        # reader can see; the rejected positional rule silences all 123, so every one of these
        # is a catch it destroys.
        set_v11(False)
        broad_surface = broad_unlanded = 0
        for r in broad:
            rel = r["rel"]
            if rel not in by_rel or off_certs[rel]["verdict"] != "OATH-HELD":
                continue
            bdoc, breceipts = by_rel[rel]
            mut = mutate_sig(r["token"], rng)
            cert, landed = _mutant_cert(bdoc, breceipts, r["line"], r["token"], mut)
            broad_unlanded += not landed
            if cert["verdict"] != "OATH-HELD":
                broad_surface += 1

        seeds.append({"seed": seed, "off_arm_caught": off_caught,
                      "off_arm_did_not_land": off_unlanded,
                      "override_missed_mutant": override_missed,
                      "collateral_abstentions_lost": collateral,
                      "off_arm_catch_surfacing": off_surface,
                      "on_arm_catch_surfacing": on_surface,
                      "broad_class_surfacing_destroyed": broad_surface,
                      "broad_class_did_not_land": broad_unlanded})

    valid = [s for s in seeds if s["off_arm_caught"] >= 1]
    caught = [s["off_arm_caught"] for s in seeds]
    broad_vals = [s["broad_class_surfacing_destroyed"] for s in seeds]
    broad_mean = sum(broad_vals) / len(broad_vals)
    override_bad = [s["seed"] for s in seeds if s["override_missed_mutant"] > 0]
    collat_bad = [s["seed"] for s in seeds if s["collateral_abstentions_lost"] > 0]

    if len(valid) < G5_MIN_VALID_SEEDS:
        verdict = "VOID:V11_BATTERY_VOID"
    elif override_bad:
        verdict = "FAIL:V11_FUSE"
    elif collat_bad:
        verdict = "FAIL:V11_COLLATERAL"
    else:
        verdict = "PASS"
    lo, hi = G5_COMPARISON_BAND
    return {"gate": "G5", "name": "CATCH DECOMPOSITION",
            "verdict": verdict, "seeds": seeds,
            "positive_control_valid_seeds": len(valid), "positive_control_bar": G5_MIN_VALID_SEEDS,
            "off_arm_caught_mean": sum(caught) / len(caught),
            "off_arm_caught_range": [min(caught), max(caught)],
            "identity_audit_override_missed_seeds": override_bad,
            "collateral_nonzero_seeds": collat_bad,
            "comparison_arm": {
                "severable": True,
                "class": "the broad 123-token sole-token first-cell class — what the REJECTED "
                         "positional rule would silence",
                "n_tokens": len(broad),
                "surfacing_destroyed_mean": broad_mean,
                "surfacing_destroyed_range": [min(broad_vals), max(broad_vals)],
                "band": list(G5_COMPARISON_BAND),
                "reproduces": lo <= broad_mean <= hi,
                "note": "Irreproducibility VOIDs this comparison claim only, never the gated "
                        "legs above. This contrast — not the entailed zero — is the measured "
                        "content of the catch leg."},
            "audit_I1_note": "Raw ON-arm caught is an identity for a structure-reading clause "
                             "and is not reported as a tamper result. Its audit is "
                             "override_missed_mutant = 0, and that audit IS gated above.",
            "audit_I3_on_arm_surfacing": [s["on_arm_catch_surfacing"] for s in seeds],
            "audit_I3_note": "Entailed by G2 plus the frozen frame: every in-frame firing sits "
                             "in PROSPECTUS_knowsay, OATH-FAILED at the OFF baseline, and "
                             "surfacing counts only HELD->FAILED transitions. Cannot fail while "
                             "G2 holds; audited, never counted as a passed bar."}


# ---------------------------------------------------------------- G6: exhaustive sweep

def gate_g6(docs, off_certs) -> dict:
    """All 117 single-significant-digit mutants of the 11 tokens, OFF arm.

    The affirmative case — a 0.427 false-attestation rate under tamper on tokens that assert
    nothing — must reproduce, or the argument for retracting loses its measured content.
    """
    by_rel = {d.relative_to(ROOT).as_posix(): (d, r) for d, r in docs}
    doc, receipts = by_rel[PROSPECTUS]
    cols = {(e["line"], e["token"]): e.get("col") for e in off_certs[PROSPECTUS]["ledger"]}
    set_v11(False)
    counts, unlanded, rows = collections.Counter(), 0, []
    for ln, tok, _s in ROSTER:
        for pos, ch in enumerate(tok):
            if not ch.isdigit():
                continue
            for d in range(10):
                if d == int(ch):
                    continue
                mut = tok[:pos] + str(d) + tok[pos + 1:]
                cert, landed = _mutant_cert(doc, receipts, ln, tok, mut)
                unlanded += not landed
                row = _row_at(cert, ln, cols[(ln, tok)])
                st = row["status"] if row else "NOT_EXTRACTED"
                counts[st] += 1
                rows.append({"line": ln, "token": tok, "mutant": mut, "status": st})
    n = sum(counts.values())
    got = {k: counts[k] for k in ("UNGROUNDED", "VERIFIED", "ABSTAIN")}
    if unlanded:
        verdict = "VOID:V11_BATTERY_VOID"
    elif n == G6_MUTANTS and got == G6_SWEEP:
        verdict = "PASS"
    else:
        verdict = "FAIL:V11_SWEEP_DRIFT"
    return {"gate": "G6", "name": "EXHAUSTIVE SWEEP REPRODUCTION",
            "verdict": verdict, "mutants_expected": G6_MUTANTS, "mutants_run": n,
            "expected": G6_SWEEP, "observed": got, "outcomes_all": dict(counts),
            "did_not_land": unlanded,
            "false_attestation_rate_under_tamper": round(counts["VERIFIED"] / n, 4) if n else None,
            "note": "did_not_land > 0 VOIDs rather than fails: the operator broke, not the clause.",
            "mutant_ledger": rows}


# ---------------------------------------------------------------- G8: mechanism proof

def gate_g8(on_certs) -> dict:
    """Post-clause extracted token count exactly 5,681. Non-extraction would read 5,670.

    Artifact home, stated plainly: the battery is non-destructive and G7 forbids touching
    committed certificates, so the countable trail of record this cycle is THIS result JSON. The
    committed PROSPECTUS certificate stays byte-identical (and stale in its counts) until a
    future re-certification cycle regenerates it.
    """
    tokens = sum(len(c["ledger"]) for c in on_certs.values())
    rows = [{"line": e["line"], "col": e.get("col"), "token": e["token"],
             "status": e["status"], "reason": e["receipt_ref"]}
            for e in on_certs[PROSPECTUS]["ledger"] if e["receipt_ref"] == REASON]
    ok = (tokens == G8_TOKENS and len(rows) == 11
          and all(r["status"] == "ABSTAIN" and r["reason"] == REASON for r in rows))
    return {"gate": "G8", "name": "MECHANISM PROOF",
            "verdict": "PASS" if ok else "FAIL:V11_MECHANISM_DRIFT",
            "post_clause_tokens": tokens, "expected_tokens": G8_TOKENS,
            "non_extraction_would_read": 5670,
            "silenced_rows_by_coordinate": rows,
            "abstained_array_carries_all_11": all(
                any(a["line"] == r["line"] and a["token"] == r["token"]
                    for a in on_certs[PROSPECTUS]["abstained"]) for r in rows)}


# ---------------------------------------------------------------- G9 inputs: boundary surfaces

def boundary_surfaces(off_certs) -> dict:
    """Re-derive, at this verifier, the surfaces G9 requires the RESULT to publish.

    Two surfaces kept deliberately apart: the REGENERATION surface (the census's wider 16-entry
    vocabulary — what the class regrows into if documents are merely repaired) and the frozen
    clause's OWN firing surface. Conflating them is how a cycle talks itself into believing a
    narrow rule covers a wide problem.
    """
    in_frame = set(off_certs)
    census_hits, clause_hits = [], []
    blind = collections.Counter()
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        rel = md.relative_to(ROOT).as_posix()
        trows = C._table_rows(lines)
        for num in C.extract_numbers(text):
            hdr_ln = trows.get(num["line"])
            if hdr_ln is None or "col" not in num:
                continue
            row = lines[num["line"] - 1].replace("−", "-")
            span = C._first_cell(row)
            hspan = C._first_cell(lines[hdr_ln - 1])
            if span is None or hspan is None or not span[0] <= num["col"] < span[1]:
                continue
            hdr = lines[hdr_ln - 1][hspan[0]:hspan[1]].replace("`", "").strip(" \t*_").strip()
            cell = row[span[0]:span[1]]
            if not C._v11_sole_int(cell):
                continue
            rec = {"rel": rel, "line": num["line"], "token": num["token"], "header": hdr,
                   "certified": rel in in_frame}
            if hdr.casefold() in CENSUS_VOCAB:
                census_hits.append(rec)
            if C._v11_header_ok(lines[hdr_ln - 1][hspan[0]:hspan[1]]):
                clause_hits.append(rec)
        # Blind spots: tables the clause structurally cannot reach, counted over MAXIMAL runs of
        # pipe-leading lines. A run the header machinery never binds is a completeness gap, not a
        # silencing gap — its tokens stay OBLIGATED, so PROSPECTUS-shaped false accusations
        # remain possible there. Disclosed, not fixed.
        i = 0
        while i < len(lines):
            if not lines[i].lstrip().startswith("|"):
                i += 1
                continue
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            run = lines[i:j]
            seps = [k for k, ln in enumerate(run) if C._TABLE_SEP.match(ln)]
            blind["tables_total"] += 1
            if not seps:
                blind["separator_less"] += 1
            elif seps[0] == 0:
                # The separator opens the run, so there is no preceding '|' line to be a header:
                # `_table_rows` binds nothing here.
                blind["header_less"] += 1
            if len(seps) > 1:
                # A scope oddity of the SHIPPED header machinery: a second separator inside one
                # run does not start a new table, so later rows keep inheriting the first header.
                blind["multi_separator"] += 1
            i = j

    def tally(hits):
        return {"tokens": len(hits), "documents": len({h["rel"] for h in hits}),
                "in_certified_frame": sum(1 for h in hits if h["certified"]),
                "outside_certified_frame": sum(1 for h in hits if not h["certified"]),
                "by_header": dict(collections.Counter(h["header"] for h in hits).most_common())}

    rank_obligated = [h for h in census_hits if h["header"].casefold() == "rank"]
    return {
        "regeneration_surface_census_vocabulary": tally(census_hits),
        "frozen_clause_firing_surface": tally(clause_hits),
        "excluded_rank_tokens_still_obligated": tally(rank_obligated),
        "blind_spots": dict(blind),
        "tables_the_header_machinery_never_binds":
            blind["separator_less"] + blind["header_less"],
        "blind_spot_reconciliation":
            "The prereg's G9 list names '37 headerless, 10 separator-less, 27 multi-separator' "
            "as if the first two were disjoint classes. They are not: a table with NO separator "
            "also has no header row, so the 10 are a subset of the 37. Re-derived here the 37 "
            "decomposes exactly — 27 runs whose separator opens the run (a header line could "
            "exist and does not) PLUS 10 runs with no separator at all — and the union "
            "reproduces the red-team receipt's `tables_without_a_header_row = 37`. The "
            "multi-separator 27 is a genuinely separate class and reproduces independently. "
            "Published as a decomposition rather than a correction: no number in the prereg is "
            "wrong, the list was ambiguous about nesting.",
        "blind_spot_definition": "counted over maximal runs of pipe-leading lines: "
                                 "separator_less = no _TABLE_SEP in the run; header_less = the "
                                 "separator opens the run, so no header line precedes it; "
                                 "multi_separator = more than one separator inside one run, "
                                 "which the shipped header machinery treats as ONE table.",
        "per_document_ordinal_abstain_count": dict(collections.Counter(
            h["rel"] for h in clause_hits).most_common()),
        "note": "The regeneration surface is measured under the census's wider 16-entry "
                "vocabulary and is NOT what the clause fires on; the clause's own surface is "
                "the second tally. Every token in the first-minus-second difference stays "
                "OBLIGATED — a disclosed false-accusation surface, not a silenced one.",
    }


# ---------------------------------------------------------------- main

def main() -> int:                                                   # noqa: C901 - one report
    t0 = time.time()
    if not BASELINE.exists():
        print(f"missing {BASELINE.name}: run make_oath_v11_baseline.py on the PRE-change tree")
        return 2
    verifier = hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest()

    docs = resolvable_docs()
    print(f"frame: {len(docs)} documents")
    off_certs = frame_pass(docs, False)
    on_certs = frame_pass(docs, True)

    g1 = gate_g1(off_certs)
    g2 = gate_g2(on_certs)
    g3 = gate_g3(off_certs, on_certs)
    g4a = gate_g4a(off_certs)
    g4b = gate_g4b()
    print(f"  G1 {g1['verdict']}  G2 {g2['verdict']}  G3 {g3['verdict']}  "
          f"G4'a {g4a['verdict']}  G4'b {g4b['verdict']}")
    g5 = gate_g5(docs, off_certs, on_certs)
    print(f"  G5 {g5['verdict']}")
    g6 = gate_g6(docs, off_certs)
    g8 = gate_g8(on_certs)
    print(f"  G6 {g6['verdict']}  G8 {g8['verdict']}")
    set_v11(False)
    surfaces = boundary_surfaces(off_certs)
    set_v11(True)

    gates = [g1, g2, g3, g4a, g4b, g5, g6, g8]
    fails = [g["verdict"] for g in gates if g["verdict"].startswith("FAIL")]
    voids = [g["verdict"] for g in gates if g["verdict"].startswith("VOID")]
    # ABSENT and INCOMPLETE are the two ways a gate declines to score. Neither is a PASS and
    # neither is a FAIL: an unscored gate must never fall through to the ship outcome.
    absent = [g["gate"] for g in gates if g["verdict"] in ("ABSENT", "INCOMPLETE")]
    # Defensive on purpose: a gate that reports a bare "VOID"/"FAIL" must still produce a
    # written outcome. Losing the result JSON on the exact path a VOID exists to take is how a
    # pre-committed outcome token silently becomes unreachable.
    def _token(verdict: str, fallback: str) -> str:
        return verdict.split(":", 1)[1] if ":" in verdict else fallback

    if voids:
        outcome = _token(voids[0], "V11_BATTERY_VOID")
    elif absent:
        outcome = "V11_INCOMPLETE__" + ",".join(absent)
    elif fails:
        outcome = _token(fails[0], "V11_UNSPECIFIED_FAILURE")
    else:
        outcome = "V11_ORDINAL_RETRACTION_SHIPS"

    payload = {
        "battery": "OATH v0.11 — the row-ordinal retraction",
        "prereg": ("papers/closed-model-frontier/"
                   "PREREG_oath_v11_row_ordinal_retraction_2026_08_25.md"),
        "prereg_frozen_on_commit": "cbd2864",
        "verifier_sha256": verifier,
        "baseline_verifier_sha256":
            json.loads(BASELINE.read_text(encoding="utf-8"))["verifier_sha256"],
        "clause": "V11_ORDINAL_LABEL",
        "reason_code": REASON,
        "outcome": outcome,
        "gates": gates,
        "boundary_surfaces_for_G9": surfaces,
        "out_of_battery": {
            "G7": "SUITE CLOSURE — a property of the ship commit, not of a harness run.",
            "G9": "BOUNDARY DISCLOSURE — a property of the RESULT document; its re-derived "
                  "numbers are computed above under boundary_surfaces_for_G9.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nOUTCOME: {outcome}   ({payload['elapsed_s']}s) -> {OUT.name}")
    return 0 if outcome == "V11_ORDINAL_RETRACTION_SHIPS" else 1


if __name__ == "__main__":
    sys.exit(main())
