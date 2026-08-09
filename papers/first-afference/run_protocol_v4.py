"""Protocol v4 exam, per PREREG_protocol_v4_composition_2026_08_09.md.

Mutant battery + valid cases + the E1 retro-case + whole-corpus byte-identity. Unlike run_p1.py
— which claimed to reconstruct from receipts and never opened one — the retro-case here READS the
committed ``e1_result.json`` and the corpus check re-scores every committed result file it can
pair with its prereg. Every case is listed in the receipt with what it did.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.protocol import Experiment, GateSpecError  # noqa: E402

PREREG = "PREREG_protocol_v4_composition_2026_08_09.md"
SMOKE = "--smoke" in sys.argv


def _mk_prereg(tmp: Path, gates_json: str) -> Path:
    """Write a throwaway prereg carrying the given gates block and commit it (the freeze check
    requires a commit; the temp repo is separate from the real one)."""
    p = tmp / "PREREG_case.md"
    p.write_text("# case\n\n```gates\n" + gates_json + "\n```\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
    subprocess.run(["git", "-c", "user.email=exam@local", "-c", "user.name=exam",
                    "commit", "-qm", "case"], cwd=tmp, check=True)
    return p


BASE = {"gates": {"G": {"metric": "m", "op": "<=", "value": 0.2,
                        "agg": "min", "over": "pool", "excluding": "dq"}},
        "outcomes": [{"when": {"G": True}, "verdict": "PASS"},
                     {"when": {"G": False}, "verdict": "FAIL"}],
        "smoke_verdict": "SMOKE"}


def _case(gates=None, result=None) -> tuple[bool, str]:
    """Run one constructed case. Returns (refused, detail)."""
    spec = json.loads(json.dumps(BASE))
    if gates:
        spec["gates"]["G"].update(gates)
        for k, v in list(spec["gates"]["G"].items()):
            if v is None:
                del spec["gates"]["G"][k]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        p = _mk_prereg(tmp, json.dumps(spec))
        try:
            v = Experiment(p).score(result)
            return False, f"scored {v.verdict}"
        except GateSpecError as e:
            return True, f"refused: {str(e)[:100]}"


def main() -> int:
    ok_pool = {"a": 0.30, "b": 0.15, "c": 0.25}      # min over all = 0.15; min excl b = 0.25

    violations = {
        "e1_shape_quotes_unrestricted_min":
            _case(result={"m": 0.15, "pool": ok_pool, "dq": ["b"]}),
        "over_path_missing":
            _case(result={"m": 0.25, "dq": ["b"]}),
        "over_not_a_dict":
            _case(result={"m": 0.25, "pool": [0.3, 0.15], "dq": ["b"]}),
        "over_empty_dict":
            _case(result={"m": 0.25, "pool": {}, "dq": ["b"]}),
        "excluding_unknown_member":
            _case(result={"m": 0.15, "pool": ok_pool, "dq": ["zz"]}),
        "excluding_not_a_list":
            _case(result={"m": 0.25, "pool": ok_pool, "dq": "b"}),
        "every_member_excluded":
            _case(result={"m": 0.25, "pool": ok_pool, "dq": ["a", "b", "c"]}),
        "member_value_nan":
            _case(result={"m": 0.25, "pool": {"a": 0.3, "b": float("nan")}, "dq": []}),
        "member_value_bool":
            _case(result={"m": 0.25, "pool": {"a": 0.3, "b": True}, "dq": []}),
        "agg_not_min_or_max":
            _case(gates={"agg": "mean"}, result={"m": 0.2333, "pool": ok_pool, "dq": []}),
        "half_declaration_agg_without_over":
            _case(gates={"over": None, "excluding": None},
                  result={"m": 0.15, "pool": ok_pool}),
        "quoted_off_by_rounding_1e6":
            _case(result={"m": 0.250001, "pool": ok_pool, "dq": ["b"]}),
        "max_agg_quotes_excluded_max":
            _case(gates={"agg": "max", "op": ">=", "value": 0.1},
                  result={"m": 0.30, "pool": ok_pool, "dq": ["a"]}),
        "mixed_type_excluding_list":
            _case(result={"m": 0.25, "pool": ok_pool, "dq": [1, "b"]}),
    }

    # ---- parser-shadowing mutants (red team D1/D2): the frozen document must not be able to
    # show a reader one gates block while the machine scores another. These need raw prereg
    # text, not the BASE-spec path.
    def _raw_case(md_text, result):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            p = tmp / "PREREG_case.md"
            p.write_text(md_text, encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "-A"], cwd=tmp, check=True)
            subprocess.run(["git", "-c", "user.email=exam@local", "-c", "user.name=exam",
                            "commit", "-qm", "case"], cwd=tmp, check=True)
            try:
                v = Experiment(p).score(result)
                return False, f"scored {v.verdict}"
            except GateSpecError as e:
                return True, f"refused: {str(e)[:100]}"

    honest = json.dumps(BASE)
    decoy = json.dumps({"gates": {"G": {"metric": "m", "op": "<=", "value": 99}},
                        "outcomes": [{"when": {"G": True}, "verdict": "PASS"},
                                     {"when": {"G": False}, "verdict": "FAIL"}],
                        "smoke_verdict": "SMOKE"})
    e1_shape = {"m": 0.15, "pool": ok_pool, "dq": ["b"]}
    F = "```"
    violations["hidden_fence_in_html_comment"] = _raw_case(
        f"# doc\n\n<!--\n{F}gates\n{decoy}\n{F}\n-->\n\n{F}gates\n{honest}\n{F}\n", e1_shape)
    violations["second_display_fence"] = _raw_case(
        f"# doc\n\n{F}gates\n{honest}\n{F}\n\nexample:\n\n{F}gates\n{decoy}\n{F}\n", e1_shape)
    violations["duplicate_excluding_key"] = _raw_case(
        f"# doc\n\n{F}gates\n"
        + honest.replace('"excluding": "dq"', '"excluding": "dq", "excluding": "dq_decoy"')
        + f"\n{F}\n", {"m": 0.15, "pool": ok_pool, "dq": ["b"], "dq_decoy": []})
    violations["duplicate_gates_toplevel_key"] = _raw_case(
        f"# doc\n\n{F}gates\n" + honest[:-1] + ', "gates": '
        + json.dumps({"G": {"metric": "m", "op": "<=", "value": 99}}) + "}" + f"\n{F}\n",
        e1_shape)
    # Verification-pass F1/F2 vectors: the fence guard must count blocks a HUMAN could read as
    # gates, and keys a human cannot distinguish must refuse.
    violations["tilde_honest_fence_hidden_backtick_decoy"] = _raw_case(
        f"# doc\n\n<!--\n{F}gates\n{decoy}\n{F}\n-->\n\n~~~gates\n{honest}\n~~~\n", e1_shape)
    violations["uppercase_gates_fence_only"] = _raw_case(
        f"# doc\n\n{F}GATES\n{honest}\n{F}\n", e1_shape)
    violations["zero_width_char_in_fence_info"] = _raw_case(
        f"# doc\n\n{F}ga​tes\n{honest}\n{F}\n", e1_shape)
    violations["fence_only_inside_html_comment"] = _raw_case(
        f"# doc\n\n<!--\n{F}gates\n{decoy}\n{F}\n-->\n", e1_shape)
    violations["homoglyph_excluding_key"] = _raw_case(
        f"# doc\n\n{F}gates\n"
        + honest.replace('"excluding": "dq"',
                         '"exсluding": "dq", "excluding": "dq_decoy"')
        + f"\n{F}\n", {"m": 0.15, "pool": ok_pool, "dq": ["b"], "dq_decoy": []})
    # Round-3 vectors: the counter and the extractor must be ONE scanner.
    violations["cyrillic_fence_honest_then_ascii_evil"] = _raw_case(
        f"# doc\n\n{F}gаtes\n{honest}\n{F}\n\n{F}gates\n{decoy}\n{F}\n", e1_shape)
    violations["tab_indented_opener_with_commented_honest"] = _raw_case(
        f"# doc\n\n<!--\n{F}gates\n{honest}\n{F}\n-->\n\n\t{F}gates\n{decoy}\n{F}\n", e1_shape)
    violations["unterminated_comment_hides_block_from_renderer"] = _raw_case(
        f"# doc\n\n<!--\n\n{F}gates\n{honest}\n{F}\n", e1_shape)
    violations["unclosed_gates_fence"] = _raw_case(
        f"# doc\n\n{F}gates\n{honest}\n", e1_shape)

    valids = {
        "correct_min_with_exclusion":
            _case(result={"m": 0.25, "pool": ok_pool, "dq": ["b"]}),
        "correct_min_no_exclusion_list_empty":
            _case(result={"m": 0.15, "pool": ok_pool, "dq": []}),
        "correct_without_excluding_key":
            _case(gates={"excluding": None}, result={"m": 0.15, "pool": ok_pool}),
        "correct_max":
            _case(gates={"agg": "max", "op": ">=", "value": 0.1},
                  result={"m": 0.25, "pool": ok_pool, "dq": ["a"]}),
        "no_declaration_at_all_still_scores":
            _case(gates={"agg": None, "over": None, "excluding": None},
                  result={"m": 0.15}),
        "single_member_after_exclusion":
            _case(result={"m": 0.30, "pool": ok_pool, "dq": ["b", "c"]}),
    }

    # ---- the E1 retro-case: the committed receipt, a v4 declaration, must REFUSE ------------
    e1 = json.loads((HERE / "e1_result.json").read_text(encoding="utf-8"))
    retro_spec = {"gates": {"G1": {"metric": "best_median_abs_rel_error", "op": "<=",
                                   "value": 0.20, "agg": "min",
                                   "over": "pooled_median_abs_rel_error",
                                   "excluding": "disqualified_by_silent_probe"}},
                  "outcomes": [{"when": {"G1": True}, "verdict": "USABLE"},
                               {"when": {"G1": False}, "verdict": "NOT_ESTIMABLE"}],
                  "smoke_verdict": "SMOKE"}
    with tempfile.TemporaryDirectory() as td:
        p = _mk_prereg(Path(td), json.dumps(retro_spec))
        try:
            v = Experiment(p).score(e1)
            retro = (False, f"scored {v.verdict} — THE DEFECT SURVIVED")
        except GateSpecError as e:
            retro = (True, f"refused: {str(e)[:160]}")

    # ---- corpus byte-identity ---------------------------------------------------------------
    diffs, checked = [], 0
    pairs = []
    for res_file in sorted(ROOT.glob("papers/*/*_result.json")):
        try:
            d = json.loads(res_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        pr, old = d.get("prereg"), d.get("verdict")
        if not isinstance(pr, str) or not isinstance(old, str) or old.startswith("UNSCORED"):
            continue
        # Only results whose verdict styxx.protocol itself produced are in scope: protocol
        # stamps prereg_commit and gates on every verdict it returns. Results older than the
        # protocol lane were scored by other mechanisms against preregs with no gates block,
        # and re-scoring them here would report OUR pairing error as a v4 regression (which is
        # exactly what the first run of this exam did: 47 spurious diffs, all GateSpecError).
        if "prereg_commit" not in d or "gates" not in d:
            continue
        prereg_path = res_file.parent / pr
        if not prereg_path.exists():
            continue
        pairs.append((prereg_path, res_file, d, old))
    if SMOKE:
        pairs = pairs[:5]
    for prereg_path, res_file, d, old in pairs:
        try:
            new = Experiment(prereg_path).score(d, smoke=bool(d.get("smoke"))).verdict
        except Exception as e:
            new = f"RAISED:{type(e).__name__}"
        checked += 1
        if new != old:
            diffs.append({"result": res_file.name, "was": old, "now": new})

    res = {"prereg": PREREG, "smoke": SMOKE,
           "refusal_site_note": ("agg_not_min_or_max and half_declaration_agg_without_over "
                                 "refuse at Experiment construction (spec validation), not in "
                                 "score(); the frozen metric_means says 'score() raised' and "
                                 "is imprecise on those two (red team D7). Both refusals occur "
                                 "before any verdict, which is the property the gate protects."),
           "violation_cases": {k: {"refused": r, "detail": s} for k, (r, s) in violations.items()},
           "valid_cases": {k: {"scored": not r, "detail": s} for k, (r, s) in valids.items()},
           "e1_retro_case": {"refused": retro[0], "detail": retro[1]},
           "n_violation_mutants": len(violations),
           "frac_violation_mutants_refused": round(
               sum(r for r, _ in violations.values()) / len(violations), 4),
           "n_valid_cases": len(valids),
           "frac_valid_cases_scored": round(
               sum(not r for r, _ in valids.values()) / len(valids), 4),
           "e1_retro_case_refused": 1.0 if retro[0] else 0.0,
           "n_corpus_results_rescored": checked,
           "n_corpus_verdict_diffs": len(diffs), "corpus_diffs": diffs}

    try:
        e = Experiment(HERE / PREREG, require_power_basis=True)
        res["metric_check"] = e.check_metrics(res)
        bad = sorted(n for n, dd in res["metric_check"].items() if not dd["usable"])
        if bad and not SMOKE:
            raise SystemExit(f"unresolvable gate metrics: {bad}")
        v = e.score(res, smoke=SMOKE)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as exc:
        res["verdict"] = f"UNSCORED__{type(exc).__name__}: {exc}"

    (HERE / f"protocol_v4_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"violations refused {res['frac_violation_mutants_refused']} of {len(violations)} | "
          f"valid scored {res['frac_valid_cases_scored']} of {len(valids)}")
    print(f"E1 retro: {retro[1][:120]}")
    print(f"corpus: {checked} rescored, {len(diffs)} diffs")
    for d in diffs[:5]:
        print("  DIFF:", d)
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
