"""Run the resolution probes against styxx.claim_audit — today's HEAD vs this morning's HEAD.

The probes are only worth something if they SEPARATE a known-defective instrument from a fixed
one. This runs both: the module as it stands (after 1fb1de5 / 4de77d1 / af62490) and the module
as it was at 18b8c61 this morning, loaded from git into a temp file.

If the pre-fix module passes the probes, the probes are decoration and this script says so.
"""
from __future__ import annotations
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from resolution_probe import run_suite  # noqa: E402

PRE_FIX_COMMIT = "18b8c61"   # the C6 prereg commit, before any claim_audit work today


def load_module_from_commit(commit: str, name: str):
    src = subprocess.run(["git", "show", f"{commit}:styxx/claim_audit.py"],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if src.returncode != 0:
        raise RuntimeError(f"git show failed: {src.stderr[:200]}")
    tmp = pathlib.Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def adapters(mod, modern: bool):
    """Read the four quantities the probes need out of whichever module version."""
    def audit(doc, src):
        return mod.audit_grounding(doc, src)

    def rate_of(rep):
        n = rep.n_total or 1
        return (rep.n_grounded + rep.n_derived) / n

    def candidates_of(rep):
        items = [i for i in rep.items if i.status == "grounded"]
        if not items:
            return 0
        if modern:
            return getattr(items[0], "n_candidates", 1)
        return 1  # pre-fix module has no notion of multiple candidates

    def source_of(rep):
        items = [i for i in rep.items if i.status == "grounded"]
        return items[0].source if items else ""

    def ambiguity_of(rep):
        # Does the instrument DISCLOSE that the match was not unique?
        if getattr(rep, "n_ambiguous", 0) > 0:
            return True
        s = rep.summary().lower()
        return "more than one source path" in s or "arbitrary" in s

    def discloses_floor(rep):
        """Does the instrument state a chance floor next to its rate?"""
        if getattr(rep, "chance_floor", None) is None:
            return False
        s = rep.summary().lower()
        return "chance floor" in s or "excess over chance" in s

    return dict(rate_of=rate_of, candidates_of=candidates_of,
                source_of=source_of, ambiguity_of=ambiguity_of,
                discloses_floor=discloses_floor), audit


def main():
    out = {}

    pre = load_module_from_commit(PRE_FIX_COMMIT, "claim_audit_prefix")
    ad, audit = adapters(pre, modern=False)
    pre_res = run_suite(audit, label=f"styxx.claim_audit @ {PRE_FIX_COMMIT} (BEFORE today)", **ad)
    out["before"] = {r.name: {"verdict": r.verdict, **r.detail} for r in pre_res}

    print()
    import styxx.claim_audit as post
    ad2, audit2 = adapters(post, modern=True)
    post_res = run_suite(audit2, label="styxx.claim_audit @ HEAD (AFTER today)", **ad2)
    out["after"] = {r.name: {"verdict": r.verdict, **r.detail} for r in post_res}

    pre_fail = sum(1 for r in pre_res if r.verdict == "FAIL")
    post_fail = sum(1 for r in post_res if r.verdict == "FAIL")
    print("\n" + "=" * 78)
    print(f"SEPARATION: before {pre_fail} FAIL -> after {post_fail} FAIL")
    if pre_fail > post_fail:
        verdict = "PROBES_DISCRIMINATE__they_separate_the_known_defective_from_the_fixed"
    elif pre_fail == post_fail == 0:
        verdict = "PROBES_ARE_DECORATION__even_the_known_defective_version_passes"
    else:
        verdict = "INCONCLUSIVE__fixes_did_not_move_the_probes"
    print(f"VERDICT: {verdict}")
    out["separation"] = {"before_fails": pre_fail, "after_fails": post_fail, "verdict": verdict}
    (HERE / "resolution_probe_result.json").write_text(json.dumps(out, indent=2) + "\n",
                                                       encoding="utf-8")
    print("wrote resolution_probe_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
