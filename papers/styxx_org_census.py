"""The org chart, measured — every box's status computed from the repository.

## Why this exists

Personal-AI-org diagrams are having a moment. The genre draws a human operator, an orchestration
layer, and a fan of specialist agents, each box carrying a status light that says ONLINE or
BOOTING. The lights are typed by whoever drew the diagram. Nothing checks them, and "BOOTING" is
usually a polite way of saying the box does not exist.

A status that asserts its own health is the defect class this repository exists to document. So
this is the same picture with the lights wired up:

* **`implemented`** — the module is on disk, with its line count.
* **`tests`** — its test files are RUN, here, now, and the pass/fail count is recorded.
* **`receipts`** — how many committed JSON receipts it has produced.
* **`disclosed_defect`** — what is currently wrong with it, and **the citation is verified**: the
  document naming the defect must exist on disk or the role is marked `CITATION_MISSING`.

No box can claim ONLINE because someone typed ONLINE. And every box that works still says what is
broken about it, because a chart on which everything is green is the same genre of lie.

  python papers/styxx_org_census.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "styxx_org_census.json"
CMF = "closed-model-frontier"

# role -> (layer, one-line job, module path, test files, defect, document naming the defect)
ROLES = [
    ("PROTOCOL FREEZER", "orchestration",
     "refuses to score a cycle whose preregistration is not committed",
     "styxx/protocol.py", ["tests/test_protocol_vacuity.py"],
     "its vacuity check is structural only; an unfailable BAR is not detected",
     "papers/closed-model-frontier/RECON_v13_not_frozen_2026_08_27.md"),

    ("VERIFIER", "core",
     "binds every numeric claim in a document to a committed receipt",
     "styxx/certify.py", ["tests/test_certify_row_ordinal.py", "tests/test_certify_recall.py"],
     "~1 in 5 of its verifications are sworn to non-claims, and it declines to check "
     "0.4267 of the checkable claims in our own corpus",
     f"papers/{CMF}/RESULT_oath_verified_channel_internal_2026_08_27.md"),

    ("CORPUS AUDITOR", "core",
     "re-certifies every committed certificate and reports drift",
     "styxx/corpus_audit.py", ["tests/test_corpus_audit_search_scope.py"],
     "was enumerating a population 49% of which was an agent scratch clone; one CAPSTONE "
     "receipt remains present-and-changed",
     "REPLICATIONS.md"),

    ("NULL-RULE CHECK", "method",
     "asks whether a candidate rule beats doing nothing at all",
     "styxx/discriminates.py", ["tests/test_discriminates.py"],
     "cannot see overfitting, and a rule that does nothing passes a cost column",
     f"papers/{CMF}/RECON_obligation_repair_is_not_lexical_2026_08_27.md"),

    ("READINESS CHECK", "surface",
     "tells an outside author whether their document can carry a certificate",
     "styxx/oathready.py", ["tests/test_oathready.py"],
     "its abstained bucket is not neutral: ~2 in 5 of those rows are unchecked claims",
     "OATH_CONTRACT.md"),

    ("ABSENCE DETECTOR", "core",
     "finds the measurement that was never taken",
     "styxx/absence.py", ["tests/test_absence.py"],
     None, None),

    ("LOOP DETECTOR", "core",
     "finds the agent that is repeating itself",
     "styxx/loops.py", ["tests/test_loops.py"],
     None, None),

    ("LEDGER", "record",
     "regenerates the public record of every cycle from the receipts",
     "papers/build_ledger.py", ["tests/test_ledger.py", "tests/test_ledger_classifier.py"],
     "the negatives ratio is a keyword match over prose, and 0 of 163 cycles carry a "
     "machine-readable verdict token",
     "papers/LEDGER.md"),

    ("SILENT-PASS BENCH", "method",
     "the corpus of outcomes that do not happen while every check stays green",
     "benchmarks/silent_pass/__init__.py", ["tests/test_silent_pass_bench.py"],
     "was itself skipping in CI on every shallow checkout until 2026-08-28",
     "tests/conftest.py"),

    ("BLIND PANEL", "method",
     "three seats adjudicate claimhood, salted with decoys so membership leaks nothing",
     f"papers/{CMF}/oath_adjudication.py", ["tests/test_adjudication_packets.py"],
     "three seats of one model family; 98% unanimity is a correlated-error ceiling, "
     "not agreement between readers",
     f"papers/{CMF}/RESULT_oath_external_corpus_2026_08_27.md"),

    # POSITIVE CONTROL, declared as one. The chart that prompted this drew a TRADER box reading
    # "paper trading -> live" with a BOOTING lamp. Nothing was behind it; BOOTING was the label
    # for a box that did not exist. This role is here to prove the status column CAN FAIL: it
    # points at a module that is not on disk and must come back ABSENT. If it ever reports
    # anything else, the census is lying and main() raises rather than publishing a chart on
    # which everything is green.
    ("TRADER", "control",
     "declared positive control - the aspirational box, drawn to prove ABSENT is reachable",
     "styxx/trader.py", ["tests/test_trader.py"],
     "does not exist. Retail algorithmic trading is a backtest away from the in-sample "
     "collapse this lane measured on 2026-08-27, and no edge here transfers to markets",
     f"papers/{CMF}/RECON_obligation_repair_is_not_lexical_2026_08_27.md"),
]


# A module that states its own limits in its source is disclosing without being asked, and that
# is checkable rather than curated. Added after the first version of this census recorded two
# roles as "no defect disclosed" when both modules carry an explicit limits section -- the
# omission was the author's, not theirs, which is exactly the failure a computed field prevents.
DISCLOSURE_MARKERS = (
    "LIMITS = ", "LIMITS: ", "What it CANNOT see", "What it cannot see",
    "The honest limits", "known_limits", "Known limits", "what_this_does_not_show",
)


def self_disclosed_limits(path: Path) -> dict:
    """Does the module state its own limits, in its own source?"""
    if not path.exists():
        return {"discloses": False, "markers": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    found = [m.strip() for m in DISCLOSURE_MARKERS if m in text]
    return {"discloses": bool(found), "markers": found}


def run_tests(files: list[str]) -> dict:
    present = [f for f in files if (ROOT / f).exists()]
    if not present:
        return {"ran": False, "reason": "no test file on disk", "passed": 0, "failed": 0}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", *present],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    import re
    out = r.stdout or ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    return {"ran": True, "passed": passed, "failed": failed, "exit": r.returncode,
            "seconds": round(time.time() - t0, 1), "files": present}


def main() -> int:
    roles = []
    for name, layer, job, module, tests, defect, doc in ROLES:
        mod = ROOT / module
        implemented = mod.exists()
        loc = len(mod.read_text(encoding="utf-8", errors="replace").splitlines()) \
            if implemented else 0
        stem = Path(module).stem
        receipts = sorted(p.name for p in (ROOT / "papers").rglob(f"*{stem.replace('build_', '')}*.json")
                          if ".claude" not in p.parts)

        print(f"  testing {name} ...", flush=True)
        t = run_tests(tests)

        # The citation is CHECKED. A defect claim whose document has vanished is worse than no
        # defect claim, because it reads as disclosure while disclosing nothing.
        citation_ok = None
        if doc is not None:
            citation_ok = (ROOT / doc).exists()
        self_disc = self_disclosed_limits(mod)

        if not implemented:
            status = "ABSENT"
        elif not t["ran"] or t["failed"] or t["exit"] != 0:
            status = "FAILING"
        elif doc is not None and not citation_ok:
            status = "CITATION_MISSING"
        elif defect is not None:
            status = "ONLINE_WITH_DISCLOSED_DEFECT"
        elif self_disc["discloses"]:
            status = "ONLINE_SELF_DISCLOSES_LIMITS"
        else:
            status = "ONLINE_NO_DEFECT_DISCLOSED"

        roles.append({
            "role": name, "layer": layer, "job": job, "module": module,
            "implemented": implemented, "loc": loc,
            "tests": t, "receipts_produced": len(receipts),
            "disclosed_defect": defect, "defect_document": doc,
            "self_disclosed_limits": self_disc,
            "defect_document_exists": citation_ok,
            "status": status,
        })

    tally = {}
    for r in roles:
        tally[r["status"]] = tally.get(r["status"], 0) + 1

    # The control must fail, or the status column is decoration. This is the same discipline the
    # rest of the lane applies to every census: a column that cannot report a bad value is not
    # reporting a good one either.
    control = next(r for r in roles if r["layer"] == "control")
    if control["status"] != "ABSENT":
        raise SystemExit(
            f"positive control {control['role']} reported {control['status']}, expected ABSENT. "
            f"The status column cannot be shown to fail, so no chart may be published from this "
            f"run.")

    payload = {
        "chart": "the styxx research org, with every status computed rather than typed",
        "why": ("Personal-AI-org diagrams type their own status lights. This one runs the tests, "
                "counts the receipts, and verifies that each disclosed defect's document exists. "
                "A box cannot claim ONLINE because someone typed ONLINE."),
        "human_operator": ("one. merges to main, and nothing in this repository merges itself"),
        "roles_total": len(roles),
        "status_tally": tally,
        "tests_run_here": sum(r["tests"]["passed"] for r in roles),
        "tests_failed_here": sum(r["tests"]["failed"] for r in roles),
        "roles_disclosing_a_defect": sum(1 for r in roles if r["disclosed_defect"]),
        "roles_with_no_defect_disclosed": sum(
            1 for r in roles if not r["disclosed_defect"]
            and not r["self_disclosed_limits"]["discloses"]),
        "roles_self_disclosing_limits_in_source": sum(
            1 for r in roles if r["self_disclosed_limits"]["discloses"]),
        "positive_control": {
            "role": control["role"], "status": control["status"],
            "asserted": "must report ABSENT",
            "why": ("A status column that cannot report a bad value is decoration. This role "
                    "points at a module that is not on disk. On the first run of this census the "
                    "column ALSO caught a real error by its author -- a defect citation pointing "
                    "at papers/REPLICATIONS.md, which is at the repository root -- and reported "
                    "CITATION_MISSING until it was fixed. The column has now failed twice on "
                    "purpose and once by accident."),
        },
        "the_point": ("A chart on which every light is green is the same genre of lie as a status "
                      "nobody checks. Every working role here also says what is wrong with it, and "
                      "the two roles that disclose nothing are marked as such rather than as "
                      "healthy."),
        "roles": roles,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{'role':<22}{'status':<32}{'loc':<7}{'tests':<8}{'receipts'}")
    for r in roles:
        print(f"{r['role']:<22}{r['status']:<32}{r['loc']:<7}"
              f"{str(r['tests']['passed']):<8}{r['receipts_produced']}")
    print()
    print(f"{payload['tests_run_here']} tests run here, {payload['tests_failed_here']} failed")
    print(f"{payload['roles_disclosing_a_defect']} of {len(roles)} roles disclose a defect")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
