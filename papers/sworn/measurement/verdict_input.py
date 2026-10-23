# -*- coding: utf-8 -*-
"""The projection from score.py's measurement_result.json to the numbers the
leg-2 PREREG's gates block reads, and the one entry point that turns a fold into its verdict
(`styxx.protocol` resolves dotted paths into dicts only, refuses a non-number, and cannot index a
list, so `gates.G_C.wilson95[0]` and the literal `WITHHELD` cannot be read in place).

Committed beside the PREREG, in the signing commit, as papers/sworn/measurement/verdict_input.py;
its blob is in the PREREG's lock (section 5). It reads one fold's measurement_result.json and
writes nothing unless `--out` names a file that does not exist yet. Every rate is recomputed from
the counts score.py printed, with score.py's own rounding (`round(k / n, 4)`, score.py l.78-81);
the Wilson lower bound is `styxx.mind.wilson` unrounded, as score.py l.421 and l.428 compare it.

G-F is read per panel (SPEC l.467), and the WITHHELD rule per panel (SPEC l.481-485): fewer than
two families clearing Panel L withholds G-S1 and G-S2; fewer than two clearing Panel R withholds
G-P; G-C and G-G1 are decided regardless. A withheld share is carried by a fail-closed sentinel,
-1.0 under a `>=` bar and 2.0 under a `<=` bar, so it never reads as a pass, and the gates block's
VOID rows do not consult it. A share whose panel qualified but whose denominator is 0 carries the
same sentinel and reads as a miss; it is listed in `empty_denominators`, and the RESULT prints
"0 of 0" and "the bar was not tested" beside the title (PREREG section 6).

stdlib only at import; `main()` imports styxx.protocol and styxx.mind from the repository it runs
in. No model; no network.

CLI: python papers/sworn/measurement/verdict_input.py --tranche A
     [--result papers/sworn/measurement/measurement_result.json] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

SCHEMA = "styxx-sworn/leg2-verdict-input/v1"
RECORD_SCHEMA = "styxx-sworn/leg2-verdict/v1"
FAMILIES = ("claude", "local")
PANELS = ("L", "R")
N_DECOYS = 30                 # common.py N_DECOYS_PER_SIDE = 15, two sides
N_SIDE = 15
LOW = -1.0                    # fail-closed sentinel under a ">=" bar
HIGH = 2.0                    # fail-closed sentinel under a "<=" bar
PREREG_GLOB = "papers/sworn/PREREG_sworn_measurement_*.md"     # common.py l.575-577, the lowest sort wins
N_TRANCHE_A = 9               # population.json pinned at f28d35ad
TRANCHE_B_MAX = 4
TRANCHE_PACKET_L = {"A": "papers/sworn/measurement/packet_L.json",
                    "B": "papers/sworn/measurement/tranche_B/packet_L.json"}
HERE = Path(__file__).resolve().parent


class Refused(SystemExit):
    pass


def _rate(k: int, n: int) -> float:
    return round(k / n, 4)    # score.py `_rate`, l.78-81


def tranche_check(result: dict, tranche: str) -> None:
    """The fold is the tranche it is labelled: score.py's lock names the Panel L packet it read
    (score.py l.534-538, `common.bind_inputs`), and its population counts the documents (l.539)."""
    read = [r.get("path") for r in (result.get("lock") or {}).get("inputs", [])
            if str(r.get("path", "")).endswith("/packet_L.json")]
    if read != [TRANCHE_PACKET_L[tranche]]:
        raise Refused("REFUSED: --tranche %s, but the fold read %s, not %s" % (tranche, read, TRANCHE_PACKET_L[tranche]))
    n = (result.get("population") or {}).get("documents")
    if not (n == N_TRANCHE_A if tranche == "A" else isinstance(n, int) and 1 <= n <= TRANCHE_B_MAX):
        raise Refused("REFUSED: --tranche %s, but the fold has %r documents" % (tranche, n))


def project(result: dict, wilson: Callable[[int, int], Tuple[float, float]],
            tranche: str, path_check_only: bool = False) -> Dict:
    """One fold's measurement_result.json -> the verdict input. Refuses a dry run unless
    `path_check_only`, in which case every value is its fail-closed sentinel and no rate is
    computed from the dry run's counts."""
    if tranche not in ("A", "B"):
        raise Refused("REFUSED: tranche must be A or B")
    dry = bool(result.get("dry_run"))
    if dry and not path_check_only:
        raise Refused("REFUSED: a dry run has no verdict input")
    if not dry and not result.get("prereg"):
        raise Refused("REFUSED: a fold with no PREREG at HEAD has no verdict input")
    if not dry:
        tranche_check(result, tranche)
    g = result["gates"]
    out = {"schema": SCHEMA, "tranche": tranche, "path_check_only": bool(path_check_only),
           "g_d": {}, "g_f": {}, "g_s1": {}, "g_s2": {}, "g_p": {}, "g_c": {}, "g_g1": {},
           "withheld": [], "empty_denominators": []}

    for f in FAMILIES:
        for p in PANELS:
            row = g["G_D"][p][f]
            if dry or not row.get("present"):
                overall, min_side = 0, 0
            else:
                sides = row["per_side"]
                if row["n_overall"] != N_DECOYS or len(sides) != 2 or any(v["n"] != N_SIDE for v in sides.values()):
                    raise Refused("REFUSED: G_D %s/%s has a decoy denominator other than 30 = 15 + 15" % (f, p))
                overall = int(row["correct_overall"])
                min_side = int(min(v["correct"] for v in sides.values()))
            out["g_d"]["%s_%s" % (f, p)] = {"overall": overall, "min_side": min_side}

    fc = g["G_F"]["families_clearing"]
    for p in PANELS:
        out["g_f"][p] = 0 if dry else int(fc[p])
    qualified = {p: (not dry) and fc[p] == 2 for p in PANELS}

    def share(name: str, panel: str, sentinel: float) -> float:
        gate = g[name]
        num, den = gate.get("numerator"), gate.get("denominator")
        if not qualified[panel]:
            out["withheld"].append(name)
            return sentinel
        if not isinstance(den, int) or den == 0:
            out["empty_denominators"].append(name)
            return sentinel
        return _rate(int(num), den)

    out["g_s1"]["share"] = share("G_S1", "L", LOW)
    out["g_s2"]["share"] = share("G_S2", "L", HIGH)
    out["g_p"]["share"] = share("G_P", "R", HIGH)

    k, n = g["G_C"]["k"], g["G_C"]["n"]
    if dry or not n:
        out["g_c"]["wilson_lower"] = LOW
        if not dry:
            out["empty_denominators"].append("G_C")
    else:
        out["g_c"]["wilson_lower"] = float(wilson(int(k), int(n))[0])

    pairs, lower = g["G_G1"]["pairs"], g["G_G1"]["lower"]
    if dry or not pairs:
        out["g_g1"]["share"] = LOW
        if not dry:
            out["empty_denominators"].append("G_G1")
    else:
        out["g_g1"]["share"] = _rate(int(lower), int(pairs))
    return out


def crosscheck(result: dict, fired: Dict[str, bool]) -> Dict[str, str]:
    """After `styxx.protocol.Experiment.score`: every gate score.py also decided must agree with
    the frozen block, or the fold is refused. Returns {gate: 'agrees' | 'score.py undecided: ...'}."""
    g = result["gates"]
    report = {}
    for f in FAMILIES:
        for p in PANELS:
            sp = g["G_D"][p][f].get("pass")
            both = fired["G_D_%s_%s_overall" % (f, p)] and fired["G_D_%s_%s_side" % (f, p)]
            if isinstance(sp, bool) and sp != both:
                raise Refused("REFUSED: G_D %s/%s: score.py says %s, the frozen block says %s" % (f, p, sp, both))
    fc = g["G_F"]["families_clearing"]
    for p in PANELS:
        if (fc[p] == 2) != fired["G_F_%s" % p]:
            raise Refused("REFUSED: G_F_%s: score.py counts %s families clearing, the frozen block says %s"
                          % (p, fc[p], fired["G_F_%s" % p]))
    gf = g["G_F"].get("pass")                  # score.py l.522-525: both panels, printed beside, never gated
    if isinstance(gf, bool) and gf != (fired["G_F_L"] and fired["G_F_R"]):
        raise Refused("REFUSED: G_F: score.py's whole-fold pass is %s, the two panel gates say %s"
                      % (gf, fired["G_F_L"] and fired["G_F_R"]))
    report["G_F_L"] = report["G_F_R"] = "agrees"
    for name in ("G_S1", "G_S2", "G_P", "G_C", "G_G1"):
        sp = g[name].get("pass")
        if isinstance(sp, bool):
            if sp != fired[name]:
                raise Refused("REFUSED: %s: score.py says %s, the frozen block says %s" % (name, sp, fired[name]))
            report[name] = "agrees"
        elif g[name].get("share") == "WITHHELD":
            report[name] = "score.py undecided: withheld"
        else:
            report[name] = "score.py undecided: empty denominator"
    return report


def run(result: dict, prereg_rel: str, repo_root, tranche: str, experiment_cls, wilson) -> Dict:
    """The whole verdict path for one fold: project, score under the frozen block, cross-check."""
    if result.get("prereg") != prereg_rel:
        raise Refused("REFUSED: the fold was scored under %r, the PREREG at HEAD is %r"
                      % (result.get("prereg"), prereg_rel))
    vin = project(result, wilson, tranche)
    exp = experiment_cls(Path(repo_root) / prereg_rel, repo_root=repo_root,
                         require_power_basis=True, require_nonvacuous_gates=True)
    v = exp.score(vin)
    agree = crosscheck(result, v.gates)
    return {"schema": RECORD_SCHEMA, "tranche": tranche, "prereg": prereg_rel,
            "prereg_commit": v.prereg_commit, "gates_sha256": v.gates_sha256, "verdict": v.verdict,
            "gates": dict(v.gates), "crosscheck": agree, "withheld": list(vin["withheld"]),
            "empty_denominators": list(vin["empty_denominators"]),
            "score_py_quotable": result.get("quotable"), "verdict_input": vin}


def _prereg_at_head(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "ls-files", "--", PREREG_GLOB], capture_output=True, check=False)
    hits = sorted(ln for ln in r.stdout.decode("utf-8", "replace").split("\n") if ln.strip()) if r.returncode == 0 else []
    if not hits:
        raise Refused("REFUSED: no %s is committed at HEAD" % PREREG_GLOB)
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the leg-2 verdict for one fold")
    ap.add_argument("--tranche", required=True, choices=("A", "B"))
    ap.add_argument("--result", default=str(HERE / "measurement_result.json"))
    ap.add_argument("--root", default=str(HERE.parent.parent.parent))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from styxx.mind import wilson
    from styxx.protocol import Experiment
    result = json.loads(Path(a.result).read_text(encoding="utf-8"))
    rec = run(result, _prereg_at_head(root), root, a.tranche, Experiment, wilson)
    text = json.dumps(rec, indent=1, ensure_ascii=False) + "\n"
    if a.out:
        out = Path(a.out)
        if out.exists():
            raise Refused("REFUSED: %s exists; a verdict is written once" % out)
        out.write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
