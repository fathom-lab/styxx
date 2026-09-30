"""PATH-2a recall (D): how many of main's decided claims the overlay withholds (NOTE_path2a_abstain_overlay_2026_09_30,
NOTE_path2a_second_pass_2026_09_30).

    python path2a_recall.py                        # main's committed corpora (py_side.CORPORA) found here
    python path2a_recall.py --corpora DIR          # also look for the gitignored corpora (corpus_real.json ...) in DIR
    python path2a_recall.py EXTRA.json ...         # extra files, reported separately (e.g. #161's path2_pairs.json)
    python path2a_recall.py --truth EXTRA.json     # for files whose cases carry base/head models, the verdicts lost
    python path2a_recall.py --path-flavour posix   # main's Path read as PurePosixPath (or windows), both modules
    python path2a_recall.py --json OUT.json        # the figures as JSON as well

The reference is main itself, reconstructed from this checkout (tests/_p2a_ref.py), run in the same process over the
same bytes as the branch. For each file: pairs, main's decided claims, how many the branch turns UNCHECKABLE, by kind,
phrase and defect, the file's sha256 (LF), and the mean time per gate call for each. Three totals are printed:
main's committed corpora (the figure the README and the CHANGELOG quote), the overlay's own pins
(path2a_pairs.json, built to abstain, so never added to main's total), and main's corpora with the extra files.
main's find_path reads Path(p).name, and a drive-like name reads otherwise under the two path flavours, so the
flavour the figures were taken under is printed with them. Nothing is written but --json.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from py_side import CORPORA  # noqa: E402
from tests import _p2a_ref as R  # noqa: E402

DECIDED = ("VERIFIED", "CONTRADICTED")


def load(path: Path) -> list:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["cases"] if isinstance(d, dict) else d


def measure(items: list, M, N, truth: bool) -> dict:
    c = collections.Counter()
    by = collections.Counter()
    t_main = t_new = 0.0
    T = None
    if truth:
        from tests import _p2a_truth as T
    for it in items:
        c["pairs"] += 1
        try:
            t0 = time.perf_counter()
            a = M.gate_diff_text(it["summary"], it["diff"]).to_dict()
            t1 = time.perf_counter()
            b = N.gate_diff_text(it["summary"], it["diff"]).to_dict()
            t2 = time.perf_counter()
        except Exception:
            c["main raises"] += 1
            continue
        t_main, t_new = t_main + (t1 - t0), t_new + (t2 - t1)
        for x, y in zip(a["claims"], b["claims"]):
            if x["verdict"] not in DECIDED:
                continue
            c["decided"] += 1
            ab = y["verdict"] == "UNCHECKABLE"
            if ab:
                c["abstained"] += 1
                key = R.phrase_key(y["why"], N._P2A_PHRASES)
                defect = y["why"].split("(", 1)[1].split(")", 1)[0]
                by[f"{x['kind']} {x['verdict']} {key} ({defect})"] += 1
                c["defect " + defect] += 1
            if truth and it.get("model"):
                t = T.truth(it["model"], x["kind"], x["detail"])
                what = ("unjudged" if t is None else "undecided" if t == "?" else
                        "false" if T.wrong(x["verdict"], t) else "right")
                c["truth " + what] += 1
                c["truth " + what + " abstained"] += ab
    n = max(1, c["pairs"] - c["main raises"])
    return {"counts": dict(c), "by": dict(sorted(by.items())),
            "ms_per_call": {"main": round(1000 * t_main / n, 3), "branch": round(1000 * t_new / n, 3)}}


def main(argv: list) -> int:
    truth = "--truth" in argv
    out_json = argv[argv.index("--json") + 1] if "--json" in argv else None
    dirs = [HERE] + ([Path(argv[argv.index("--corpora") + 1])] if "--corpora" in argv else [])
    skip = {out_json, argv[argv.index("--corpora") + 1] if "--corpora" in argv else None}
    skip.update(argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--path-flavour")
    extras = [Path(a) for a in argv if not a.startswith("--") and a not in skip]
    M, N = R.main_module(), __import__("styxx.diffgate", fromlist=["x"])
    if "--path-flavour" in argv:
        import pathlib
        flavour = argv[argv.index("--path-flavour") + 1]
        M.Path = N.Path = {"posix": pathlib.PurePosixPath, "windows": pathlib.PureWindowsPath}[flavour]
        skip.add(flavour)
    flavour = "windows" if M.Path("c:x.py").name == "x.py" else "posix"
    report = {"instrument_sha256": R.sha(R.lf(R.INSTRUMENT)), "main_sha256": R.MAIN_PY_SHA, "path_flavour": flavour,
              "files": {}, "own": {}, "extra": {}}
    total = collections.Counter()
    keep = ("pairs", "decided", "abstained", "main raises")
    for name in CORPORA:
        path = next((d / name for d in dirs if (d / name).is_file()), None)
        if path is None:
            print(f"{name:24s} not found (gitignored corpora: build them, or pass --corpora DIR)")
            continue
        r = measure(load(path), M, N, truth)
        r["sha256"] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        report["files"][name] = r
        total.update({k: v for k, v in r["counts"].items() if k in keep})
    report["total"] = dict(total)
    own = HERE / "path2a_pairs.json"
    r = measure(load(own), M, N, truth)
    r["sha256"] = hashlib.sha256(own.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    report["own"][own.name] = r
    with_extra = collections.Counter(total)
    for path in extras:
        r = measure(load(path), M, N, truth)
        r["sha256"] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        report["extra"][str(path)] = r
        with_extra.update({k: v for k, v in r["counts"].items() if k in keep})
    report["total_with_extra"] = dict(with_extra)
    print(f"path flavour: {flavour} (main's Path('c:x.py').name is {M.Path('c:x.py').name!r})")
    for section in ("files", "own", "extra"):
        for name, r in report[section].items():
            c = r["counts"]
            print(f"{Path(name).name:24s} sha256 {r['sha256'][:12]}  pairs {c.get('pairs', 0):5d}  decided "
                  f"{c.get('decided', 0):5d}  abstained {c.get('abstained', 0):4d}  ms/call main "
                  f"{r['ms_per_call']['main']} branch {r['ms_per_call']['branch']}")
            for k, v in r["by"].items():
                print(f"    {v:5d}  {k}")
            for k, v in sorted(c.items()):
                if k.startswith("truth "):
                    print(f"    {k}: {v}")
        if section == "files":
            print(f"{'TOTAL main':24s} pairs {total['pairs']}  decided {total['decided']}  abstained "
                  f"{total['abstained']}  ({100 * total['abstained'] / max(1, total['decided']):.1f}%)  "
                  f"main's committed corpora, the overlay's own pins not included")
        if section == "extra" and extras:
            w = with_extra
            print(f"{'TOTAL main + extra':24s} pairs {w['pairs']}  decided {w['decided']}  abstained {w['abstained']}"
                  f"  ({100 * w['abstained'] / max(1, w['decided']):.1f}%)")
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
