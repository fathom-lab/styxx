# -*- coding: utf-8 -*-
"""The census papers/sworn/NOTE_certifies_correction_2026_10_24.md swears to.

Counts, at the base commit the /v2 step was cut from, where the /v1 certificate sentence stands and
what the `path:` spans of the committed sworn documents rest on; and, at HEAD, that no committed
receipt was edited and that the /v1 conformance set's core vectors did not move. Every number is
read from git objects and from styxx.sworn, never typed.

  python papers/sworn/certifies_census.py            # write certifies_census_2026_10_24.json
  python papers/sworn/certifies_census.py --check    # regenerate in memory; exit 1 on any difference

WHAT IT COUNTS AND HOW
  files_carrying_the_v1_clause_at_base   git grep -l -F "bytes the author did not write" at BASE,
                                         each file put in one class by its path (a file no rule
                                         places refuses the census)
  v1_receipts_*, v0_receipts_at_base     top-level JSON objects whose schema is a verdict-receipt
                                         schema, in tracked files at BASE; "carrying" compares
                                         certifies with styxx.sworn._CERTIFIES_V1 byte for byte
  sidecars, spans, path_spans,           every papers/**/*.sworn.json at BASE; a path: span's
  path_targets                           target is its receipt minus "path:" and minus any #fragment
  path_*_speaker_touched                 git log --format=%an%x00%cn BASE -- <target> names one of
                                         SPEAKERS as author or committer (lowercased). It counts who
                                         touched a file, never what the file holds
  sidecar_receipts(_verified)            the receipt beside each sidecar at BASE, re-derived by
                                         styxx.sworn.verify_receipt at the commit its sidecar names,
                                         under the verifier this checkout runs
  committed_receipts_modified_or_        git diff --name-status --diff-filter=MD BASE HEAD over the
  deleted_since_base                     receipt-bearing paths below; an added file is not a change
                                         to history
  conformance_v1_core_vectors(_          the core vectors of conformance/sworn/ at HEAD, and how many
  unchanged_since_base)                  are byte-identical (same id, same JSON) to BASE's
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

OUT = HERE / "certifies_census_2026_10_24.json"
SCHEMA = "styxx.sworn.certifies-census/v1"
BASE = "2e98debc9d2aa8f4d70f255fae7445b8e9262ef5"
V1_OPENING = "the spans the author bound were checked against bytes the author did not write"
NEEDLE = "bytes the author did not write"
SPEAKERS = ["claude", "claude (for flobi)", "darkflobi", "darkflobi-autopilot", "styxx-coding-agent"]
RECEIPT_PREFIX = "styxx.sworn.verdict-receipt/"
HISTORY_PATHS = ["*.sworn-receipt.json", "papers/sworn/measurement/dryrun/", "*.capsule.html",
                 "papers/sworn/sworn_action_sample.*", "papers/sworn/sworn_action_sample_2026_09_06.*"]
CLASSES = ["sworn_receipt_json", "dryrun_receipt", "capsule", "action_run_json", "action_summary",
           "sworn_result_and_sidecar", "frozen_spec", "verifier_source"]


def _git(*args: str) -> bytes:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("REFUSED: git %s failed: %s" % (" ".join(args[:3]), r.stderr.decode("utf-8", "replace")[:200]))
    return r.stdout


def _show(rev: str, path: str) -> bytes:
    return _git("show", "%s:%s" % (rev, path))


def _class_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if path in ("styxx/sworn.py", "styxx/_data/sworn_verify.js"):
        return "verifier_source"
    if name.endswith(".sworn-receipt.json"):
        return "sworn_receipt_json"
    if path.startswith("papers/sworn/measurement/dryrun/") and name.endswith(".receipt.json"):
        return "dryrun_receipt"
    if name.endswith(".capsule.html"):
        return "capsule"
    if name.startswith("sworn_action_sample") and name.endswith(".run.json"):
        return "action_run_json"
    if name.startswith("sworn_action_sample") and name.endswith(".summary.md"):
        return "action_summary"
    if path.startswith("papers/sworn/RESULT_") and (name.endswith(".md") or name.endswith(".sworn.json")):
        return "sworn_result_and_sidecar"
    if path.startswith("papers/sworn/SPEC_") and name.endswith(".md"):
        return "frozen_spec"
    raise SystemExit("REFUSED: %s carries the /v1 clause and no class rule places it" % path)


def _ls(rev: str, *pathspec: str) -> List[str]:
    out = _git("ls-tree", "-r", "--name-only", "-z", rev, *pathspec).decode("utf-8")
    return sorted(p for p in out.split("\0") if p)


def census(verify_receipts: bool = True) -> Dict:
    # ------------------------------------------------------------ where the sentence stands at BASE
    hits = _git("grep", "-l", "-F", NEEDLE, BASE).decode("utf-8").splitlines()
    files = sorted(h.split(":", 1)[1] for h in hits)
    by_class = {c: 0 for c in CLASSES}
    for f in files:
        by_class[_class_of(f)] += 1

    receipts = {"styxx.sworn.verdict-receipt/v0": 0, "styxx.sworn.verdict-receipt/v1": 0}
    carrying = 0
    rec_files = _git("grep", "-l", "-F", RECEIPT_PREFIX, BASE, "--", "*.json").decode("utf-8").splitlines()
    for h in sorted(rec_files):
        path = h.split(":", 1)[1]
        try:
            obj = json.loads(_show(BASE, path).decode("utf-8"))
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("schema"), str) and obj["schema"].startswith(RECEIPT_PREFIX):
            if obj["schema"] not in receipts:
                raise SystemExit("REFUSED: %s carries receipt schema %r at base" % (path, obj["schema"]))
            receipts[obj["schema"]] += 1
            if obj["schema"].endswith("/v1") and obj.get("certifies") == sworn._CERTIFIES_V1:
                carrying += 1

    # ------------------------------------------------------------ what the path: spans rest on, at BASE
    sidecars = [p for p in _ls(BASE, "papers") if p.endswith(".sworn.json")]
    spans = path_spans = 0
    per_target: Dict[str, int] = {}
    beside: List[tuple] = []
    for side_path in sidecars:
        side = json.loads(_show(BASE, side_path).decode("utf-8"))
        for s in side.get("spans") or []:
            spans += 1
            r = s.get("receipt") or ""
            if r.startswith("path:"):
                path_spans += 1
                target = r[len("path:"):].split("#", 1)[0]
                per_target[target] = per_target.get(target, 0) + 1
        rec_path = side_path[:-len(".sworn.json")] + ".sworn-receipt.json"
        if rec_path in _ls(BASE, rec_path):
            beside.append((side_path, rec_path, side))
    touched = []
    for target in sorted(per_target):
        names = _git("log", "--format=%an%x00%cn", BASE, "--", target).decode("utf-8", "replace")
        who = {n.strip().lower() for line in names.splitlines() for n in line.split("\0")}
        if who & set(SPEAKERS):
            touched.append(target)

    # ------------------------------------------------------------ the receipts beside a sidecar re-derive
    verified = None
    if verify_receipts:
        verified = 0
        for side_path, rec_path, side in beside:
            rec = json.loads(_show(BASE, rec_path).decode("utf-8"))
            obj = sworn.load_sidecar(side)
            res = sworn.verify_receipt(rec, sidecar=obj, tree=sworn.GitTree(ROOT, obj["commit"]))
            verified += 1 if res["status"] == "VERIFIED" else 0

    # ------------------------------------------------------------ at HEAD: history not edited, cores not moved
    changed = [ln for ln in _git("diff", "--name-status", "--diff-filter=MD", BASE, "HEAD", "--",
                                 *HISTORY_PATHS).decode("utf-8").splitlines() if ln.strip()]

    def vectors(rev: str) -> Dict[str, str]:
        idx = json.loads(_show(rev, "conformance/sworn/index.json").decode("utf-8"))
        out = {}
        for meta in idx["families"].values():
            for v in json.loads(_show(rev, "conformance/sworn/" + meta["file"]).decode("utf-8"))["vectors"]:
                if v["expect"]["outcome"] == "core":
                    out[v["id"]] = json.dumps(v, sort_keys=True)
        return out

    base_cores, head_cores = vectors(BASE), vectors("HEAD")
    unchanged = sum(1 for vid, text in base_cores.items() if head_cores.get(vid) == text)

    return {
        "schema": SCHEMA,
        "base": BASE,
        "v1_opening": V1_OPENING,
        "files_carrying_the_v1_clause_at_base": len(files),
        "by_class_at_base": by_class,
        "v1_receipts_at_base": receipts["styxx.sworn.verdict-receipt/v1"],
        "v1_receipts_carrying_the_v1_sentence": carrying,
        "v0_receipts_at_base": receipts["styxx.sworn.verdict-receipt/v0"],
        "sidecars": len(sidecars),
        "spans": spans,
        "path_spans": path_spans,
        "path_targets": len(per_target),
        "speaker_names": SPEAKERS,
        "path_spans_on_speaker_touched_files": sum(per_target[t] for t in touched),
        "path_targets_speaker_touched": len(touched),
        "sidecar_receipts": len(beside),
        "sidecar_receipts_verified": verified,
        "committed_receipts_modified_or_deleted_since_base": len(changed),
        "conformance_v1_core_vectors": len(head_cores),
        "conformance_v1_core_vectors_unchanged_since_base": unchanged,
        "what_this_does_not_count": ("whether any file a path: span rests on is true, or who wrote the bytes "
                                     "of any receipt: a speaker name in a file's history says who touched it, "
                                     "nothing more"),
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    data = census()
    if argv == ["--check"]:
        with tempfile.TemporaryDirectory() as td:
            fresh = sworn._write_json_lf(Path(td) / OUT.name, data).read_bytes()
        if not OUT.exists() or OUT.read_bytes() != fresh:
            sys.stderr.write("the census does not reproduce: %s differs from a fresh run\n" % OUT.name)
            return 1
        print("certifies census reproduces (%s)" % OUT.name)
        return 0
    if argv:
        sys.stderr.write(__doc__)
        return 2
    sworn._write_json_lf(OUT, data)
    print("wrote %s" % OUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
