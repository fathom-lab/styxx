# -*- coding: utf-8 -*-
"""Tranche B of the leg-2 PREREG, selected and built with the committed machinery.

Committed beside the PREREG, in the signing commit, as papers/sworn/measurement/tranche_b.py; its blob
is in the PREREG's lock (section 5), so the code that builds tranche B exists before any seat reads a
tranche-A item. It calls only committed functions:
`population.select`, `population.describe`, `population.build`, `build_packets.build`,
`canaries.build`, `seal_key.new_salt`, `common.write_json_lf`, `common.tracked_at`, and reads
`build_packets.DECOY_PICKS_L`. It runs no seat and loads no model.

The rule (PREREG Amendment 2): the signing point S is the earliest mainline commit whose tree holds
the PREREG ("mainline" = the parent-1 chain from the tip). A document is `<stem>.md` with its
`<stem>.sworn.json`. Tranche B is the four earliest documents that population.py's rule selects and
whose `.md` AND `.sworn.json` are both absent from S's tree, in merge order: a document enters at
the earliest mainline commit whose tree holds both (the sidecar alone when the tip has no `.md`),
ties broken by path. A rule-selected document is excluded and listed, never picked, when a commit in
S's history added its `.md`: at its path (`git log --diff-filter=A`), or at an earlier path that a
commit renamed to it, by git's rename detection at RENAME_SIMILARITY, git's default, on the commit
that added the later path (`git diff-tree -M50% --diff-filter=R`), back through every such rename.
Renames only, never copies: `git log --follow` is not the test, because it turns on copy detection
over the whole parent tree and so follows a new document back to any older file it resembles. That
covers a `.md` in S's tree whose sidecar comes later, a `.md` moved after S, and a `.md` deleted
before S and added again after it. A move git scores below 50% similar is not followed and reads as a
new document; `md_adds` (every add found, with the path it renamed from) and `md_added_by` (the oldest)
are printed for every document the plan picks or excludes, so such a move shows beside the pick.

The close (sign sheet line 7): CLOSE_AT below. Only documents entering at or before the close cut,
the last mainline commit whose committer date is at or before CLOSE_AT, are candidates; later ones
are listed as entered after the close. P_B, tranche B's pin, is the mainline commit where the
fourth candidate enters. With fewer than four, `--close-short` pins P_B at the close cut and labels
the tranche short. It refuses while the clock reads earlier than CLOSE_AT, and it refuses when the
mainline read from the tip has no commit after the close cut: the cut is not known from that tip (a
stale origin/main would pin early), so fetch and re-run. The CLI takes no date and no tip; it reads
origin/main. A selected document that `population.describe` refuses is printed as selected and
refused, with the refusal, and is not replaced.

Decoys: tranche A's ten decoy sources (`population.json`'s `excluded` at f28d35ad), passed to
`population.build` as tranche B's `excluded`, so `build_packets.build` builds the same thirty Panel L
and thirty Panel R decoys with the same seed. The default refuses unless each source's `.md`,
`.sworn.json` and `.sworn-receipt.json` read the same blob (or are absent alike) at f28d35ad and at
P_B. `--decoys-from-rule-at-pin`, the PREREG's one named fallback, runs only after that refusal:
it refuses while the default holds, and it refuses when any moved file belongs to a Panel L pick's
source (`build_packets.DECOY_PICKS_L` fixes a unit index per pick, which a moved source would point
at another sentence); then tranche B has no G-D and is printed as such. Otherwise it passes the
rule's excluded list at P_B.

CLI:
  python papers/sworn/measurement/tranche_b.py plan  --signing-point-of <PREREG path> [--close-short]
  python papers/sworn/measurement/tranche_b.py build --signing-point-of <PREREG path> --sealed <DIR>
         [--close-short] [--decoys-from-rule-at-pin] [--out papers/sworn/measurement/tranche_B]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402

PREFIXES = ("RESULT_", "FINDING_", "DECLARATION_")
TRANCHE_B_SIZE = 4
CLOSE_AT = "2027-01-10T23:59:59-05:00"              # sign sheet line 7; a changed date is a changed blob
SELECTION_SCHEMA = "styxx-sworn/leg2-tranche-B-selection/v3"
RENAME_SIMILARITY = "50%"                            # git's default rename threshold, named in the rule
DECOY_SUFFIXES = (".md", ".sworn.json", ".sworn-receipt.json")


class NoGD(SystemExit):
    """The fallback refused because a Panel L pick's source moved: terminal, written to the selection."""


def _git(root, *args) -> bytes:
    rc, out = C.git(*args, root=root)
    if rc != 0:
        raise SystemExit("REFUSED: git %s failed" % " ".join(args))
    return out


def _epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def mainline(root, tip: str):
    """(the parent-1 chain from `tip`, oldest at index 0; {commit: committer epoch})."""
    parents, when = {}, {}
    for ln in _git(root, "log", "--format=%H %ct %P", tip).decode("ascii").splitlines():
        parts = ln.split()
        when[parts[0]] = int(parts[1])
        parents[parts[0]] = parts[2] if len(parts) > 2 else None
    chain, c = [], tip
    while c is not None:
        chain.append(c)
        c = parents.get(c)
    return list(reversed(chain)), when


def entry_index(root, chain: List[str], path: str) -> Optional[int]:
    """Index of the earliest mainline commit whose tree holds `path` (None if the tip does not)."""
    lo, hi = 0, len(chain) - 1
    if not C.tracked_at(chain[hi], path, root=root):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if C.tracked_at(chain[mid], path, root=root):
            hi = mid
        else:
            lo = mid + 1
    return lo


def rule_selects(stem: str) -> bool:
    """population.py's RULE (l.29-30, l.43-55) for a sidecar already known to be under papers/."""
    return not stem.startswith("papers/sworn/") and stem.rsplit("/", 1)[-1].startswith(PREFIXES)


def _tree(root, commit: str) -> set:
    return {n for n in _git(root, "ls-tree", "-r", "--name-only", commit, "--", "papers").decode("utf-8").split("\n") if n}


def _adds(root, tip: str, path: str) -> List[List[str]]:
    """[commit, committer epoch, author date] for every commit reachable from the tip that added `path`
    (any branch), newest to oldest."""
    rc, out = C.git("log", "--diff-filter=A", "--format=%H %ct %aI", tip, "--", path, root=root)
    if rc != 0:
        raise SystemExit("REFUSED: git log --diff-filter=A failed for %s" % path)
    return [ln.split() for ln in out.decode("ascii", "replace").splitlines() if ln.strip()]


def _renamed_from(root, commit: str, path: str) -> Optional[str]:
    """The path `commit` renamed to `path`, by git's rename detection at RENAME_SIMILARITY against the
    commit's parent (renames only: copy detection stays off), or None."""
    rc, out = C.git("diff-tree", "-r", "--root", "--no-commit-id", "-z", "--name-status", "-M" + RENAME_SIMILARITY,
                    "--diff-filter=R", commit, root=root)
    if rc != 0:
        raise SystemExit("REFUSED: git diff-tree -M%s failed on %s" % (RENAME_SIMILARITY, commit[:12]))
    f = out.decode("utf-8", "replace").split("\x00")
    for i in range(0, len(f) - 2, 3):
        if f[i].startswith("R") and f[i + 2] == path:
            return f[i + 1]
    return None


def _in_history(root, commit: str, s_commit: str) -> bool:
    rc, _ = C.git("merge-base", "--is-ancestor", commit, s_commit, root=root)
    if rc not in (0, 1):
        raise SystemExit("REFUSED: git merge-base --is-ancestor %s %s failed" % (commit[:12], s_commit[:12]))
    return rc == 0


def md_history(root, tip: str, s_commit: str, md_path: str) -> Dict:
    """Every commit that added the `.md` at its path and, back through renames, at each earlier path;
    whether one is in S's history; and the oldest of them (`md_added_by`)."""
    adds, seen, paths, todo = [], set(), {md_path}, [md_path]
    while todo:
        path = todo.pop(0)
        for commit, ct, adate in _adds(root, tip, path):
            if commit in seen:
                continue
            seen.add(commit)
            src = _renamed_from(root, commit, path)
            adds.append({"commit": commit, "author_date": adate, "committer_epoch": int(ct), "path": path,
                         "renamed_from": src, "in_signing_point_history": _in_history(root, commit, s_commit)})
            if src and src not in paths:
                paths.add(src)
                todo.append(src)
    oldest = min(adds, key=lambda r: (r["committer_epoch"], r["commit"])) if adds else None
    return {"md_added_by": {k: oldest[k] for k in ("commit", "author_date", "path")} if oldest else None,
            "md_adds": adds, "md_add_in_signing_point_history": any(r["in_signing_point_history"] for r in adds)}


def plan(root, prereg_path: str, tip: str = "origin/main", close_short: bool = False,
         close_at: str = CLOSE_AT, now: Optional[float] = None) -> Dict:
    tip = _git(root, "rev-parse", tip).decode("ascii").strip()
    chain, when = mainline(root, tip)
    s = entry_index(root, chain, prereg_path)
    if s is None:
        raise SystemExit("REFUSED: %s is not in the tree at %s; there is no signing point" % (prereg_path, tip[:12]))
    close_ts = _epoch(close_at)
    cut = max((i for i, c in enumerate(chain) if when[c] <= close_ts), default=-1)
    if cut < s:
        raise SystemExit("REFUSED: the close %s precedes the signing point %s" % (close_at, chain[s][:12]))
    tree_s = _tree(root, chain[s])
    at_s = {n[: -len(".sworn.json")] for n in tree_s if n.endswith(".sworn.json")}
    at_tip = {n[: -len(".sworn.json")] for n in _tree(root, tip) if n.endswith(".sworn.json")}
    later, in_history = [], []
    for stem in sorted(at_tip - at_s):
        if not rule_selects(stem):
            continue
        i_side = entry_index(root, chain, stem + ".sworn.json")
        i_md = entry_index(root, chain, stem + ".md")
        row = {"stem": stem, "sidecar_entry": chain[i_side] if i_side is not None else None,
               "md_entry": chain[i_md] if i_md is not None else None,
               "md_in_signing_point_tree": (stem + ".md") in tree_s}
        row.update(md_history(root, tip, chain[s], stem + ".md"))
        if row["md_in_signing_point_tree"]:
            row["why"] = "the .md was in the signing point's tree; only its sidecar came after"
            in_history.append(row)
            continue
        if row["md_add_in_signing_point_history"]:
            row["why"] = ("a commit in the signing point's history added this .md, at this path or at an earlier one "
                          "renamed to it (git rename detection at %s): it was written before the signature"
                          % RENAME_SIMILARITY)
            in_history.append(row)
            continue
        if i_side is None or i_side <= s:
            row["why"] = "the sidecar's mainline history is not monotone (present before the signing point, absent at it)"
            in_history.append(row)
            continue
        i = i_side if i_md is None else max(i_side, i_md)
        row["mainline_entry"] = chain[i]
        later.append((i, stem, row))
    later.sort(key=lambda t: (t[0], t[1]))
    candidates = [t for t in later if t[0] <= cut]
    picked = candidates[:TRANCHE_B_SIZE]
    complete = len(picked) == TRANCHE_B_SIZE
    now = time.time() if now is None else now
    if complete:
        p_b = chain[picked[-1][0]]
    elif close_short:
        if now < close_ts:
            raise SystemExit("REFUSED: --close-short before %s; tranche B is still open (%d of %d entered)"
                             % (close_at, len(picked), TRANCHE_B_SIZE))
        if cut == len(chain) - 1:
            raise SystemExit("REFUSED: --close-short: the mainline read from %s has no commit after the close cut, so "
                             "the close cut is not known from this tip; fetch origin/main and re-run" % tip[:12])
        p_b = chain[cut]
    else:
        p_b = None
    return {"schema": SELECTION_SCHEMA, "tip": tip, "prereg": prereg_path, "signing_point": chain[s],
            "close_at": close_at, "close_cut": chain[cut],
            "documents": [r for _, _, r in picked],
            "complete": complete, "closed_short": (not complete) and close_short, "pin_B": p_b,
            "rule_selected_after_signing_point_so_far": len(later),
            "entered_after_the_close": [r["stem"] for i, _, r in later if i > cut],
            "excluded_md_in_signing_point_history": in_history}


def _blob(root, commit: str, path: str):
    return C.git("rev-parse", "--verify", "--quiet", "%s:%s" % (commit, path), root=root)


def _decoy_sources(root, pin_B: str, from_rule: bool) -> Dict:
    import build_packets as B
    import population as P
    pop_A = json.loads((HERE / "population.json").read_text(encoding="utf-8"))
    moved = []
    for e in pop_A["excluded"]:
        for suffix in DECOY_SUFFIXES:
            if _blob(root, pop_A["pinned_commit"], e["stem"] + suffix) != _blob(root, pin_B, e["stem"] + suffix):
                moved.append(e["stem"] + suffix)
    if not moved:
        if from_rule:
            raise SystemExit("REFUSED: --decoys-from-rule-at-pin runs only after the default refuses; tranche A's "
                             "ten decoy sources read the same at %s" % pin_B[:12])
        return {"from": "tranche A's ten decoy sources, unchanged at P_B", "excluded": list(pop_A["excluded"]),
                "moved": []}
    if not from_rule:
        raise SystemExit("REFUSED: tranche A's decoy sources changed between %s and %s: %s; the PREREG's named "
                         "fallback is --decoys-from-rule-at-pin" % (pop_A["pinned_commit"][:12], pin_B[:12], moved))
    l_stems = {stem for stem, _, _ in B.DECOY_PICKS_L}
    l_moved = [m for m in moved if any(m == st + sfx for st in l_stems for sfx in DECOY_SUFFIXES)]
    if l_moved:
        raise NoGD("REFUSED: a Panel L decoy pick's source moved (%s); its authored unit index may name another "
                         "sentence, so the fallback builds nothing: tranche B has no G-D, printed as such" % l_moved)
    _, excluded = P.select(pin_B, root)
    return {"from": "the rule's excluded list at P_B (the named fallback, after the default refused)",
            "excluded": excluded, "moved": moved}


def build(root, prereg_path: str, sealed, out_dir=None, tip: str = "origin/main", close_short: bool = False,
          decoys_from_rule: bool = False, close_at: str = CLOSE_AT, now: Optional[float] = None) -> Dict:
    import build_packets as B
    import canaries as CAN
    import population as P
    import seal_key as K
    root = Path(root)
    out_dir = Path(out_dir or HERE / "tranche_B")
    sealed = Path(sealed)
    if sealed.resolve() == Path(C.SEALED).resolve():
        raise SystemExit("REFUSED: tranche B never writes into tranche A's sealed directory %s" % C.SEALED)
    try:
        sealed.resolve().relative_to(root.resolve())
        raise SystemExit("REFUSED: a sealed directory inside the repository is not sealed")
    except ValueError:
        pass
    if out_dir.resolve() == HERE.resolve():
        raise SystemExit("REFUSED: tranche B never writes into tranche A's measurement directory")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("REFUSED: %s is not empty; a rebuild is a new directory at a new commit" % out_dir)
    sel = plan(root, prereg_path, tip, close_short, close_at, now)
    if sel["pin_B"] is None:
        raise SystemExit("REFUSED: %d of %d tranche-B documents have entered the mainline; nothing is built "
                         "before the fourth, or after %s without --close-short"
                         % (len(sel["documents"]), TRANCHE_B_SIZE, close_at))
    if not sel["documents"]:
        raise SystemExit("REFUSED: no tranche-B document entered the mainline before the close; tranche B has no "
                         "fold (printed)")
    pin_B = sel["pin_B"]
    selected_at_pin, _ = P.select(pin_B, root)
    missing = [d["stem"] for d in sel["documents"] if d["stem"] not in selected_at_pin]
    if missing:
        raise SystemExit("REFUSED: not selected by population.select at P_B: %s" % missing)
    entries, refused = [], []
    for d in sel["documents"]:
        stem = d["stem"]
        try:
            P.describe({"stem": stem, "pinned_commit": pin_B}, root=root)
        except SystemExit as e:
            refused.append({"stem": stem, "refusal": str(e)})
            continue
        role = "design_eight" if C.tracked_at(C.DESIGN_COMMIT, stem + ".sworn.json", root=root) else "prospective"
        entries.append({"stem": stem, "role": role})
    sel["refused_by_population_describe"] = refused
    sel["scorable"] = [e["stem"] for e in entries]
    if not entries:
        out_dir.mkdir(parents=True, exist_ok=True)
        C.write_json_lf(out_dir / "tranche_B_selection.json", sel)
        raise SystemExit("REFUSED: every tranche-B document was refused; tranche B has no fold (printed)")
    try:
        decoys = _decoy_sources(root, pin_B, decoys_from_rule)
    except NoGD as e:      # terminal, so written; the default's refusal writes nothing and the fallback reuses the dir
        sel["decoy_sources"] = {"refused": str(e), "G_D": "none: tranche B has no qualified panel"}
        out_dir.mkdir(parents=True, exist_ok=True)
        C.write_json_lf(out_dir / "tranche_B_selection.json", sel)
        raise
    sel["decoy_sources"] = {"from": decoys["from"], "moved": decoys["moved"],
                            "stems": [e["stem"] for e in decoys["excluded"]]}
    out_dir.mkdir(parents=True, exist_ok=True)
    K.new_salt(sealed)
    pop = P.build(entries, pin_B, seed=C.SEED, prefix="B", excluded=decoys["excluded"], root=root)
    pop_path = C.write_json_lf(out_dir / "population.json", pop)
    pk = B.build(pop, pop_path, root=root, sealed=sealed, out_dir=out_dir, keys_dir=out_dir / "keys", seed=C.SEED)
    cn = CAN.build(pop, root=root, sealed=sealed, keys_dir=out_dir / "keys",
                   digest_path=out_dir / "twins" / "canary_digest.txt", seed=C.SEED)
    sel["built"] = {"documents": [{"doc_id": d["doc_id"], "stem": d["stem"], "units": d["units"],
                                   "sworn_total": d["sworn_total"]} for d in pop["documents"]],
                    "packets": {k: v for k, v in pk.items() if isinstance(v, (int, str))},
                    "canaries_pooled_n": cn["pooled_n"],
                    "canaries_per_twin": {k: v["total"] for k, v in cn["capacity"].items()}}
    C.write_json_lf(out_dir / "tranche_B_selection.json", sel)
    return sel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="leg-2 tranche B: select, then build with the committed machinery")
    sub = ap.add_subparsers(dest="verb", required=True)
    for name in ("plan", "build"):
        s = sub.add_parser(name)
        s.add_argument("--signing-point-of", dest="prereg", required=True)
        s.add_argument("--close-short", action="store_true")
        s.add_argument("--root", default=str(C.ROOT))
        if name == "build":
            s.add_argument("--sealed", required=True)
            s.add_argument("--out", default=None)
            s.add_argument("--decoys-from-rule-at-pin", dest="from_rule", action="store_true")
    a = ap.parse_args(argv)
    if a.verb == "plan":                      # the CLI reads origin/main; `tip` is a Python argument only
        r = plan(a.root, a.prereg, "origin/main", a.close_short)
    else:
        r = build(a.root, a.prereg, a.sealed, a.out, "origin/main", a.close_short, a.from_rule)
    print(json.dumps(r, indent=1, ensure_ascii=False))
    print("no seat was run and no number above is a measurement of sworn output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
