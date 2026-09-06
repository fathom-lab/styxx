# -*- coding: utf-8 -*-
"""Every committed receipt that claims `committed` names a commit this repository actually has.

THE GAP THIS CLOSES. A span resolved through the tree channel gets the strongest provenance the
format states -- `rungs: committed`, and the note `committed object at <40-hex>; authorship
unchecked`. The verifier prints that on the word of whatever tree handle the caller passed. Only
`GitTree` consults git; `MemoryTree` answers from a dict the caller filled, and `SnapshotTree` only
checks that the handle's commit equals the commit its own entries were read at -- which a fabricated
snapshot satisfies trivially. The tree channel records no kind of tree, so a reader holding a
receipt cannot tell which of the three answered.

Contrast the manifest channel, which demands `kind_of_source` on every entry, validates it against
an allowlist and refuses `kind_of_source_unknown`. The asymmetry is the finding; see
FINDING_tree_channel_states_no_kind_2026_09_06.md.

The format-level remedy moves the digested core and is therefore not taken here. This is the
corpus-level one: rather than requiring a reader to go out of band -- to the population file, to a
generator's source, to the GitHub API -- to learn whether a receipt's "committed" is backed, the
answer is computed here and the exceptions are declared once, in a committed file, with reasons.

WHAT IT CHECKS. For every tracked receipt with at least one `path:`/`prereg:` span: the commit that
receipt names must be a commit object in this repository. A receipt that names one which is not must
appear in `receipt_provenance_declarations.json` with a kind and a reason, or the audit fails.

WHAT IT DOES NOT CHECK. That the bytes behind a real commit are the bytes the span quoted -- that is
re-derivation's job, and re-derivation is the format's actual defence. This checks only that the
history a receipt appeals to exists. A receipt naming a real commit can still be wrong; this catches
the narrower case of a receipt appealing to history that is not there.

AN ABSENT COMMIT IS NOT AN ACCUSATION UNLESS THE HISTORY IS COMPLETE. In a shallow clone -- which
is what `actions/checkout` gives by default, and therefore what CI has -- nearly every commit is
missing, and an audit that read absence as fabrication would accuse the whole corpus at once. That
failure mode is not hypothetical here: a path-claim accusation shipped at 0.23 precision by making
exactly this inference. So the audit establishes whether it can see history before it says anything,
and reports INDETERMINATE rather than guessing.

  EXIT 0  every tree-claiming receipt is backed by history, or declared with a reason
  EXIT 1  at least one is neither -- an accusation, only ever made against complete history
  EXIT 2  the history here is incomplete, so the question cannot be answered

  python papers/sworn/receipt_provenance_audit.py            # report
  python papers/sworn/receipt_provenance_audit.py --list     # every tree-claiming receipt and its state
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECLARATIONS = HERE / "receipt_provenance_declarations.json"

TREE_FORMS = ("path", "prereg")

EXIT_OK, EXIT_UNDECLARED, EXIT_INDETERMINATE = 0, 1, 2


def _git(*args: str) -> tuple:
    p = subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True)
    return p.returncode, p.stdout


def _is_shallow() -> bool:
    """Can this checkout see its own history? A clone that cannot must not accuse anyone."""
    rc, out = _git("rev-parse", "--is-shallow-repository")
    if rc != 0:
        return True                       # completeness not established: assume we cannot see
    return out.strip() == b"true"


def _tracked_json() -> list:
    rc, out = _git("ls-files", "-z", "*.json")
    if rc != 0:
        raise SystemExit("REFUSED: git ls-files failed; this audit only runs inside the repository")
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def _is_receipt(obj) -> bool:
    return isinstance(obj, dict) and "spans" in obj and "document_verdict" in obj


def _tree_claim(obj: dict) -> int:
    """How many spans in this receipt were resolved through the tree channel."""
    return sum(1 for s in obj.get("spans", [])
               if ((s.get("provenance") or {}).get("form")) in TREE_FORMS)


def _commit_exists(sha) -> bool:
    if not isinstance(sha, str):
        return False
    rc, out = _git("cat-file", "-t", sha)
    return rc == 0 and out.strip() == b"commit"


def _declarations() -> dict:
    if not DECLARATIONS.exists():
        return {}
    d = json.loads(DECLARATIONS.read_text(encoding="utf-8"))
    out = {}
    for e in d.get("declared", []):
        for k in ("path", "commit", "kind", "reason"):
            if not isinstance(e.get(k), str) or not e[k].strip():
                raise SystemExit("REFUSED: a declaration is missing %r: %r" % (k, e))
        out[(e["path"], e["commit"])] = e
    return out


def main(argv) -> int:
    declared = _declarations()
    shallow = _is_shallow()
    listing, undeclared, stale = [], [], []
    backed = 0

    for rel in _tracked_json():
        try:
            obj = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        if not _is_receipt(obj):
            continue
        n = _tree_claim(obj)
        if n == 0:
            continue                                             # names a commit but appeals to none
        sha = obj.get("commit")
        key = (rel.replace("\\", "/"), sha if isinstance(sha, str) else "")
        e = declared.get(key)

        if _commit_exists(sha):
            backed += 1
            listing.append(("backed", rel, sha, n, ""))
            if e is not None:
                stale.append(rel)                                # declared, but no longer needs to be
        elif e is not None:
            listing.append(("declared", rel, sha, n, "%s: %s" % (e["kind"], e["reason"])))
        elif shallow:
            # The commit is not here, but almost nothing is. Absence proves nothing.
            listing.append(("indeterm", rel, sha, n, "history is shallow"))
        else:
            undeclared.append((rel, sha, n))
            listing.append(("UNDECLARED", rel, sha, n, ""))

    indeterminate = sum(1 for row in listing if row[0] == "indeterm")

    if "--list" in argv:
        for state, rel, sha, n, why in sorted(listing):
            print("%-11s %-58s %s  %d span(s) %s" % (state, rel, (sha or "null")[:12], n, why))
        print()

    print("receipts appealing to the tree channel : %d" % len(listing))
    print("  naming a commit this repository has  : %d" % backed)
    print("  naming one it does not, declared     : %d"
          % (len(listing) - backed - len(undeclared) - indeterminate))
    print("  naming one it does not, UNDECLARED   : %d" % len(undeclared))
    if indeterminate:
        print("  INDETERMINATE (shallow history)      : %d" % indeterminate)

    for rel in stale:
        print("\nSTALE DECLARATION: %s now names a commit that exists; remove its declaration" % rel)
    for (rel, sha, n) in undeclared:
        print("\nUNDECLARED: %s" % rel)
        print("  claims `committed` on %d span(s) at %s, which is not a commit in this repository."
              % (n, sha))
        print("  Either the receipt is wrong, or it is a fixture/synthetic sample; in that case say")
        print("  so in %s with a reason." % DECLARATIONS.name)

    if undeclared or stale:
        return EXIT_UNDECLARED
    if indeterminate:
        print("\nINDETERMINATE: this clone is shallow, so %d receipt(s) could not be checked."
              % indeterminate)
        print("An absent commit here is not evidence of a fabricated one. For a real answer, run")
        print("against complete history:  git fetch --unshallow")
        return EXIT_INDETERMINATE
    print("\nevery tree-channel receipt is backed by history or declared.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
