"""EXTERNAL-1 blind adjudication packet — `build [--as-published]`, then `score`.

Prereg: PREREG_external1_aidev_2026_08_31.md, seed 20260831. 100 accusations and
30 decoys (15 gate-VERIFIED, 15 synthetic contradictions made by perturbing a
verified claim's path), shuffled; the adjudicator never sees the gate's verdict.
The key goes to a SEALED file whose salted SHA-256 is committed BEFORE any
adjudication is recorded, and `score` refuses unless the key still hashes to it.
This repair closes the id channel only: the synthetic decoys stay recognisable
by their `zz_` perturbation, and a less conspicuous one needs its own prereg.
RECIPE, defined below build(), has the #125 id history and the regeneration steps.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "external1_ledger.jsonl"
DB = HERE / "external1_shelf.sqlite"
PACKET = HERE / "external1_packet.json"
KEY = HERE / "external1_key_SEALED.json"
DIGEST = HERE / "external1_key_digest.txt"
ANSWERS = HERE / "external1_answers.json"
RESULT = HERE / "external1_adjudication.json"

SEED = 20260831
SALT = "styxx-external1-blind-2026-08-31"
N_ACC, N_VER, N_SYN = 100, 15, 15


def _facts(con, pr_id):
    rows = con.execute("SELECT filename, status FROM f WHERE pr_id=?", (pr_id,)).fetchall()
    seen, out = set(), []
    for fn, st in rows:
        if fn and fn not in seen:
            seen.add(fn)
            out.append({"path": fn, "status": (st or "").lower() or "modified"})
    return out


def build(as_published: bool = False) -> int:
    if refuses_to_overwrite(as_published) or inputs_missing():      # decided before any read
        return 1
    rng = random.Random(SEED)
    acc, ver = [], []
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for i, c in enumerate(r["claims"]):
                item = {"pr_id": r["pr_id"], "agent": r["agent"], "url": r["html_url"],
                        "claim_index": i, "kind": c["kind"], "text": c["text"],
                        "detail": c["detail"]}
                if c["verdict"] == "CONTRADICTED":
                    acc.append(item)
                elif c["verdict"] == "VERIFIED":
                    ver.append(item)
    print(f"population: {len(acc)} accusations, {len(ver)} verified")

    sample_acc = rng.sample(acc, N_ACC)
    sample_ver = rng.sample(ver, N_VER + N_SYN)
    decoy_ver = sample_ver[:N_VER]
    to_perturb = sample_ver[N_VER:]

    # Read-only: the shelf is 5 GB and is one of the recipe's inputs. A read-write connection
    # can leave journal sidecars beside it, and nothing here ever writes to it.
    con = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
    entries = []          # (arm-order position, packet fields without the id, key entry)

    def add(item, truth, note):
        facts = _facts(con, item["pr_id"])
        entries.append((len(entries),
                        {"agent": item["agent"], "url": item["url"],
                         "claim_kind": item["kind"], "claim_text": item["text"],
                         "claim_detail": item["detail"], "changed_files": facts},
                        {"truth": truth, "note": note, "pr_id": item["pr_id"],
                         "claim_index": item["claim_index"]}))

    for it in sample_acc:
        add(it, "gate_says_contradicted", "sampled accusation")
    for it in decoy_ver:
        add(it, "decoy_verified", "gate verified this")
    for it in to_perturb:
        p = dict(it)
        d = dict(p.get("detail") or {})
        path = d.get("path") or ""
        if path:
            parts = path.rsplit("/", 1)
            parts[-1] = "zz_" + parts[-1]
            newp = "/".join(parts)
            p["text"] = p["text"].replace(path, newp)
            d["path"] = newp
            p["detail"] = d
        add(p, "decoy_synthetic_contradiction",
            "verified claim with its path perturbed — must read as contradicted")

    con.close()
    # The shuffle is the published one in both modes: Random.shuffle's draws depend only on the
    # list's length. What differs is where the id comes from. The published packet numbered each
    # item in arm order before this shuffle, so the id carried the arm (#125); the repaired build
    # numbers it from its shuffled position, as compat2_packet.py does.
    rng.shuffle(entries)
    items, key = [], {}
    for pos, (arm_pos, fields, truth) in enumerate(entries):
        iid = f"E1-{(arm_pos if as_published else pos):03d}"
        items.append({"id": iid, **fields})
        key[iid] = truth
    arms_in_id_order = [key[iid]["truth"] for iid in sorted(key)]
    runs, n_arms = arm_runs(arms_in_id_order), len(set(arms_in_id_order))
    if as_published:
        print(f"AS PUBLISHED: ids numbered in arm order ({runs} runs over {n_arms} arms); "
              f"the id carries the arm (#125). This reproduces the record, leak included.")
    elif runs <= n_arms + 2:
        raise AssertionError(f"ids still cluster by arm ({runs} runs over {n_arms} arms); "
                             f"the shuffle did not take")

    packet_text = packet_json(
        {"prereg": "PREREG_external1_aidev_2026_08_31.md", "seed": SEED,
         "n_items": len(items),
         "instructions": ("For each item answer SUPPORTED or CONTRADICTED: do the "
                          "PR's changed files support the claim, or contradict it? "
                          "You are not told the gate's verdict. Answer every item."),
         "items": items})
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    digest_text = f"sha256(salt+key) = {digest}\nsalt = {SALT}\nitems = {len(items)}\n"
    outputs = ((PACKET, packet_text), (KEY, body + "\n"), (DIGEST, digest_text))
    # Last guard, on the bytes themselves: refuses_to_overwrite() above cannot compare contents
    # it has not built yet, so `--as-published` gets its exact-bytes check here. Git stores the
    # receipts with LF and a Windows checkout renders them with CRLF, so that rendering counts
    # as the record; no other difference does.
    clash = [p.name for p, text in outputs
             if p.exists() and p.read_bytes().replace(b"\r\n", b"\n") != text.encode("utf-8")]
    if clash:
        print(f"REFUSED: {', '.join(clash)} already exist with different contents and are "
              f"EXTERNAL-1's receipts. To regenerate the published packet run "
              f"`build --as-published`; a build under the repaired numbering is a new cycle "
              f"with its own prereg and its own paths.")
        return 1
    for p, text in outputs:
        write_lf(p, text)
    print(f"packet: {len(items)} items -> {PACKET.name}")
    print(f"SEALED KEY DIGEST (commit this before adjudicating):\n  {digest}")
    return 0


def packet_json(packet) -> str:
    """The packet's serialisation: one-space indent, non-ASCII written as itself, one final LF.

    The committed external1_packet.json is exactly this rendering of its own parsed contents,
    and it is not ASCII, so `ensure_ascii` is part of the record.
    """
    return json.dumps(packet, indent=1, ensure_ascii=False) + "\n"


def write_lf(path, text) -> None:
    """Write `text` as UTF-8 with LF line endings on every platform.

    LF is what git stores for the committed receipts and what the pre-repair builder wrote on
    Linux. Path.write_text emits the platform's separator (CRLF on Windows), and its `newline`
    argument needs Python 3.10, so this opens the file itself.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def arm_runs(arms_in_id_order) -> int:
    """How many maximal runs of one arm the id order contains (3 = every arm contiguous)."""
    arms = list(arms_in_id_order)
    return sum(1 for i, a in enumerate(arms) if i == 0 or a != arms[i - 1])


def refuses_to_overwrite(as_published: bool) -> bool:
    """Would a plain build rewrite or half-complete EXTERNAL-1's receipts? Decided from the files.

    Called before the ledger and the shelf are opened, so a build that is going to be refused
    reads nothing, opens nothing and leaves nothing behind. `--as-published` is never refused
    here: it is the mode whose job is to write where the receipts are, the fresh-clone state
    included, and the exact-bytes comparison at the end of build() is what holds it.
    """
    if as_published:
        return False
    present = [p.name for p in (PACKET, KEY, DIGEST) if p.exists()]
    absent = [p.name for p in (PACKET, KEY, DIGEST) if not p.exists()]
    if present and absent:
        print(f"REFUSED: {', '.join(present)} present but {', '.join(absent)} missing. That is "
              f"the fresh-clone state — the packet and the digest are committed, the sealed key "
              f"is gitignored — and a key minted under the repaired numbering would be a "
              f"different key under a committed digest. `build --as-published` restores the "
              f"sealed key from this state: it writes only if the packet and the digest it "
              f"builds are the ones on disk, and the digest is the key's salted hash. A build "
              f"under the repaired numbering is a new cycle in its own directory.")
        return True
    if present:
        print(f"REFUSED: {', '.join(present)} already exist and are EXTERNAL-1's receipts; the "
              f"repaired numbering would rewrite them, leaving the recorded answers keyed to "
              f"items nobody was given. To regenerate the published packet run "
              f"`build --as-published`; a build under the repaired numbering is a new cycle "
              f"with its own prereg and its own paths.")
        return True
    return False


def inputs_missing() -> bool:
    """Is either gitignored recipe input absent? Checked before either is opened, so that
    `build --as-published` run from a clone as it ships refuses instead of raising."""
    missing = [p.name for p in (LEDGER, DB) if not p.exists()]
    if missing:
        print(f"REFUSED: {', '.join(missing)} not found beside this file. The ledger and the "
              f"shelf are gitignored inputs; RECIPE (printed for any unrecognised command) says "
              f"which ledger the published packet was drawn from.")
        return True
    return False


RECIPE = """Item ids (issue #125). The packet that was published and adjudicated was built by a
version of this file that numbered each item as it was added (100 sampled accusations, then 15
gate-VERIFIED decoys, then 15 synthetic contradictions) and shuffled the list afterwards. The
shuffle moved the items and renumbered nothing, so the id carries the arm: E1-000..E1-099 are
accusations, E1-100..E1-114 verified decoys, E1-115..E1-129 synthetic contradictions. The last
range can be read off the published packet alone, because those fifteen items show the `zz_` path
perturbation. The protocol's blinding was weaker than it asserted, on the id.

`build` assigns each id from the item's shuffled position, as compat2_packet.py does, and refuses
to write if the arms still cluster in id order. It closes the id channel and nothing else: the
synthetic contradictions are still the fifteen items whose path carries a `zz_` prefix, so an
adjudicator who looks for it can still name that arm. A perturbation that does not announce
itself is a change to the design, and belongs to a new cycle under its own preregistration.

Regenerating the published packet. external1_packet.json, the sealed key and
external1_key_digest.txt are EXTERNAL-1's receipts; the repair does not rebuild them. Put the
gitignored external1_shelf.sqlite in place, and as external1_ledger.jsonl the ledger the packet
was drawn from: the pre-correction ledger whose counts are external1_summary_PREFIX.json (7,029
CONTRADICTED, 16,868 VERIFIED claims). The ledger regenerated for
CORRECTION_external1_cause_2026_08_31.md (665 / 17,887, external1_summary.json) yields a different
sample under any version of this file. Then run `build --as-published`. It draws the same sample
and the same shuffle (Random.shuffle's draws depend only on the list's length) and numbers every
item by its pre-shuffle, arm-order position, which reproduces the published numbering, the key and
the digest in external1_key_digest.txt. It reproduces the leak with them, on purpose: the leak is
part of the record. The published id order follows from the two population counts alone, and
tests/test_external1_packet_ids.py pins it against the committed packet.

This works from the repository as cloned, where the packet and the digest are committed and the
sealed key is gitignored and absent. `build --as-published` writes nothing unless every receipt
already on disk is what it built, byte for byte once a Windows checkout's CRLF is read as LF. From
a clone that means it restores the sealed key only when the packet and the digest it builds are
the committed ones, and the digest is the key's salted SHA-256, so the key it writes is the sealed
key: the digest covers every byte of its body. It writes LF on every platform, the bytes git stores
for the receipts. A plain build writes nothing where any receipt exists, the fresh-clone state
included. A run under the repaired numbering is a new cycle with its own preregistration and its
own paths, never a rewrite of this one.
"""


def score() -> int:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    body = json.dumps(key, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256((SALT + body).encode("utf-8")).hexdigest()
    committed = DIGEST.read_text(encoding="utf-8")
    if digest not in committed:
        print("REFUSED: sealed key does not match the committed digest.")
        return 1
    ans = json.loads(ANSWERS.read_text(encoding="utf-8"))
    dec_ok = dec_n = 0
    tp = fp = 0
    misses = []
    for iid, truth in key.items():
        a = ans.get(iid)
        if a is None:
            print(f"REFUSED: item {iid} unanswered")
            return 1
        t = truth["truth"]
        if t == "decoy_verified":
            dec_n += 1
            dec_ok += (a == "SUPPORTED")
            if a != "SUPPORTED":
                misses.append((iid, t, a))
        elif t == "decoy_synthetic_contradiction":
            dec_n += 1
            dec_ok += (a == "CONTRADICTED")
            if a != "CONTRADICTED":
                misses.append((iid, t, a))
        else:
            if a == "CONTRADICTED":
                tp += 1
            else:
                fp += 1
    reliable = dec_ok >= 27
    precision = tp / (tp + fp) if (tp + fp) else None
    out = {"decoys_correct": dec_ok, "decoys_total": dec_n,
           "adjudicator_reliable": reliable,
           "accusations_scored": tp + fp,
           "accusations_upheld": tp, "accusations_rejected": fp,
           "precision": None if not reliable else round(precision, 4),
           "gate_G_E1_pass": bool(reliable and precision is not None
                                  and precision >= 0.95),
           "decoy_misses": misses}
    RESULT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not reliable:
        print("\nADJUDICATION VOID — fewer than 27/30 decoys correct. "
              "No headline number may be published.")
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args.pop(0) if args else "build"
    if cmd == "build" and args in ([], ["--as-published"]):
        return build(as_published=bool(args))
    if cmd == "score" and not args:
        return score()
    print(__doc__, file=sys.stderr)
    print(RECIPE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
