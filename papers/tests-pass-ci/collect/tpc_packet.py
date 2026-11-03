# -*- coding: utf-8 -*-
"""The packet builder (PREREG P3, P7, P8, P9; sections 3, 5, 6): the seeded draw re-derived from the frame, the
per-repository cap, candidates and audits AU1 to AU4, the packet's size cut, items in two locked steps, decoys placed by
the shuffle of P8.5, and the key sealed by a salt.

    py -3.12 tpc_packet.py draw-check --root ROOT
        re-derive the draw from the stored frames and beacon and compare it with the stored draw files
    py -3.12 tpc_packet.py decoys --spec SPEC.json --packet P2 --out DIR [--salts DIR] [--testreceipt DIR]
        [--other-set SET.json ...] [--other-spec SPEC.json ...] [--other-key KEY.json ...] [--other-packet PACKET.json ...]
        check a decoy spec against P9 and seal the set: decoyset_<P>.json and .sha256 to --out, the salt to --salts
        (P2's digest fills its lock row at the signing commit)
    py -3.12 tpc_packet.py p2 --root ROOT --decoys DIR --out DIR [--dry-run]
    py -3.12 tpc_packet.py p1 --testreceipt DIR --snapshot FILE --logs DIR --bodies FILE --decoys DIR --root ROOT --out DIR [--dry-run]
    py -3.12 tpc_packet.py q --root DEVROOT --decoys DIR --out DIR [--exclude-spec SPEC.json ...]

The beacon (section 3.1): the draw and the shuffles of P1 and P2 read Bitcoin block 975000, or the block the late rule
names, read from mempool.space and blockstream.info, which must agree; its header must hash to the id and meet its own
target. Until it exists, `--dry-run` builds under a stated stand-in hash, `STAND_IN_HEX`, marked `stand_in` in every
file it writes; nothing built under it is ever read for real: the official builds, the seat runners and the scorer
refuse it. Packet Q needs no beacon (P8.5: "tests-pass-ci:Q" stands in its place by the PREREG's own rule).

The seal: the official P2 build reads window rows from the store only when the seal guard is open (P21), and its decoy
set must hash to the lock row of the PREREG's section 14; a dry run refuses any row opened on or after 2026-11-02.

Selection (P7, section 9.2), all orders inside M by v = u / p_M, the top-up cut by w = u / P_AGENT:
- capped M and each capped A_g: at most 3 claimers with a CI test result per repository, the lowest v;
- candidates: rows of a capped set whose CI test result is "tests failed"; k, d_W, k_PC and d_P read those of capped M;
- AU1: eligible rows of M neither claimer nor implied-only with "tests failed", the 40 lowest v; AU2: capped claimers in
  M with "tests passed" and a failed or timed-out job or check of any name, 30; AU3: as AU1 with "tests passed", 30,
  step 1 only; AU4: capped claimers in M with "tests passed", 100, step 1 only, never cut (an AU2 row that is also an
  AU4 row is shown once, with both steps);
- the size cut, when candidates and audits exceed 370: AU3 to 0, AU2 to 0, top-up candidates (rows not in M) by
  highest w, AU1 to 0. Candidates in M, AU4 rows and rows on which rule R fires are never cut; if they alone exceed
  370 the packet grows and the build record says so. Reading: "candidates in M" are candidates whose row is in M,
  whichever capped set made them candidates; "top-ups" are the rest (section 5's "not in M").
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tpc_beacon  # noqa: E402
import tpc_decoys  # noqa: E402
import tpc_seal  # noqa: E402
from tpc_collect import capped  # noqa: E402
from tpc_const import (AGENT_ORDER, AU_SIZES, DECOYS_PER_PACKET, P_AGENT, PACKET_MAX,  # noqa: E402
                       PREREG_REPO_PATH, SALT_DIR, WINDOW_START, UTC, iso, parse_utc)
from tpc_draw import membership  # noqa: E402
from tpc_frame import dedupe, kept  # noqa: E402
from tpc_items import (body_from_row, build_item, ci_from_row, ci_from_snapshot,  # noqa: E402
                       ci_from_transcription, date_traces)
from tpc_store import dumps  # noqa: E402

STAND_IN_HEX = tpc_beacon.STAND_IN_HEX
Q_SEED = "tests-pass-ci:Q"  # P8.5: packet Q, built before the beacon exists
EXPLORER_NAMES = tpc_beacon.EXPLORERS
PREREG_FILE = HERE.parent / Path(PREREG_REPO_PATH).name
DEFAULT_OUT = Path("C:/Users/heyzo/clawd/data/direction_a/packets")


class PacketError(ValueError):
    pass


# ---- the beacon ---------------------------------------------------------------------------------------------
def stand_in_beacon() -> dict:
    return {"kind": "stand_in", "beacon_hex": STAND_IN_HEX,
            "rule": "DRY RUN ONLY: a stated stand-in hash, not a block; nothing built under it is ever read for real"}


def beacon_problems(b: Optional[dict]) -> list:
    """Why a beacon record may not draw rows or shuffle P1 and P2 (section 3.1): `tpc_beacon.named_problems`, the
    check the collector's window draw makes too."""
    return tpc_beacon.named_problems(b)


def require_named_beacon(b: Optional[dict]) -> str:
    probs = beacon_problems(b)
    if probs:
        raise PacketError("beacon refused: " + "; ".join(probs))
    return b["beacon_hex"]


# ---- the draw, re-derived from the frame (P3) ----------------------------------------------------------------
def derive_draw(frame_rows: list, beacon_hex: str, day: str) -> list:
    """The drawn rows of one day's frame, as `tpc_collect.stage_draw` writes them, sorted by u."""
    uniq, doubles = dedupe([r for r in frame_rows if kept(r)], key=lambda r: r["pr_id"])
    drawn = []
    for r in uniq:
        if r["pr_id"] is None:
            continue
        m = membership(beacon_hex, int(r["pr_id"]), r["agent"])
        if not m["in_D"]:
            continue
        flags = []
        if r["repo"] is None:
            flags.append("NULL_REPO")
        if r["author"] is None:
            flags.append("NULL_AUTHOR")
        drawn.append({"pr_id": r["pr_id"], "repo": r["repo"], "number": r["number"], "agent": r["agent"],
                      "created_at": r["created_at"], "stars": r["stars"], "author": r["author"],
                      "author_type": r["author_type"], "day": day, "u": m["u"], "v": m["v"],
                      "in_D": True, "in_M": m["in_M"], "flags": flags,
                      "matched": doubles.get(r["pr_id"], [r["agent"]])})
    drawn.sort(key=lambda x: x["u"])
    return drawn


def _jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_bytes().decode("utf-8").splitlines() if ln.strip()]


def frame_days(root: Path) -> list:
    return sorted(p.stem.replace("frame_", "") for p in (Path(root) / "frames").glob("frame_20*.jsonl"))


def draw_check(root: Path, beacon_hex: str) -> dict:
    """Re-derive each day's draw from its frame and compare it with the stored draw file."""
    root = Path(root)
    out = {"days": {}, "differ": []}
    for day in frame_days(root):
        got = derive_draw(_jsonl(root / "frames" / f"frame_{day}.jsonl"), beacon_hex, day)
        stored = _jsonl(root / "draw" / f"draw_{day}.jsonl")
        a = [(r["pr_id"], r["in_M"], r["agent"]) for r in got]
        b = [(r["pr_id"], r["in_M"], r["agent"]) for r in stored]
        out["days"][day] = {"derived": len(a), "stored": len(b), "same": a == b}
        if a != b:
            out["differ"].append(day)
    return out


# ---- P7, the sets ----------------------------------------------------------------------------------------------
def _outcome(r: dict) -> Optional[str]:
    return (r.get("ci") or {}).get("outcome")


def _claim_ci(r: dict) -> bool:
    return bool(r.get("eligible")) and r.get("screen") == "claimer" and bool((r.get("ci") or {}).get("has_ci_test_result"))


def w_of(r: dict) -> float:
    return float(r["u"]) / float(P_AGENT[r["agent"]])


def analysis_sets(rows: list) -> dict:
    """capped M, each capped A_g, their candidates and the audits' populations (section 9.2, P7)."""
    claim_ci = [r for r in rows if _claim_ci(r)]
    m_cap = capped([r for r in claim_ci if r.get("in_M")])
    a_cap = {g: capped([r for r in claim_ci if r["agent"] == g]) for g in AGENT_ORDER}
    elig_m = [r for r in rows if r.get("eligible") and r.get("in_M")]
    by_v = lambda xs: sorted(xs, key=lambda r: (r["v"], r["u"]))  # noqa: E731
    return {
        "claim_ci": claim_ci, "m_cap": m_cap, "a_cap": a_cap,
        "cand_m": [r for r in m_cap if _outcome(r) == "tests failed"],
        "cand_a": {g: [r for r in a_cap[g] if _outcome(r) == "tests failed"] for g in AGENT_ORDER},
        "au1_pop": by_v([r for r in elig_m if r.get("screen") == "none" and _outcome(r) == "tests failed"]),
        "au2_pop": by_v([r for r in m_cap if _outcome(r) == "tests passed"
                         and (r.get("ci") or {}).get("any_failed_job_or_check")]),
        "au3_pop": by_v([r for r in elig_m if r.get("screen") == "none" and _outcome(r) == "tests passed"]),
        "au4_pop": by_v([r for r in m_cap if _outcome(r) == "tests passed"]),
    }


def select_p2(rows: list, budget: int = PACKET_MAX - DECOYS_PER_PACKET) -> dict:
    """The rows of packet P2 with their strata and steps, and what the size cut removed (P7)."""
    s = analysis_sets(rows)
    cands = {}
    for r in s["cand_m"]:
        cands[r["pr_id"]] = r
    for g in AGENT_ORDER:
        for r in s["cand_a"][g]:
            cands.setdefault(r["pr_id"], r)
    cand_m_ids = {r["pr_id"] for r in s["cand_m"]}
    in_m_cands = {k for k, r in cands.items() if r.get("in_M")}
    topups = {k: r for k, r in cands.items() if not r.get("in_M")}
    au = {name: [r["pr_id"] for r in s[name.lower() + "_pop"][:AU_SIZES[name]]] for name in ("AU1", "AU2", "AU3", "AU4")}
    capped_ids = {r["pr_id"] for r in s["m_cap"]} | {r["pr_id"] for g in AGENT_ORDER for r in s["a_cap"][g]}
    by_id = {r["pr_id"]: r for r in rows}
    rule_r = sorted(k for k in capped_ids if (by_id[k].get("rule_r") or {}).get("fires"))
    never = set(in_m_cands) | set(au["AU4"]) | set(rule_r)
    keep_topups = sorted(topups, key=lambda k: w_of(topups[k]))

    def chosen() -> set:
        return never | set(au["AU1"]) | set(au["AU2"]) | set(au["AU3"]) | set(keep_topups)

    cut = {"AU3": [], "AU2": [], "topup": [], "AU1": []}
    for step in ("AU3", "AU2", "topup", "AU1"):
        if len(chosen()) <= budget:
            break
        if step == "topup":
            while keep_topups and len(chosen()) > budget:
                k = keep_topups.pop()  # the highest w
                if k not in never:
                    cut["topup"].append(k)
        else:
            cut[step] = [k for k in au[step] if k not in never]
            au[step] = []
    ids = chosen()
    items = {}
    for k in ids:
        r = by_id[k]
        strata = []
        if k in cand_m_ids:
            strata.append("candidate_M")
        elif k in in_m_cands:
            strata.append("candidate_in_M_agent_set")
        if k in topups:
            strata.append("topup:" + r["agent"])
        for name in ("AU1", "AU2", "AU3", "AU4"):
            if k in au[name]:
                strata.append(name)
        if k in rule_r:
            strata.append("rule_R")
        one_step = set(strata) <= {"AU3", "AU4"}
        items[k] = {"pr_id": k, "strata": strata, "steps": 1 if one_step else 2}
    return {"items": items, "cut": cut, "grew": len(never) > budget, "budget": budget,
            "counts": {"candidates_M": len(cand_m_ids), "candidates_in_M": len(in_m_cands), "topups": len(topups),
                       "rule_R": len(rule_r), "AU1": len(au["AU1"]), "AU2": len(au["AU2"]), "AU3": len(au["AU3"]),
                       "AU4": len(au["AU4"]), "items": len(ids)}}


# ---- P8.5, the shuffle, and the key -----------------------------------------------------------------------------
def shuffle_seed(seed_hex: str, packet: str, salt: bytes) -> int:
    """P8.5: the shuffle's seed, keyed by the packet key's 32-byte salt. The salt is secret until the key is released
    and bound by the key digest committed before any seat reads, so no one holding the public packet (its seed and
    item count are printed in it) can recompute the order, and anyone holding the released key can."""
    if len(salt) != 32:
        raise PacketError("the packet key's salt is 32 bytes (P9)")
    return int(hashlib.sha256((seed_hex + ":" + packet + ":" + salt.hex()).encode("utf-8")).hexdigest(), 16)


def assemble(packet: str, seed_hex: str, reals: list, decoys: list, *, beacon: dict, salt: Optional[bytes] = None,
             extra: Optional[dict] = None) -> dict:
    """Order real items and decoys (P8.5), number them after the shuffle, and seal the key (P9).

    reals: [{"key": {...}, "item": build_item(...)}], in a deterministic order; decoys: the sealed set's entries.
    All entries, real and decoy together, are put in the order of their item sha256, so the starting order says
    nothing about which are decoys; then shuffled under `shuffle_seed`, keyed by the key's salt. A decoy whose bytes
    equal a real item's is refused (the key could not tell them apart).
    Returns {"public": the packet the seats get, "key": the key object, "key_bytes", "salt", "key_digest"}."""
    salt = salt if salt is not None else tpc_decoys.new_salt()
    entries = ([("real", r, r["item"]["item_sha256"]) for r in reals]
               + [("decoy", d, d["item_sha256"]) for d in decoys])
    shared = {r["item"]["item_sha256"] for r in reals} & {d["item_sha256"] for d in decoys}
    if shared:
        raise PacketError(f"{len(shared)} decoys have the bytes of a real item of {packet}")
    entries.sort(key=lambda e: e[2])
    rng = random.Random(shuffle_seed(seed_hex, packet, salt))
    rng.shuffle(entries)
    public_items, key_items = [], {}
    for i, (kind, e, _) in enumerate(entries, 1):
        pid = f"{packet}-{i:03d}"
        if kind == "real":
            it = e["item"]
            public_items.append({"id": pid, "steps": it["steps"], "step1_text": it["step1_text"],
                                 "item_text": it["item_text"]})
            key_items[pid] = dict(e["key"], decoy=False, steps=it["steps"], item_sha256=it["item_sha256"])
        else:
            public_items.append({"id": pid, "steps": 2, "step1_text": e["step1_text"], "item_text": e["item_text"]})
            key_items[pid] = {"decoy": True, "steps": 2, "item_sha256": e["item_sha256"], "side": e["side"],
                              "construction": e["construction"], "label": e["label"], "scope": e["scope"],
                              "evidence": e["evidence"], "preexisting": e["preexisting"],
                              "names_only": e.get("names_only", False), "source_kind": e.get("source_kind")}
    from tpc_seats import PROMPT_SHA256  # the prompts the seats read (P12; section 14)
    public = {"packet": packet, "beacon": {k: beacon.get(k) for k in ("kind", "height", "beacon_hex", "label")
                                           if beacon.get(k) is not None},
              "seed": seed_hex, "prompt_sha256": dict(PROMPT_SHA256), "items": public_items}
    if beacon.get("kind") == "stand_in":
        public["DRY_RUN"] = "built under the stand-in hash; never read for real"
    key = {"packet": packet, "seed": seed_hex, "beacon_kind": beacon.get("kind"), "items": key_items}
    if extra:
        key.update(extra)
    key_bytes = tpc_decoys.canonical(key)
    return {"public": public, "key": key, "key_bytes": key_bytes, "salt": salt,
            "key_digest": tpc_decoys.salted_digest(salt, key_bytes)}


# ---- where secrets may be written (P9; finding 10 of 2026-11-02) -------------------------------------------------
def git_working_tree(path: Path) -> Optional[Path]:
    """The top of the git working tree holding `path` (it need not exist yet), or None. Git runs without the
    caller's GIT_* variables (a stray GIT_DIR would answer for another tree). A path under a folder holding a .git
    entry is inside a working tree whatever git answers: when git cannot confirm the tree there (dubious ownership,
    a broken .git), the path is refused, never read as outside (the review of 2026-11-02)."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    marked = next((q for q in [p] + list(p.parents) if (q / ".git").exists()), None)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    try:
        r = subprocess.run(["git", "-C", str(p), "rev-parse", "--show-toplevel"], capture_output=True, timeout=60,
                           env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise PacketError(f"cannot tell whether {path} is inside a git working tree: {type(e).__name__}") from e
    out = r.stdout.decode("utf-8", "replace").strip()
    if r.returncode == 0 and out:
        return Path(out)
    if marked is not None:
        raise PacketError(f"{path} lies under {marked}, which holds .git, and git cannot confirm that working tree "
                          f"(exit {r.returncode}); nothing secret is written there")
    return None


def private_dir_problem(path: Path, *, salt: bool) -> Optional[str]:
    """Why secrets may not be written under `path`. A salt lives outside any git working tree (P9); a decoy set, a key
    or a packet may sit inside one only where that working tree ignores it (`git check-ignore`), so no `git add -A`
    there takes it."""
    top = git_working_tree(path)
    if top is None:
        return None
    if salt:
        return f"{path} is inside the git working tree {top}; a salt lives outside any (P9)"
    probe = Path(path).resolve() / "probe"
    r = subprocess.run(["git", "-C", str(top), "check-ignore", "-q", str(probe)], capture_output=True, timeout=60)
    if r.returncode == 0:
        return None
    return f"{path} is inside the git working tree {top} and not ignored there; add it to that tree's exclude file"


def _require_private(out: Path, salt_dir: Path, *, dry_run: bool = False) -> None:
    """Refuse unless `out` is outside any git working tree or ignored there, and `salt_dir` is outside any (a dry
    run's stand-in salt, never read for real, may sit where its packet sits)."""
    for path, salt in ((out, False), (salt_dir, not dry_run)):
        why = private_dir_problem(path, salt=salt)
        if why:
            raise PacketError(why)


def write_packet(out: Path, built: dict, build_record: dict, *, salt_dir: Path, dry_run: bool = False) -> dict:
    """Write packet_<P>.json (the seats'), key_<P>.json, its digest and build_<P>.json to `out`, and the key's salt to
    `salt_dir`, outside any git working tree (P9). Never overwrites."""
    out, salt_dir = Path(out), Path(salt_dir)
    _require_private(out, salt_dir, dry_run=dry_run)
    p = built["public"]["packet"]
    files = {out / f"packet_{p}.json": dumps(built["public"], indent=1), out / f"key_{p}.json": built["key_bytes"],
             salt_dir / f"key_{p}.salt": (built["salt"].hex() + "\n").encode("utf-8"),
             out / f"key_{p}.sha256": (built["key_digest"] + "\n").encode("utf-8"),
             out / f"build_{p}.json": dumps(build_record, indent=1)}
    for path in files:
        if path.exists():
            raise PacketError(f"{path} exists; a packet is never rebuilt over itself")
    out.mkdir(parents=True, exist_ok=True)
    salt_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for path, data in files.items():
        path.write_bytes(data)
        if path.read_bytes() != data:
            raise OSError(f"short write: {path}")
        digests[path.name] = hashlib.sha256(data).hexdigest()
    (out / f"packet_{p}.sha256").write_bytes((digests[f"packet_{p}.json"] + "\n").encode("utf-8"))
    return digests


def _read_salt(name: str, salt_dir: Optional[Path], beside: Path) -> bytes:
    """A salt from the salt folder, or beside its file once released (the release publishes it there)."""
    for d in ([Path(salt_dir)] if salt_dir is not None else []) + [Path(beside)]:
        if (d / name).exists():
            return bytes.fromhex((d / name).read_bytes().decode("utf-8").strip())
    raise PacketError(f"no {name} in the salt folder {salt_dir} or beside its file in {beside}")


def load_decoy_set(d: Path, packet: str, *, salt_dir: Optional[Path], lock_digest: Optional[str] = None) -> list:
    """The sealed decoy set of a packet, refused unless it hashes to its digest (and to the lock row, for P2). Its
    salt is read from `salt_dir`, or beside the set once released."""
    d = Path(d)
    data = (d / f"decoyset_{packet}.json").read_bytes()
    salt = _read_salt(f"decoyset_{packet}.salt", salt_dir, d)
    digest = (d / f"decoyset_{packet}.sha256").read_bytes().decode("utf-8").strip()
    if lock_digest is not None and digest != lock_digest:
        raise PacketError(f"the {packet} decoy set's digest {digest[:12]}... is not the lock row's {lock_digest[:12]}...")
    s = tpc_decoys.open_set(data, salt, digest)
    if s.get("packet") != packet or len(s.get("decoys") or []) != DECOYS_PER_PACKET:
        raise PacketError(f"the decoy set is not {packet}'s {DECOYS_PER_PACKET}")
    return s["decoys"]


def p2_lock_digest(prereg_text: str) -> Optional[str]:
    """The sha256 the lock table's P2 decoy set row holds, or None unless one such row holds exactly one (the seal
    guard's own reading, `tpc_seal.p2_set_digest`)."""
    return tpc_seal.p2_set_digest(prereg_text)


# ---- packets ---------------------------------------------------------------------------------------------------
def _read_rows(root: Path) -> list:
    out = []
    for p in sorted((Path(root) / "a2" / "rows").glob("*.json")):
        out.append(json.loads(p.read_bytes().decode("utf-8")))
    return out


def build_p2(root: Path, decoys: list, *, beacon: dict, dry_run: bool, guard=None,
             salt: Optional[bytes] = None) -> tuple:
    """(assembled packet, build record) for P2 from a collector store."""
    root = Path(root)
    rows = _read_rows(root)
    if dry_run:
        if beacon.get("kind") != "stand_in":
            raise PacketError("a dry run builds under the stand-in hash only")
        late = [r["pr_id"] for r in rows if parse_utc(r["created_at"]) >= WINDOW_START]
        if late:
            raise PacketError(f"a dry run reads pull requests opened before 2026-11-02 only; {len(late)} rows are not")
        bh = STAND_IN_HEX
        drawn = {}
        for day in frame_days(root):
            for d in derive_draw(_jsonl(root / "frames" / f"frame_{day}.jsonl"), bh, day):
                drawn[d["pr_id"]] = d
        use = []
        for r in rows:
            d = drawn.get(r["pr_id"])
            if d is not None:
                use.append(dict(r, u=d["u"], v=d["v"], in_D=True, in_M=d["in_M"]))
        missing = len(drawn) - len(use)
    else:
        bh = require_named_beacon(beacon)
        if guard is None or not guard.open:
            raise PacketError("SEAL_GUARD: the official P2 build reads window rows only when the seal is open (P21)")
        chk = draw_check(root, bh)
        if chk["differ"]:
            raise PacketError(f"the stored draw differs from the frame's under the beacon on {chk['differ']}")
        for r in rows:
            m = membership(bh, int(r["pr_id"]), r["agent"])
            if (m["in_D"], m["in_M"]) != (True, bool(r["in_M"])):
                raise PacketError(f"row {r['pr_id']} is not drawn as its record says")
        use = rows
        missing = None
    sel = select_p2(use)
    by_id = {r["pr_id"]: r for r in use}
    reals = []
    for k in sorted(sel["items"]):
        r = by_id[k]
        meta = sel["items"][k]
        body = body_from_row(root, r)
        ci = None if meta["steps"] == 1 else ci_from_row(root, r)
        it = build_item(body, ci, r.get("repo"), steps=meta["steps"])
        reals.append({"key": {"pr_id": k, "repo": r["repo"], "number": r["number"], "agent": r["agent"],
                              "strata": meta["strata"], "in_M": bool(r.get("in_M"))}, "item": it, "build": it["build"]})
    built = assemble("P2", bh, reals, decoys, beacon=beacon, salt=salt)
    seal = getattr(guard, "state", None) or {}
    # P9: the decoys' date check (`date_traces`) counted over P2's real items, where it refuses nothing; the RESULT
    # prints these counts beside the decoys' (zero by construction)
    kinds, with_trace = Counter(), 0
    for r in reals:
        t = date_traces(r["item"]["step1_text"] or "") + date_traces(r["item"]["item_text"] or "")
        kinds.update(t)
        with_trace += int(bool(t))
    record = {"packet": "P2", "dry_run": dry_run, "beacon": built["public"]["beacon"], "selection": sel["counts"],
              "prereg_blob_sha256": seal.get("prereg_blob_sha256"),
              "p2_decoy_set_sealed": seal.get("p2_decoy_set_sealed"), "cut": sel["cut"], "grew": sel["grew"], "rows_read": len(rows), "drawn_without_record": missing,
              "items": len(built["public"]["items"]), "decoys": len(decoys),
              "date_traces_real": {"items": len(reals), "items_with_trace": with_trace, "by_kind": dict(kinds)},
              "item_builds": {r["key"]["pr_id"]: r["build"] for r in reals}, "built_at": iso(dt.datetime.now(UTC))}
    return built, record


# ---- P1: testreceipt's items (P8.4, P19) -----------------------------------------------------------------------
SHEET_HEAD = re.compile(r"^## (\d+)\. (.+)$", re.M)


def parse_sheet(text: str, year: str) -> list:
    """testreceipt's sheet-m2.md (2025) or sheet-fresh.md (2026): one dict per judged item."""
    out = []
    parts = SHEET_HEAD.split(text)
    for i in range(1, len(parts), 3):
        n, head, rest = int(parts[i]), parts[i + 1], parts[i + 2]
        url = re.search(r"https://github\.com/[^\s)]+/pull/\d+", rest)
        claim = re.search(r"claim=([a-z ]+)", head)
        sha = re.search(r"head commit: ([0-9a-f]{40})", rest)
        failing = re.search(r"^- all failing checks: (.*)$", rest, re.M)
        agent = head.split(",")[0].split("—")[-1].strip()
        out.append({"year": year, "n": n, "url": url.group(0) if url else None, "agent": agent,
                    "claim": claim.group(1).strip() if claim else None, "head_sha": sha.group(1) if sha else None,
                    "failing": failing.group(1) if failing else None})
    return out


def build_p1(tr_dir: Path, snapshot: dict, logs_dir: Optional[Path], bodies: dict, decoys: list, *, beacon: dict,
             dry_run: bool, salt: Optional[bytes] = None) -> tuple:
    """(assembled packet, build record) for P1: testreceipt's 95 + 43 items and P1's decoys. bodies: {url: body},
    AIDev's at c63c8a5 for 2025 and GitHub's at build time for 2026 (fetch times in the build record)."""
    if dry_run:
        if beacon.get("kind") != "stand_in":
            raise PacketError("a dry run builds under the stand-in hash only")
        bh = STAND_IN_HEX
    else:
        bh = require_named_beacon(beacon)
    tr_dir = Path(tr_dir)
    items = (parse_sheet((tr_dir / "evals/results/sheet-m2.md").read_text(encoding="utf-8"), "2025")
             + parse_sheet((tr_dir / "evals/results/sheet-fresh.md").read_text(encoding="utf-8"), "2026"))
    snap = {(it["year"], it["n"]): it for it in snapshot.get("items") or []}
    reals, builds, missing = [], {}, []
    for it in items:
        s = snap.get((it["year"], it["n"]))
        if s is not None and s.get("url") != it["url"]:
            raise PacketError(f"snapshot item {it['year']}#{it['n']} is {s.get('url')}, the sheet's is {it['url']}")
        if s is not None and (s.get("check_runs") or s.get("statuses")):
            ci = ci_from_snapshot(s, logs_dir)
        else:
            ci = ci_from_transcription(it["failing"] or "")
        body = bodies.get(it["url"])
        if body is None:
            missing.append(it["url"])
        repo = re.sub(r"^https://github\.com/([^/]+/[^/]+)/pull/\d+$", r"\1", it["url"] or "")
        built = build_item(body or "", ci, repo)
        reals.append({"key": {"year": it["year"], "n": it["n"], "url": it["url"], "agent": it["agent"],
                              "claim": it["claim"]}, "item": built})
        builds[f"{it['year']}#{it['n']}"] = dict(built["build"], names_only=bool(ci.get("names_only")))
    if missing and not dry_run:
        raise PacketError(f"{len(missing)} items have no description (OD-12: AIDev's bodies at c63c8a5)")
    built = assemble("P1", bh, reals, decoys, beacon=beacon, salt=salt)
    record = {"packet": "P1", "dry_run": dry_run, "beacon": built["public"]["beacon"], "items": len(reals),
              "decoys": len(decoys), "without_description": missing, "item_builds": builds,
              "built_at": iso(dt.datetime.now(UTC))}
    return built, record


def fetch_bodies(gh, urls: list, guard, batch: int = 20) -> dict:
    """P8.4: the 2026 items' descriptions read from GitHub when P1 is built, each with its fetch time. Every pull
    request is asked of the seal guard on the createdAt GitHub returns (testreceipt's items opened before 2026-11-02)."""
    out = {}
    refs = []
    for u in urls:
        m = re.match(r"^https://github\.com/([^/]+)/([^/]+)/pull/(\d+)$", u or "")
        if not m:
            raise PacketError(f"not a pull request URL: {u!r}")
        refs.append((u, m.group(1), m.group(2), int(m.group(3))))
    for i in range(0, len(refs), batch):
        part = refs[i:i + batch]
        q = "query {\n" + "\n".join(
            f"  p{j}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) "
            f"{{ pullRequest(number: {num}) {{ body createdAt }} }}" for j, (_, o, n, num) in enumerate(part)) + "\n}"
        payload = gh.graphql(q, {}, kind="graphql", what="p1_bodies")
        data = payload.get("data") or {}
        at = iso(dt.datetime.now(UTC))
        for j, (u, _, _, _) in enumerate(part):
            node = ((data.get(f"p{j}") or {}).get("pullRequest")) if data.get(f"p{j}") else None
            if node is None:
                continue
            guard.check(node.get("createdAt"), f"P1 body read of {u}")
            out[u] = {"body": node.get("body") or "", "created_at": node.get("createdAt"), "fetched_at": at}
    return out


def aidev_bodies(parquet: Path, urls: Iterable[str], url_col: str = "html_url", body_col: str = "body") -> dict:
    """P8.4: the 2025 items' descriptions from AIDev's pull_request.parquet at c63c8a5 (OD-12; its sha256 fills the
    lock row at download). Reads with pyarrow, imported only here."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on the box
        raise PacketError("reading AIDev's parquet needs pyarrow") from e
    want = set(urls)
    table = pq.read_table(str(parquet), columns=[url_col, body_col])
    out = {}
    for u, b in zip(table.column(url_col).to_pylist(), table.column(body_col).to_pylist()):
        if u in want:
            out[u] = b or ""
    return out


# ---- Q: 30 pre-window items (P8.5, P10.3) -------------------------------------------------------------------------
def q_order(pr_id) -> str:
    return hashlib.sha256((Q_SEED + ":" + str(pr_id)).encode("utf-8")).hexdigest()


def _ci_shows_failure(r: dict) -> bool:
    ci = r.get("ci") or {}
    return ci.get("outcome") == "tests failed" or bool(ci.get("any_failed_job_or_check"))


def build_q(root: Path, decoys: list, n: int = 30, salt: Optional[bytes] = None, exclude: Iterable[str] = ()) -> tuple:
    """Q's 30 real items (P10.3): eligible claimers with a CI test result opened before 2026-11-02 whose CI shows a
    failure (tests failed, or a failed job or check), so Q's decoys sit among items like them (the review of
    2026-11-02: 28 of 30 decoys showed a failure, 8 of 30 random claimers did), none of them a pull request another
    packet's decoys come from (`exclude`, P9), the 30 lowest by sha256("tests-pass-ci:Q:" + pr_id)."""
    ex = {tpc_decoys.norm_pr(x) for x in exclude}
    rows = [r for r in _read_rows(root) if _claim_ci(r) and _ci_shows_failure(r)
            and f"{(r.get('repo') or '').lower()}#{int(r['number'])}" not in ex]
    late = [r["pr_id"] for r in rows if parse_utc(r["created_at"]) >= WINDOW_START]
    if late:
        raise PacketError(f"packet Q takes pull requests opened before 2026-11-02 only; {len(late)} are not")
    pick = sorted(rows, key=lambda r: q_order(r["pr_id"]))[:n]
    if len(pick) < n:
        raise PacketError(f"packet Q needs {n} pre-window claimers with a CI test result and a failure; the store "
                          f"has {len(pick)}")
    reals = []
    for r in sorted(pick, key=lambda r: r["pr_id"]):
        it = build_item(body_from_row(root, r), ci_from_row(root, r), r.get("repo"))
        reals.append({"key": {"pr_id": r["pr_id"], "repo": r["repo"], "number": r["number"], "agent": r["agent"]},
                      "item": it})
    built = assemble("Q", Q_SEED, reals, decoys, beacon={"kind": "q_rule", "label": Q_SEED}, salt=salt)
    record = {"packet": "Q", "items": len(reals), "decoys": len(decoys), "built_at": iso(dt.datetime.now(UTC)),
              "excluded_sources": len(ex), "item_builds": {r["key"]["pr_id"]: r["item"]["build"] for r in reals}}
    return built, record


# ---- CLI -----------------------------------------------------------------------------------------------------
def testreceipt_urls(tr_dir: Optional[Path]) -> list:
    if not tr_dir:
        return []
    out = []
    for name, year in (("sheet-m2.md", "2025"), ("sheet-fresh.md", "2026")):
        p = Path(tr_dir) / "evals" / "results" / name
        if p.exists():
            out += [it["url"] for it in parse_sheet(p.read_text(encoding="utf-8"), year) if it["url"]]
    return out


def _write_set(out: Path, packet: str, sealed: dict, problems: list, *, salt_dir: Path) -> None:
    """decoyset_<P>.json and .sha256 to `out` (outside any git working tree, or ignored there), the salt to `salt_dir`
    (outside any git working tree; P9). Never reseals over a set."""
    out, salt_dir = Path(out), Path(salt_dir)
    _require_private(out, salt_dir)
    files = {out / f"decoyset_{packet}.json": sealed["set_bytes"],
             salt_dir / f"decoyset_{packet}.salt": (sealed["salt"].hex() + "\n").encode("utf-8"),
             out / f"decoyset_{packet}.sha256": (sealed["digest"] + "\n").encode("utf-8")}
    for path in files:
        if path.exists():
            raise PacketError(f"{path} exists; a decoy set is never resealed over itself")
    out.mkdir(parents=True, exist_ok=True)
    salt_dir.mkdir(parents=True, exist_ok=True)
    for path, data in files.items():
        path.write_bytes(data)
        if path.read_bytes() != data:
            raise OSError(f"short write: {path}")


def _json(p) -> dict:
    return json.loads(Path(p).read_bytes().decode("utf-8"))


def other_packets(sets=(), specs=(), keys=(), packets=()) -> dict:
    """What another packet uses (P9), for `tpc_decoys.check_set`: its sealed sets' item sha256s and descriptions,
    its decoy specs' source pull requests, its keys' real pull requests and its public packets' descriptions."""
    out = {"other_sets": [], "other_sources": [], "other_texts": []}
    for p in sets:
        for d in _json(p)["decoys"]:
            out["other_sets"].append(d["item_sha256"])
            out["other_texts"].append(d.get("step1_text"))
    for p in specs:
        out["other_sources"] += [(e.get("source") or {}).get("pr") for e in _json(p)["decoys"]
                                 if (e.get("source") or {}).get("pr")]
    for p in keys:
        for it in (_json(p).get("items") or {}).values():
            if it.get("decoy"):
                continue
            if it.get("url"):
                out["other_sources"].append(it["url"])
            elif it.get("repo") and it.get("number") is not None:
                out["other_sources"].append(f"{it['repo']}#{it['number']}")
    for p in packets:
        out["other_texts"] += [it.get("step1_text") for it in _json(p).get("items") or []]
    return out


def cmd_decoys(a) -> int:
    spec = _json(a.spec)
    spec["packet"] = a.packet
    others = other_packets(a.other_set or [], a.other_spec or [], a.other_key or [], a.other_packet or [])
    set_obj, probs = tpc_decoys.build_set(spec, excluded=testreceipt_urls(a.testreceipt), **others)
    if probs:
        print(json.dumps({"refused": probs}, indent=1))
        return 1
    sealed = tpc_decoys.seal_set(set_obj)
    _write_set(Path(a.out), a.packet, sealed, probs, salt_dir=_salt_dir(a))
    print(json.dumps({"packet": a.packet, "decoys": len(set_obj["decoys"]), "digest": sealed["digest"]}, indent=1))
    return 0


def cmd_draw_check(a) -> int:
    b = json.loads((Path(a.root) / "draw" / "beacon.json").read_bytes().decode("utf-8"))
    out = draw_check(Path(a.root), b["beacon_hex"])
    print(json.dumps(out, indent=1))
    return 0 if not out["differ"] else 1


def _official_out(a, packet: str) -> Path:
    """Where a packet is written: --out, or the packets folder for an official build. A dry run names its own folder
    and never writes into the packets folder."""
    if a.dry_run:
        if not a.out:
            raise PacketError("a dry run names its own --out folder")
        out = Path(a.out).resolve()
        base = DEFAULT_OUT.resolve()
        if out == base or base in out.parents:
            raise PacketError("a dry run never writes into the packets folder")
        return out
    return Path(a.out) if a.out else DEFAULT_OUT / packet


def _salt_dir(a) -> Path:
    """Where salts are written (P9): --salts, or SALT_DIR, outside any git working tree. A dry run names its own
    folder and never writes into SALT_DIR, where a stand-in salt would stand in the official build's way."""
    if getattr(a, "dry_run", False):
        if not a.salts:
            raise PacketError("a dry run names its own --salts folder")
        s = Path(a.salts).resolve()
        base = SALT_DIR.resolve()
        if s == base or base in s.parents:
            raise PacketError("a dry run never writes into the salts folder")
        return s
    return Path(a.salts) if a.salts else SALT_DIR


def cmd_p2(a) -> int:
    root = Path(a.root)
    if a.dry_run:
        beacon, lock = stand_in_beacon(), None
        guard = None
    else:
        beacon = json.loads((root / "draw" / "beacon.json").read_bytes().decode("utf-8"))
        guard = tpc_seal.SealGuard()
        try:
            # the lock row the guard read at the signing commit, and only if this checkout's PREREG is those bytes
            prereg = guard.signed_prereg(PREREG_FILE)
        except tpc_seal.SealRefused as e:
            raise PacketError(str(e)) from e
        lock = p2_lock_digest(prereg)
        if lock is None:
            raise PacketError("the signed PREREG's P2 decoy set row holds no single sha256")
    salts = _salt_dir(a)
    decoys = load_decoy_set(Path(a.decoys), "P2", salt_dir=salts, lock_digest=lock)
    built, record = build_p2(root, decoys, beacon=beacon, dry_run=a.dry_run, guard=guard)
    out = _official_out(a, "P2")
    digests = write_packet(out, built, record, salt_dir=salts, dry_run=a.dry_run)
    print(json.dumps({"packet": "P2", "dry_run": a.dry_run, "items": record["items"], "key_digest": built["key_digest"],
                      "files": digests, "selection": record["selection"]}, indent=1))
    return 0


def cmd_p1(a) -> int:
    if a.dry_run:
        beacon = stand_in_beacon()
    else:
        if not a.root:
            raise PacketError("the official P1 build reads the beacon from the window store's draw/beacon.json (--root)")
        beacon = json.loads((Path(a.root) / "draw" / "beacon.json").read_bytes().decode("utf-8"))
    snapshot = json.loads(Path(a.snapshot).read_bytes().decode("utf-8"))
    bodies = json.loads(Path(a.bodies).read_bytes().decode("utf-8")) if a.bodies else {}
    salts = _salt_dir(a)
    decoys = load_decoy_set(Path(a.decoys), "P1", salt_dir=salts)
    built, record = build_p1(Path(a.testreceipt), snapshot, Path(a.logs) if a.logs else None, bodies, decoys,
                             beacon=beacon, dry_run=a.dry_run)
    digests = write_packet(_official_out(a, "P1"), built, record, salt_dir=salts, dry_run=a.dry_run)
    print(json.dumps({"packet": "P1", "dry_run": a.dry_run, "items": len(built["public"]["items"]),
                      "key_digest": built["key_digest"], "files": digests}, indent=1))
    return 0


def cmd_q(a) -> int:
    salts = _salt_dir(a)
    decoys = load_decoy_set(Path(a.decoys), "Q", salt_dir=salts)
    exclude = other_packets(specs=a.exclude_spec or [])["other_sources"]
    built, record = build_q(Path(a.root), decoys, exclude=exclude)
    digests = write_packet(Path(a.out) if a.out else DEFAULT_OUT / "Q", built, record, salt_dir=salts)
    print(json.dumps({"packet": "Q", "items": len(built["public"]["items"]), "key_digest": built["key_digest"],
                      "files": digests}, indent=1))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw-check")
    d.add_argument("--root", required=True)
    s = sub.add_parser("decoys")
    s.add_argument("--spec", required=True)
    s.add_argument("--packet", required=True, choices=("Q", "P1", "P2"))
    s.add_argument("--out", required=True)
    s.add_argument("--testreceipt", help="testreceipt's clone: its 138 items are refused as sources")
    s.add_argument("--other-set", action="append", help="another packet's sealed set: no decoy or description is reused")
    s.add_argument("--other-spec", action="append", help="another packet's decoy spec: no source pull request is reused")
    s.add_argument("--other-key", action="append", help="another packet's key: none of its real items is a source")
    s.add_argument("--other-packet", action="append", help="another packet: no description near-duplicates its items")
    salts_help = f"where salts are written, outside any git working tree (default {SALT_DIR}; a dry run names its own)"
    s.add_argument("--salts", help=salts_help)
    p2 = sub.add_parser("p2")
    p2.add_argument("--root", required=True)
    p2.add_argument("--decoys", required=True)
    p2.add_argument("--out")
    p2.add_argument("--salts", help=salts_help)
    p2.add_argument("--dry-run", action="store_true")
    p1 = sub.add_parser("p1")
    p1.add_argument("--testreceipt", required=True)
    p1.add_argument("--snapshot", required=True)
    p1.add_argument("--logs")
    p1.add_argument("--bodies", help="JSON {url: body}")
    p1.add_argument("--decoys", required=True)
    p1.add_argument("--root", help="the window store holding draw/beacon.json")
    p1.add_argument("--out")
    p1.add_argument("--salts", help=salts_help)
    p1.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("q")
    q.add_argument("--root", required=True)
    q.add_argument("--decoys", required=True)
    q.add_argument("--out")
    q.add_argument("--salts", help=salts_help)
    q.add_argument("--exclude-spec", action="append", help="another packet's decoy spec: its sources are not Q's items")
    a = ap.parse_args(list(argv) if argv is not None else None)
    try:
        return {"draw-check": cmd_draw_check, "decoys": cmd_decoys, "p2": cmd_p2, "p1": cmd_p1, "q": cmd_q}[a.cmd](a)
    except PacketError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
