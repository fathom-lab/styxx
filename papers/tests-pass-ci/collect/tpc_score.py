# -*- coding: utf-8 -*-
"""The scorer (PREREG P13 to P16, P19; sections 8 to 10): decoys scored on the full label, the per-side bars, the
final labels, the gates' metrics, and the receipt `measurement_result.json` that `styxx.prereg_ci` re-derives.

    py -3.12 tpc_score.py score --packets DIR --root STORE --testreceipt DIR --publication FILE --out RECEIPT
    py -3.12 tpc_score.py score ... --smoke         plumbing only: the receipt is scored by type (smoke_verdict)

Order (P9, section 6): no key is opened before every seat's answer file for that packet is sealed (its sha256 beside
it, `tpc_seats.seal_answers`) and still hashes to that digest; a key opens only when it hashes, with its salt, to the
digest committed for it; P2's decoys must be the bytes of the decoy set sealed in the lock row, item by item. Scored for
real, the committed values are read where they were committed (the review of 2026-11-02: the key was checked only
against the .sha256 file beside it): the publication file of the pinned main (`tpc_seal.Publication`, section 14) must
hold, at the commit the publication times name, the key digest beside the packet's sha256 and each seat's answer-file
digest, and these must be the key's, the packet's and the sealed answer files' own; any mismatch refuses the scoring.
Whether the digests were public in that order (`G_KEYS`, and the P1 half of `G_A1_V`) is read from the publication
times the lead writes from the pull requests' opened-at times: {"P1": {"key_public_at", "key_digest": {"sha256",
"packet_sha256", "commit"}, "answers_public_at": {seat: time}, "answers": {seat: {"sha256", "commit"}},
"key_released_at"}, "P2": {"set_in_lock_row_at_signing": true, ...}}; every packet's key digest public before any seat's
earliest answer in its ledger and no later than any answer file's digest, and every answer time strictly before the key
release.

Panel (P13): a voter qualifies on a packet with at least 14 of the 15 NO-side and 13 of the 15 YES-side decoys right;
a decoy answer is right when its step-2 label equals the key's label (NOT_DECIDABLE and NO_ANSWER are wrong). A
packet's panel is valid when H and N qualify and its keys and answers were sealed in order; C votes when it
qualifies. A real item's final label comes from qualified voters only: with three, the label two or three gave, else
NOT_DECIDABLE (NO_MAJORITY); with two, the label both gave, else NOT_DECIDABLE (SPLIT); NO_ANSWER is no vote. The final
step-1 scope comes the same way.

Metrics (P14, section 9.2): n, n_pass, s_lo, c_fail, n_eff = floor(n_pass * s_lo) + c_fail, k, d_W, k_PC, d_P, the
per-agent block and rule R, with Wilson intervals by `styxx.mind.wilson` (its file must be the lock row's bytes) and the
one-sided 95% Clopper-Pearson lower bound of P15. The verdict is `styxx.prereg_ci.rescore` over the PREREG's gates
block, whose digest must be the one section 19 prints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tpc_const import AGENT_ORDER, BOOT, PREREG_REPO_PATH, QUALIFY_NO, QUALIFY_YES, UTC, iso, parse_utc  # noqa: E402

_REPO_ROOT = HERE.parent.parent.parent
_ARCHIVE = Path("C:/Users/heyzo/clawd/wt/_direction_a_2026_11_02/_scout/styxx_310df453")
STYXX_ROOT = _REPO_ROOT if (_REPO_ROOT / "styxx" / "mind.py").exists() else _ARCHIVE
if str(STYXX_ROOT) not in sys.path:
    sys.path.insert(1, str(STYXX_ROOT))  # the repository's styxx, not an installed one; pins checked

import styxx.mind  # noqa: E402
from styxx.prereg_ci import compare, rescore  # noqa: E402

import tpc_decoys  # noqa: E402
import tpc_packet  # noqa: E402
import tpc_seats  # noqa: E402
from tpc_items import item_text as _item_text  # noqa: E402

PREREG_FILE = HERE.parent / Path(PREREG_REPO_PATH).name
PREREG_NAME = Path(PREREG_REPO_PATH).name
GATES_SHA256 = "f2c378b369b6b0b3917fc2f001f02b8364dbd8f5b8f77582c33fc346401d96d8"  # section 19
STYXX_PINS = {"mind.py": "c4c5091b991061f22592e1353da3a9d14889e4817aecf9bb15645de720577115",
              "protocol.py": "45da869e1571396589dc50ed75183de56055a846432ae43dde6aa7fac7519fdc"}
TR_2025_CONTRADICTIONS = 42  # section 1, P19
TR_2025_EXPLICIT = 87
TESTRECEIPT_2026_UPPER = 0.011252  # 2 of 644, section 1 sentence 2
POWER_RATES = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01)  # P16 item 15


class ScoreError(RuntimeError):
    pass


def _blob_sha256(p: Path) -> str:
    """sha256 of a file as git stores it (a checkout with core.autocrlf writes CRLF; the blob is LF)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_styxx_pins() -> dict:
    """The styxx files the scorer runs under must be the lock table's (section 14)."""
    got = {}
    import styxx.protocol  # noqa: PLC0415
    for mod, name in ((styxx.mind, "mind.py"), (styxx.protocol, "protocol.py")):
        got[name] = _blob_sha256(Path(mod.__file__))
    bad = {k: v for k, v in got.items() if v != STYXX_PINS[k]}
    if bad:
        raise ScoreError(f"styxx differs from the lock table's pins: {sorted(bad)}")
    return got


def wilson(k: int, n: int) -> tuple:
    """styxx.mind.wilson, z = 1.96 (P15)."""
    return styxx.mind.wilson(k, n)


def _log_pmf(i: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p)
            + (n - i) * math.log1p(-p))


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p), on the exact terms (pure Python)."""
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if n <= 1000:
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return sum(math.exp(_log_pmf(i, n, p)) for i in range(k, n + 1))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k)."""
    return 1.0 - binom_tail_ge(k + 1, n, p) if k < n else 1.0


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """P15: the one-sided 95% Clopper-Pearson lower bound, the p with P(X >= k | n, p) = 0.05, by bisection to 1e-12;
    k = n gives 0.05^(1/n); k = 0 gives 0."""
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    while hi - lo > 1e-12:
        mid = (lo + hi) / 2
        if binom_tail_ge(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def kmax_clearing(n: int, bar: float = 0.01) -> int:
    """The largest k whose Wilson upper bound over n is under the bar; -1 when none."""
    if n <= 0:
        return -1
    k = -1
    while k + 1 <= n and wilson(k + 1, n)[1] < bar:
        k += 1
    return k


def power_curve(n_eff: int) -> dict:
    """P16 item 15: the chance the whole-suite line clears at each true rate, with no undecided candidate."""
    km = kmax_clearing(n_eff)
    return {str(p): (round(binom_cdf(km, n_eff, p), 4) if km >= 0 else 0.0) for p in POWER_RATES}


# ---- answers, keys and the order of sealing -------------------------------------------------------------------
def load_answers(pdir: Path, packet: str, pk_sha: str) -> dict:
    """{seat: {"rows": {(id, step): row}, "sha256": digest}} for every seat whose answer file is sealed; refuses an
    answer file that no longer hashes to its sealed digest or answered other packet bytes."""
    out = {}
    for seat in tpc_seats.SEATS:
        lp, sp = tpc_seats.ledger_path(pdir, packet, seat), tpc_seats.seal_path(pdir, packet, seat)
        if not lp.exists():
            continue
        if not sp.exists():
            raise ScoreError(f"{lp.name} is not sealed: no key opens before every answer file is (section 6)")
        digest = sp.read_bytes().decode("utf-8").strip()
        if hashlib.sha256(lp.read_bytes()).hexdigest() != digest:
            raise ScoreError(f"{lp.name} no longer hashes to its sealed digest")
        rows = tpc_seats.read_ledger(lp)
        other = {r.get("packet_sha256") for r in rows} - {pk_sha}
        if other:
            raise ScoreError(f"{lp.name} answers other packet bytes than {packet}'s")
        ats = [r.get("at") for r in rows if r.get("final")]
        try:
            earliest = min(parse_utc(a) for a in ats) if ats and all(ats) else None
        except ValueError:
            earliest = None
        out[seat] = {"rows": tpc_seats.final_answers(rows), "sha256": digest,
                     "first_at": iso(earliest) if earliest is not None else None}
    return out


def open_key(pdir: Path, packet: str, answers: dict) -> dict:
    """The packet's key, opened only after its answers are sealed (checked by `load_answers`) and only if it hashes,
    with its salt, to the digest committed for it."""
    if not answers:
        raise ScoreError(f"no seat's answer file for {packet} is sealed; its key stays shut")
    data = (pdir / f"key_{packet}.json").read_bytes()
    salt = bytes.fromhex((pdir / f"key_{packet}.salt").read_bytes().decode("utf-8").strip())
    digest = (pdir / f"key_{packet}.sha256").read_bytes().decode("utf-8").strip()
    if tpc_decoys.salted_digest(salt, data) != digest:
        raise ScoreError(f"the key of {packet} does not hash to its committed digest")
    key = json.loads(data.decode("utf-8"))
    if key.get("packet") != packet:
        raise ScoreError(f"key_{packet}.json is packet {key.get('packet')}'s")
    return {"key": key, "digest": digest}


def check_items(public: dict, key: dict, decoy_set: Optional[list] = None) -> None:
    """Every item's text hashes to its key's item_sha256; with a sealed set, every decoy is the set's, label and all."""
    by_sha = {d["item_sha256"]: d for d in decoy_set} if decoy_set is not None else None
    for it in public["items"]:
        k = key["items"].get(it["id"])
        if k is None:
            raise ScoreError(f"{it['id']} is not in the key")
        text = it["item_text"] if it["steps"] == 2 else _item_text(it["step1_text"], "")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != k["item_sha256"]:
            raise ScoreError(f"{it['id']}'s bytes are not the key's")
        if k.get("decoy") and by_sha is not None:
            d = by_sha.get(k["item_sha256"])
            if d is None:
                raise ScoreError(f"{it['id']} is a decoy whose bytes are not in the sealed set")
            for f in ("side", "construction", "label"):
                if d[f] != k[f]:
                    raise ScoreError(f"{it['id']}: the key's {f} is not the sealed set's")
    if decoy_set is not None:
        n = sum(1 for k in key["items"].values() if k.get("decoy"))
        if n != len(decoy_set):
            raise ScoreError(f"the packet holds {n} decoys, the sealed set {len(decoy_set)}")


# ---- P13, qualification and final labels ---------------------------------------------------------------------
def _s(seat_rows: dict, iid: str, step: int) -> Optional[dict]:
    r = seat_rows.get((iid, step))
    return r.get("answer") if r else None


def decoy_scores(key: dict, answers: dict) -> dict:
    """Per seat: right answers per side and per construction, and whether it qualifies (P13)."""
    out = {}
    decoys = {i: k for i, k in key["items"].items() if k.get("decoy")}
    for seat, a in answers.items():
        side = {"YES": 0, "NO": 0}
        per = {}
        wrong = []
        for iid, k in decoys.items():
            s2 = _s(a["rows"], iid, 2)
            ok = bool(s2) and s2.get("label") == k["label"]
            c = per.setdefault(k["construction"], {"right": 0, "of": 0})
            c["of"] += 1
            if ok:
                side[k["side"]] += 1
                c["right"] += 1
            else:
                wrong.append({"id": iid, "construction": k["construction"],
                              "answered": (s2 or {}).get("label") or tpc_seats.NO_ANSWER})
        out[seat] = {"YES": side["YES"], "NO": side["NO"], "of": {"YES": sum(1 for k in decoys.values() if k["side"] == "YES"),
                                                                  "NO": sum(1 for k in decoys.values() if k["side"] == "NO")},
                     "per_construction": per, "wrong": wrong,
                     "qualified": side["NO"] >= QUALIFY_NO and side["YES"] >= QUALIFY_YES}
    for seat in tpc_seats.SEATS:
        out.setdefault(seat, {"sat": False, "qualified": False})
    return out


def _vote(values: list, q: int) -> tuple:
    """(the final value or None, reason): the value two or more qualified voters gave (with two voters, both)."""
    if q < 2:
        return None, "PANEL_UNDER_TWO"
    c = Counter(v for v in values if v is not None)
    if c:
        v, n = c.most_common(1)[0]
        if n >= 2:
            return v, None
    return None, ("NO_MAJORITY" if q >= 3 else "SPLIT")


def final_labels(key: dict, answers: dict, qualified: list) -> dict:
    """{id: {...}} for real items: the final step-1 scope and step-2 label (P13), the agreeing voters, strict and loose
    readings, and the voters' raw answers."""
    out = {}
    q = len(qualified)
    for iid, k in key["items"].items():
        if k.get("decoy"):
            continue
        s1 = {s: _s(answers[s]["rows"], iid, 1) for s in qualified}
        s2 = {s: _s(answers[s]["rows"], iid, 2) for s in qualified} if k["steps"] == 2 else {}
        scope, scope_why = _vote([(a or {}).get("scope") for a in s1.values()], q)
        rec = {"scope": scope, "scope_why": scope_why, "voters": list(qualified),
               "s1": {s: a for s, a in s1.items()}, "s2": {s: a for s, a in s2.items()},
               "any_whole": any((a or {}).get("scope") == "WHOLE" for a in s1.values()),
               "any_part": any((a or {}).get("scope") == "PART" for a in s1.values())}
        if k["steps"] == 2:
            labels = [(a or {}).get("label") for a in s2.values()]
            lab, why = _vote(labels, q)
            rec["label"] = lab if lab is not None else "NOT_DECIDABLE"
            rec["label_why"] = why
            agree = [s for s, a in s2.items() if (a or {}).get("label") == lab] if lab else []
            rec["agreeing"] = agree
            rec["strict"] = lab is not None and "H" in agree
            rec["loose"] = sorted({(a or {}).get("label") for a in s2.values() if a})
            ev = Counter((s2[s] or {}).get("evidence") for s in agree)
            pre = Counter((s2[s] or {}).get("preexisting") for s in agree)
            rec["evidence"] = ev.most_common(1)[0][0] if ev else None
            rec["preexisting"] = pre.most_common(1)[0][0] if pre else None
        agree1 = [s for s, a in s1.items() if scope and (a or {}).get("scope") == scope]
        boxes = Counter((s1[s] or {}).get("template_box") for s in agree1)
        local = Counter((s1[s] or {}).get("local") for s in agree1)
        rec["template_box"] = bool(boxes.most_common(1)[0][0]) if boxes else False
        rec["local"] = bool(local.most_common(1)[0][0]) if local else False
        out[iid] = rec
    return out


def keys_in_order(pub: dict, packet: str, seats_sat: list, *, p2: bool,
                  first_answer_at: Optional[dict] = None) -> int:
    """1 when the packet's key digest was public before any seat answered (for P2 also its decoy set digest in the
    lock row at the signing commit) and every seat's answer-file digest was public before the key was released, else
    0. With `first_answer_at` ({seat: the earliest "at" of its ledger}), the key digest's public time must be earlier
    than every seat's earliest answer; a seat with no time is not in order."""
    p = pub.get(packet) or {}
    try:
        rel = parse_utc(p["key_released_at"])
        times = p["answers_public_at"]
        if p2 and p.get("set_in_lock_row_at_signing") is not True:
            return 0
        kp = parse_utc(p["key_public_at"])  # every packet's key digest public before its seats read (section 14)
        for s in seats_sat:
            if s not in times or not parse_utc(times[s]) < rel:
                return 0
        if p.get("key_public_at") and any(parse_utc(p["key_public_at"]) > parse_utc(t) for t in times.values()):
            return 0
        if first_answer_at is not None:
            for s in seats_sat:
                fa = first_answer_at.get(s)
                if fa is None or not kp < parse_utc(fa):
                    return 0
    except (KeyError, TypeError, ValueError):
        return 0
    return 1


def _holds(obj: dict, want: dict) -> bool:
    return all(obj.get(k) == v for k, v in want.items())


def publication_problems(pub: dict, panel: dict, read) -> list:
    """Why the committed digests refuse a packet's scoring (section 14, P9): the publication times must name, for the
    key and for each seat's answer file, the commit that carries it and the value committed there; the record read at
    that commit of the pinned main (`read(commit)`: its objects, or None when the commit is not on the pinned main or
    holds no record) must hold it; and every value must be the key's, the packet's and the sealed answer file's own."""
    packet = panel["packet"]
    rec = pub.get(packet) or {}
    out = []
    kd = rec.get("key_digest") or {}
    if kd.get("sha256") != panel["key_digest"] or kd.get("packet_sha256") != panel["packet_sha256"]:
        out.append(f"the publication times' key digest or packet sha256 of {packet} is not this key's or packet's")
    want = {"what": "key", "packet": packet, "key_digest": panel["key_digest"], "packet_sha256": panel["packet_sha256"]}
    lines = read(kd.get("commit") or "")
    if lines is None:
        out.append(f"{packet}'s key digest commit {str(kd.get('commit'))[:12]} is not on the pinned main or holds no "
                   "publication record")
    elif not any(_holds(o, want) for o in lines):
        out.append(f"the publication record at {packet}'s key commit does not hold this key digest beside this packet")
    for seat, digest in sorted(panel["answer_digests"].items()):
        a = (rec.get("answers") or {}).get(seat) or {}
        if a.get("sha256") != digest:
            out.append(f"the publication times' answer digest of {packet}/{seat} is not the sealed answer file's")
        w = {"what": "answers", "packet": packet, "seat": seat, "sha256": digest}
        lines = read(a.get("commit") or "")
        if lines is None or not any(_holds(o, w) for o in lines):
            out.append(f"the answer-file digest of {packet}/{seat} is not committed on the pinned main")
    return out


def _default_reader():
    """The publication record at a commit of the pinned main (`tpc_seal.Publication.at`)."""
    import tpc_seal
    return tpc_seal.Publication().at


# ---- testreceipt's labels (P19) ------------------------------------------------------------------------------
JUDGE_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([a-z-]+)\s*\|", re.M)


def _judge(text: str) -> dict:
    return {int(n): lab for n, lab in JUDGE_ROW.findall(text)}


def _resolved(toml_text: str, section: str) -> dict:
    out = {}
    block = re.search(r"^\[" + re.escape(section) + r"\]\s*$(.*?)(?=^\[|\Z)", toml_text, re.M | re.S)
    if block:
        for n, lab in re.findall(r'^(\d+)\s*=\s*"([a-z-]+)"', block.group(1), re.M):
            out[int(n)] = lab
    return out


def testreceipt_labels(tr_dir: Path) -> dict:
    """{(year, n): label}: the judges' label where they agree, the resolved label where `[m2_resolved]` or
    `[fresh_resolved]` gives one (P19); an item neither settles is refused."""
    r = Path(tr_dir) / "evals" / "results"
    toml = (Path(tr_dir) / "evals" / "aidev_labels.toml").read_text(encoding="utf-8")
    out = {}
    for year, a, b, sec in (("2025", "judge1-m2.md", "judge2-m2.md", "m2_resolved"),
                            ("2026", "judge1-fresh.md", "judge2-fresh.md", "fresh_resolved")):
        j1, j2 = _judge((r / a).read_text(encoding="utf-8")), _judge((r / b).read_text(encoding="utf-8"))
        res = _resolved(toml, sec)
        for n in sorted(set(j1) | set(j2)):
            lab = res.get(n) or (j1.get(n) if j1.get(n) == j2.get(n) else None)
            if lab is None:
                raise ScoreError(f"testreceipt's {year} item {n}: the judges differ and nothing resolves it")
            out[(year, n)] = lab
    return out


# ---- the metrics ---------------------------------------------------------------------------------------------
def _panel(pdir: Path, packet: str, *, decoy_set: Optional[list] = None, dry_run: bool = False) -> Optional[dict]:
    pp = pdir / f"packet_{packet}.json"
    if not pp.exists():
        return None
    public, pk_sha = tpc_seats.load_packet(pp)
    if public.get("beacon", {}).get("kind") == "stand_in" and not dry_run:
        raise ScoreError(f"packet {packet} was built under the stand-in hash; it is never scored for real")
    answers = load_answers(pdir, packet, pk_sha)
    opened = open_key(pdir, packet, answers)
    key = opened["key"]
    check_items(public, key, decoy_set)
    scores = decoy_scores(key, answers)
    qualified = [s for s in tpc_seats.SEATS if s in answers and scores[s]["qualified"]]
    return {"packet": packet, "packet_sha256": pk_sha, "key": key, "key_digest": opened["digest"],
            "answers": answers, "scores": scores, "qualified": qualified,
            "answer_digests": {s: a["sha256"] for s, a in answers.items()},
            "first_answer_at": {s: a.get("first_at") for s, a in answers.items()},
            "finals": final_labels(key, answers, qualified)}


def _rows_of(root: Path) -> list:
    return tpc_packet._read_rows(Path(root))


def _reader_whole(r: dict) -> bool:
    return any(c.get("kind") == "pass" and c.get("explicit") and c.get("scope") == "full" for c in r.get("claims") or [])


def a2_metrics(rows: list, panel: Optional[dict], pub: dict, *, boot_seed: str, boot: int = BOOT) -> tuple:
    """(the gated a2 metrics of P14, the counts and printed readings of P16)."""
    s = tpc_packet.analysis_sets(rows)
    n = len(s["m_cap"])
    counts: dict = {"n": n}
    if n < 400 or panel is None:
        if n >= 400:
            raise ScoreError("K1 holds (n >= 400) but packet P2 is not scored: the A2 metrics cannot be computed")
        gated = {"n_claimers_ci": n, "keys_in_order": -1, "human_qualified": -1, "nonclaude_qualified": -1,
                 "whole_wilson_upper_worst": 1.0, "agent_whole_wilson_lower_max": 1.0,
                 "part_contradicted_wilson_upper": 1.0, "rule_firings": -1, "rule_cp_lower": 0.0}
        counts["p2_ran"] = False
        return gated, counts
    key, finals, scores = panel["key"], panel["finals"], panel["scores"]
    pid_of = {}
    strata_of = {}
    for iid, k in key["items"].items():
        if not k.get("decoy"):
            pid_of[k["pr_id"]] = iid
            strata_of[k["pr_id"]] = k.get("strata") or []
    lab = {pid: finals[iid] for pid, iid in pid_of.items()}

    def label(pid) -> Optional[str]:
        return (lab.get(pid) or {}).get("label")

    by_id = {r["pr_id"]: r for r in rows}
    m_cap = s["m_cap"]
    cand_m = s["cand_m"]
    passing = [r for r in m_cap if (r.get("ci") or {}).get("outcome") == "tests passed"]
    n_pass = len(passing)
    au4 = [pid for pid, st in strata_of.items() if "AU4" in st]
    au4_claims = sum(1 for pid in au4 if (lab.get(pid) or {}).get("scope") in ("WHOLE", "PART"))
    s_lo = wilson(au4_claims, len(au4))[0] if au4 else 0.0
    c_fail = sum(1 for r in cand_m if label(r["pr_id"]) != "NOT_A_CLAIM")
    n_eff = math.floor(n_pass * s_lo) + c_fail
    k = sum(1 for r in cand_m if label(r["pr_id"]) == "WHOLE_CONTRADICTED")

    def undecided(r) -> bool:
        return label(r["pr_id"]) in (None, "NOT_DECIDABLE")

    d_w = sum(1 for r in cand_m if undecided(r) and (_reader_whole(r) or (lab.get(r["pr_id"]) or {}).get("any_whole")))
    k_pc = sum(1 for r in cand_m if label(r["pr_id"]) == "PART_CONTRADICTED")
    d_p = sum(1 for r in cand_m if undecided(r) and (lab.get(r["pr_id"]) or {}).get("any_part"))
    whole_up = wilson(k + d_w, n_eff)[1] if n_eff > 0 else 1.0
    part_up = wilson(k_pc + d_p, n_eff)[1] if n_eff > 0 else 1.0
    per_agent = {}
    lows = []
    for g in AGENT_ORDER:
        a_cap, cand = s["a_cap"][g], s["cand_a"][g]
        cut = [r["pr_id"] for r in cand if r["pr_id"] not in pid_of]
        kg = sum(1 for r in cand if label(r["pr_id"]) == "WHOLE_CONTRADICTED")
        dg = sum(1 for r in cand if r["pr_id"] in pid_of and undecided(r)
                 and (_reader_whole(r) or (lab.get(r["pr_id"]) or {}).get("any_whole")))
        entry = {"n": len(a_cap), "candidates": len(cand), "k": kg, "d_W": dg, "cut": len(cut)}
        if len(a_cap) >= 100:
            lo, hi = wilson(kg + dg, len(a_cap))
            entry["wilson_worst"] = [lo, hi]
            lows.append(lo)
        per_agent[g] = entry
    capped_ids = {r["pr_id"] for r in m_cap} | {r["pr_id"] for g in AGENT_ORDER for r in s["a_cap"][g]}
    firings = sorted(pid for pid in capped_ids if (by_id[pid].get("rule_r") or {}).get("fires"))
    right = sum(1 for pid in firings if label(pid) == "WHOLE_CONTRADICTED" and pid in pid_of)
    gated = {"n_claimers_ci": n,
             "keys_in_order": keys_in_order(pub, "P2", sorted(panel["answers"]), p2=True,
                                            first_answer_at=panel["first_answer_at"]),
             "human_qualified": int(scores["H"]["qualified"]), "nonclaude_qualified": int(scores["N"]["qualified"]),
             "whole_wilson_upper_worst": whole_up, "agent_whole_wilson_lower_max": max(lows) if lows else 0.0,
             "part_contradicted_wilson_upper": part_up, "rule_firings": len(firings),
             "rule_cp_lower": cp_lower(right, len(firings)) if firings else 0.0}
    counts.update({"p2_ran": True, "n_pass": n_pass, "au4_rows": len(au4), "au4_claims": au4_claims, "s_lo": s_lo,
                   "c_fail": c_fail, "n_eff": n_eff, "k": k, "d_W": d_w, "k_PC": k_pc, "d_P": d_p,
                   "rule_firings_right": right, "per_agent": per_agent,
                   "j": int(scores["H"]["qualified"]), "qualified": panel["qualified"]})
    counts["printed"] = a2_printed(rows, s, lab, strata_of, pid_of, counts, panel, boot_seed=boot_seed, boot=boot)
    return gated, counts


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n > 0 else (None, None)
    return {"k": k, "n": n, "wilson": [lo, hi]}


def bootstrap(rows_kn: list, seed_hex: str, boot: int = BOOT) -> list:
    """P15: repository-cluster interval of k/n; rows_kn: [(repo, k_r, n_r)]; the 2.5th and 97.5th percentiles."""
    if not rows_kn:
        return [None, None]
    rng = random.Random(int(hashlib.sha256((seed_hex + ":bootstrap").encode("utf-8")).hexdigest(), 16))
    m = len(rows_kn)
    ks = [a for _, a, _ in rows_kn]
    ns = [b for _, _, b in rows_kn]
    idx = range(m)
    vals = []
    for _ in range(boot):
        picks = rng.choices(idx, k=m)
        kk = sum(map(ks.__getitem__, picks))
        nn = sum(map(ns.__getitem__, picks))
        vals.append(kk / nn if nn else 0.0)
    vals.sort()
    return [vals[int(0.025 * (boot - 1))], vals[int(0.975 * (boot - 1))]]


def a2_printed(rows: list, s: dict, lab: dict, strata_of: dict, pid_of: dict, counts: dict, panel: dict, *,
               boot_seed: str, boot: int = BOOT) -> dict:
    """P16's readings, deciding nothing."""
    def label(pid):
        return (lab.get(pid) or {}).get("label")

    m_cap, cand_m = s["m_cap"], s["cand_m"]
    n, k = counts["n"], counts["k"]
    out: dict = {}
    claim_ci_m = [r for r in s["claim_ci"] if r.get("in_M")]
    out["M_uncapped"] = {"n": len(claim_ci_m),
                         "candidates": sum(1 for r in claim_ci_m if (r.get("ci") or {}).get("outcome") == "tests failed"),
                         "k_labelled": sum(1 for r in claim_ci_m if label(r["pr_id"]) == "WHOLE_CONTRADICTED")}
    repos = Counter((r.get("repo") or "").lower() for r in m_cap)
    cand_ids = {r["pr_id"] for r in cand_m}
    if repos:
        top = repos.most_common(1)[0][0]
        rest = [r for r in m_cap if (r.get("repo") or "").lower() != top]
        out["without_largest_repository"] = dict(_rate(sum(1 for r in rest if r["pr_id"] in cand_ids and
                                                           label(r["pr_id"]) == "WHOLE_CONTRADICTED"), len(rest)),
                                                 repository_claimers=repos[top])
    box = {r["pr_id"] for r in cand_m if (lab.get(r["pr_id"]) or {}).get("template_box")}
    out["without_template_box"] = _rate(sum(1 for r in cand_m if r["pr_id"] not in box and label(r["pr_id"]) ==
                                            "WHOLE_CONTRADICTED"), n - len(box))
    stale_m = [r for r in m_cap if r.get("stale")]
    out["without_stale"] = dict(_rate(sum(1 for r in cand_m if not r.get("stale") and label(r["pr_id"]) ==
                                          "WHOLE_CONTRADICTED"), n - len(stale_m)),
                                stale_share_of_candidates=(sum(1 for r in cand_m if r.get("stale")) / len(cand_m)
                                                           if cand_m else None))
    strict_k = sum(1 for r in cand_m if label(r["pr_id"]) == "WHOLE_CONTRADICTED" and (lab.get(r["pr_id"]) or {}).get("strict"))
    out["strict_human_agreeing"] = _rate(strict_k, n)
    out["whole_strict"] = _rate(k, n)
    out["loose_whole"] = sum(1 for r in cand_m if "WHOLE_CONTRADICTED" in ((lab.get(r["pr_id"]) or {}).get("loose") or []))
    au4 = [pid for pid, st in strata_of.items() if "AU4" in st]
    au4_whole = sum(1 for pid in au4 if (lab.get(pid) or {}).get("scope") == "WHOLE")
    whole_cands = sum(1 for r in cand_m if (lab.get(r["pr_id"]) or {}).get("scope") == "WHOLE")
    denom = whole_cands + (counts["n_pass"] * au4_whole / len(au4) if au4 else 0)
    out["per_whole_suite_claimer"] = _rate(k, int(round(denom)))
    out["local_whole_beside_failing"] = sum(1 for r in cand_m if (lab.get(r["pr_id"]) or {}).get("local"))
    out["without_names_only"] = sum(1 for r in cand_m if label(r["pr_id"]) == "WHOLE_CONTRADICTED"
                                    and (lab.get(r["pr_id"]) or {}).get("evidence") != "NAMES_ONLY")
    out["without_preexisting"] = sum(1 for r in cand_m if label(r["pr_id"]) == "WHOLE_CONTRADICTED"
                                     and (lab.get(r["pr_id"]) or {}).get("preexisting") != "yes")
    au_rows = {name: [pid for pid, st in strata_of.items() if name in st] for name in ("AU1", "AU2", "AU3")}
    au1_whole = sum(1 for pid in au_rows["AU1"] if label(pid) == "WHOLE_CONTRADICTED")
    out["AU1"] = {"rows": len(au_rows["AU1"]), "whole_contradicted": au1_whole, "population": len(s["au1_pop"]),
                  "extrapolated": (au1_whole / len(au_rows["AU1"]) * len(s["au1_pop"])) if au_rows["AU1"] else None}
    au2_whole = sum(1 for pid in au_rows["AU2"] if label(pid) == "WHOLE_CONTRADICTED")
    out["AU2"] = {"rows": len(au_rows["AU2"]), "whole_contradicted": au2_whole, "population": len(s["au2_pop"]),
                  "weighted": (au2_whole * len(s["au2_pop"]) / len(au_rows["AU2"])) if au_rows["AU2"] else None}
    au3_claim = sum(1 for pid in au_rows["AU3"] if (lab.get(pid) or {}).get("scope") in ("WHOLE", "PART"))
    out["AU3_claim_share"] = _rate(au3_claim, len(au_rows["AU3"]))
    out["AU4_claim_share"] = _rate(counts["au4_claims"], counts["au4_rows"])
    pairs = {}
    agent_of = {r["pr_id"]: r["agent"] for r in rows}
    seats = [x for x in tpc_seats.SEATS if x in panel["answers"]]
    for i, a in enumerate(seats):
        for b in seats[i + 1:]:
            for stratum in ("Claude_Code", "other"):
                agree = both = 0
                for pid, iid in pid_of.items():
                    if pid not in agent_of or ((agent_of[pid] == "Claude_Code") != (stratum == "Claude_Code")):
                        continue
                    la = _s(panel["answers"][a]["rows"], iid, 2)
                    lb = _s(panel["answers"][b]["rows"], iid, 2)
                    if la and lb:
                        both += 1
                        agree += int(la["label"] == lb["label"])
                pairs[f"{a}-{b}:{stratum}"] = {"agree": agree, "both": both}
    out["pair_agreement"] = pairs
    out["per_agent"] = counts["per_agent"]
    elig = [r for r in rows if r.get("eligible")]
    imp = [r for r in elig if r.get("screen") == "implied_only"]
    out["implied_only"] = {"rows": len(imp), "beside_failing": sum(1 for r in imp if (r.get("ci") or {}).get("outcome")
                                                                     == "tests failed")}
    lo, hi = wilson(k, n) if n else (None, None)
    out["sentences"] = {"strict_wilson": [lo, hi], "strict_lower_at_or_over_1pct": bool(lo is not None and lo >= 0.01),
                        "testreceipt_2026_upper": TESTRECEIPT_2026_UPPER,
                        "strict_lower_over_testreceipt_upper": bool(lo is not None and lo > TESTRECEIPT_2026_UPPER)}
    out["power_at_n_eff"] = power_curve(counts["n_eff"])
    pb = sum(1 for r in cand_m if label(r["pr_id"]) == "PART_BESIDE_FAILURE")
    out["part_beside_failure"] = _rate(pb, n)
    out["part_literal_share"] = {"pc_plus_pb": counts["k_PC"] + pb, "n": n,
                                 "share": (counts["k_PC"] + pb) / n if n else None, "testreceipt": 0.0466}
    by_stratum = Counter()
    labels_r = Counter()
    for pid, st in strata_of.items():
        if "rule_R" in st:
            for x in st:
                by_stratum[x] += 1
            labels_r[label(pid) or "none"] += 1
    out["rule_R"] = {"by_stratum": dict(by_stratum), "labels": dict(labels_r)}
    per_repo = {}
    for r in m_cap:
        key = (r.get("repo") or "").lower()
        a, b = per_repo.get(key, (0, 0))
        per_repo[key] = (a + int(label(r["pr_id"]) == "WHOLE_CONTRADICTED"), b + 1)
    out["bootstrap_k_over_n"] = bootstrap([(rp, a, b) for rp, (a, b) in sorted(per_repo.items())], boot_seed, boot)
    out["bootstrap_resamples"] = boot
    return out


def a1_metrics(panel: Optional[dict], tr_labels: dict, pub: dict) -> tuple:
    """(the gated a1 metrics of P14, the counts and per-item rows of P19)."""
    if panel is None:
        return {"panel_valid": 0, "confirmed": -1, "added": -1, "k2025_strict": -1, "k2025_worst": -1}, {"p1_ran": False}
    key, finals, scores = panel["key"], panel["finals"], panel["scores"]
    seats = sorted(panel["answers"])
    valid = int(keys_in_order(pub, "P1", seats, p2=False, first_answer_at=panel["first_answer_at"]) == 1
                and scores["H"]["qualified"] and scores["N"]["qualified"])
    explicit, implied, rows = [], [], []
    for iid, k in key["items"].items():
        if k.get("decoy"):
            continue
        tr = tr_labels.get((k["year"], k["n"]))
        f = finals[iid]
        rows.append({"id": iid, "year": k["year"], "n": k["n"], "url": k["url"], "agent": k.get("agent"),
                     "claim": k.get("claim"), "testreceipt": tr, "final": f.get("label"), "scope": f.get("scope"),
                     "voters": {s: (f["s2"].get(s) or {}).get("label") for s in f["s2"]}})
        if k["year"] == "2025":
            (explicit if k.get("claim") == "pass" else implied).append((tr, f.get("label")))
    contra = [x for x in explicit if x[0] == "contradiction"]
    other = [x for x in explicit if x[0] != "contradiction"]
    if len(explicit) != TR_2025_EXPLICIT or len(contra) != TR_2025_CONTRADICTIONS:
        raise ScoreError(f"P1 holds {len(explicit)} explicit 2025 items and {len(contra)} contradictions; P19 reads "
                         f"{TR_2025_EXPLICIT} and {TR_2025_CONTRADICTIONS}")
    confirmed = sum(1 for _, f in contra if f == "WHOLE_CONTRADICTED")
    added = sum(1 for _, f in other if f == "WHOLE_CONTRADICTED")
    undecided = sum(1 for _, f in explicit if f == "NOT_DECIDABLE")
    counts = {"p1_ran": True, "qualified": panel["qualified"], "j": int(scores["H"]["qualified"]),
              "explicit": len(explicit), "implied": len(implied), "undecided": undecided,
              "confirmed_raw": confirmed, "added_raw": added,
              "confirmed_share_cp_lower": cp_lower(confirmed, len(contra)),
              "implied_whole_contradicted": sum(1 for _, f in implied if f == "WHOLE_CONTRADICTED"),
              "confusion_2025": dict(Counter(f"{t}->{f}" for t, f in explicit + implied)),
              "rows": sorted(rows, key=lambda r: (r["year"], r["n"]))}
    if not valid:
        return {"panel_valid": 0, "confirmed": -1, "added": -1, "k2025_strict": -1, "k2025_worst": -1}, counts
    return {"panel_valid": 1, "confirmed": confirmed, "added": added, "k2025_strict": confirmed + added,
            "k2025_worst": confirmed + added + undecided}, counts


# ---- the receipt ---------------------------------------------------------------------------------------------
def make_receipt(a1: dict, a2: dict, printed: dict, *, smoke: bool, prereg_text: Optional[str] = None) -> dict:
    """The receipt of P14: prereg, smoke, result, verdict, gates, gates_sha256; the verdict re-derived by
    styxx.prereg_ci.rescore under the PREREG's sealed block and checked by its compare."""
    text = prereg_text if prereg_text is not None else PREREG_FILE.read_text(encoding="utf-8")
    receipt = {"prereg": PREREG_NAME, "smoke": bool(smoke), "result": {"a1": a1, "a2": a2, "printed": printed}}
    scored = rescore(text, receipt)
    if scored["code"] is not None:
        raise ScoreError(f"the PREREG's table does not score this receipt: {scored['code']} {scored['detail']}")
    if scored["gates_sha256"] != GATES_SHA256:
        raise ScoreError(f"the gates block read is {scored['gates_sha256'][:12]}..., not the sealed {GATES_SHA256[:12]}...")
    receipt.update(verdict=scored["verdict"], gates=scored["gates"], gates_sha256=scored["gates_sha256"])
    found = [f for f in compare(scored, receipt) if f["kind"] == "FAIL"]
    if found:
        raise ScoreError(f"prereg-ci's compare refuses the receipt: {found}")
    return receipt


def _finite(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def receipt_bytes(receipt: dict) -> bytes:
    return (json.dumps(_finite(receipt), indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _prereg_for_scoring(prereg_text: Optional[str], *, real_p2: bool, guard=None) -> tuple:
    """(text, how): the PREREG the scorer reads. A caller's text is used as given (tests); for P2 scored for real, the
    text the open seal guard read at the signing commit, refused unless this checkout's PREREG is those bytes (so the
    lock row read is the signed one, never a working copy); otherwise this checkout's file."""
    if prereg_text is not None:
        return prereg_text, "text given by the caller"
    if real_p2:
        if guard is None:
            from tpc_seal import SealGuard
            guard = SealGuard()
        from tpc_seal import SealRefused
        try:
            return guard.signed_prereg(PREREG_FILE), "the signed blob, through the seal guard"
        except SealRefused as e:
            raise ScoreError(str(e)) from e
    return PREREG_FILE.read_text(encoding="utf-8"), "this checkout's file"


def score(packets: Path, root: Optional[Path], tr_dir: Optional[Path], pub: dict, *, smoke: bool = False,
          prereg_text: Optional[str] = None, dry_run: bool = False, boot: int = BOOT, guard=None,
          publication=None) -> dict:
    """The whole scoring: P1 and P2 panels, the A1 and A2 metrics, the receipt. Scored for real, every panel's key
    and answer-file digests are checked where they were committed (`publication`: commit -> the record's objects, by
    default read on the pinned main)."""
    pins = check_styxx_pins()
    packets = Path(packets)
    has_p2 = (packets / "P2" / "packet_P2.json").exists()
    text, how = _prereg_for_scoring(prereg_text, real_p2=has_p2 and not (smoke or dry_run), guard=guard)
    lock = tpc_packet.p2_lock_digest(text)
    p2set = None
    if has_p2:
        if lock is None and not (smoke or dry_run):
            raise ScoreError("the lock table's P2 decoy row is unfilled; P2 cannot be scored for real")
        # scored after the release: each salt is published beside its set
        p2set = tpc_packet.load_decoy_set(packets / "P2", "P2", salt_dir=None,
                                          lock_digest=None if (smoke or dry_run) else lock)
    p1set = None
    if (packets / "P1" / "decoyset_P1.json").exists():
        p1set = tpc_packet.load_decoy_set(packets / "P1", "P1", salt_dir=None)
    p1 = _panel(packets / "P1", "P1", decoy_set=p1set, dry_run=dry_run) if (packets / "P1").exists() else None
    p2 = _panel(packets / "P2", "P2", decoy_set=p2set, dry_run=dry_run) if (packets / "P2").exists() else None
    if not (smoke or dry_run):
        read = publication or _default_reader()
        for p in (p1, p2):
            probs = publication_problems(pub, p, read) if p is not None else []
            if probs:
                raise ScoreError(f"the committed digests refuse packet {p['packet']}: " + "; ".join(probs))
    rows = _rows_of(root) if root else []
    tr = testreceipt_labels(tr_dir) if (tr_dir and p1) else {}
    a1, a1_counts = a1_metrics(p1, tr, pub)
    seed = (p2 or {}).get("key", {}).get("seed") or "no-beacon"
    a2, a2_counts = a2_metrics(rows, p2, pub, boot_seed=seed, boot=boot)
    printed = {"a1": a1_counts, "a2": a2_counts, "styxx_pins": pins, "scored_at": iso(dt.datetime.now(UTC)),
               "committed_digests_checked": not (smoke or dry_run),
               "prereg_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "prereg_read": how,
               "panels": {p["packet"]: {"qualified": p["qualified"], "answer_digests": p["answer_digests"],
                                        "key_digest": p["key_digest"], "packet_sha256": p["packet_sha256"],
                                        "decoys": {s: {k: v for k, v in sc.items() if k != "wrong"}
                                                   for s, sc in p["scores"].items()}}
                          for p in (p1, p2) if p is not None}}
    return make_receipt(a1, a2, _finite(printed), smoke=smoke, prereg_text=text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--packets", required=True, help="a folder holding P1/ and P2/ (packets, keys, answers)")
    s.add_argument("--root", help="the window store (row records)")
    s.add_argument("--testreceipt", help="testreceipt's clone at 95864ef9 (its labels, P19)")
    s.add_argument("--publication", help="the publication record (when each digest became public)")
    s.add_argument("--out", required=True)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="packets built under the stand-in hash; implies --smoke")
    a = ap.parse_args(argv)
    pub = json.loads(Path(a.publication).read_bytes().decode("utf-8")) if a.publication else {}
    out = Path(a.out)
    if out.exists():
        print(f"refused: {out} exists; a receipt is never rewritten in place", file=sys.stderr)
        return 2
    try:
        rec = score(Path(a.packets), Path(a.root) if a.root else None, Path(a.testreceipt) if a.testreceipt else None,
                    pub, smoke=a.smoke or a.dry_run, dry_run=a.dry_run)
    except (ScoreError, tpc_packet.PacketError, tpc_decoys.DecoyError, tpc_seats.SeatError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    data = receipt_bytes(rec)
    out.write_bytes(data)
    print(json.dumps({"verdict": rec["verdict"], "gates_sha256": rec["gates_sha256"], "receipt": str(out),
                      "sha256": hashlib.sha256(data).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
