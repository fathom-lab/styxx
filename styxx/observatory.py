# -*- coding: utf-8 -*-
"""styxx.observatory — a model, watched.

    obs = Observatory("observatory/qwen2.5-1.5b", model_id="Qwen/Qwen2.5-1.5B")
    obs.observe(probe, n_null=2)          # one observation: floor, fingerprint, distance to the
                                          # baseline and to the previous observation, plates, a
                                          # log line chained to the last one
    Observatory.verify(root, expect_head=..., expect_entries=...)
    Observatory.status(root)              # a table for humans

Policy, fixed: the BASELINE is the first accepted observation and stays the baseline until a
`rebaseline(reason)` entry is appended — drift is never absorbed silently. The code cannot tell a
human from a script: a rebaseline is an explicit, reasoned entry in the chain (an empty reason is
refused), and anchoring its hash in the ledger is what dates it. Each observation reports the
distance to the baseline and to the previous observation, each against the floor measured that
day from n_null repeated calls on the same loaded weights — repeat-call noise, not reload noise —
and the floor actually applied (never below checksum.RESOLUTION_NATS) is written beside the
measured one, because a verdict graded against 1e-4 while the table shows 0 is a table that lies.

What verify() checks (v1, 2026-09-13; v0 checked only the chain and the fingerprint bytes, so a
forward-rehashed forgery of every verdict, a truncated log, and a replaced plate all verified):
the chain; every fingerprint file's bytes; coefficients_sha256 re-derived from the fingerprint
FILE (the rounded bytes a stranger has — v0 hashed the in-memory array, which never matched the
file); every vs_baseline and vs_previous verdict and number re-derived from the two fingerprint
files with the floor the line applied; and, when the caller pins them, the head hash and the
entry count — a prefix of a chained log is a valid chain, so the head must be pinned outside the
file (a sworn span, a ledger anchor). What it does not check: plate bytes (a rendering of the
fingerprints, not evidence), and whether `when` is true — `when` is a label the caller supplies;
`taken` is the clock at observation, and both are written and shown.

The log is append-only and chained like charon; anchoring a line's hash in the ledger dates it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable

import numpy as np

from . import checksum as ck
from .geoplate import coefficients, coefficients_sha256

TOL = 1e-9   # re-derivation of a logged number from the same bytes and the same seed is exact; this is slack for printing


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _line_hash(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


def _load_fp_file(path: str) -> ck.Fingerprint:
    """The fingerprint as a stranger has it: the rounded bytes on disk, never the in-memory array."""
    d = json.load(open(path, encoding="utf-8"))
    return ck.Fingerprint(model_id=d["model_id"], canary_sha256=d["canary_sha256"], tokenizer_id=d["tokenizer_id"],
                          ids=d["ids"], mean_lp=np.asarray(d["mean_lp"], dtype=np.float64),
                          rdm=np.asarray(d["rdm"], dtype=np.float64), n_tokens=d["n_tokens"], created=d["created"],
                          kind=d.get("kind", "full"), k=int(d.get("k", 0)), draw=d.get("draw"))


def _dist_body(ref_seq: int, a: ck.Fingerprint, b: ck.Fingerprint, floor_applied: float) -> dict:
    d = ck.distance(a, b, floor_nats=floor_applied)
    return {"seq": ref_seq, "verdict": d.verdict, "mean_abs_nats": d.mean_abs_nats, "ci": list(d.ci_mean_abs), "rdm_r": d.rdm_r}


def _same(x, y) -> bool:
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_same(p, q) for p, q in zip(x, y))
    if isinstance(x, float) and isinstance(y, float):
        if np.isnan(x) and np.isnan(y):
            return True
        return abs(x - y) <= TOL
    return x == y


class Observatory:
    def __init__(self, root: str, model_id: str, canaries=ck.CANARIES, tokenizer_id: str = "", draw: dict | None = None):
        if not tokenizer_id:
            raise ValueError("name the tokenization (tokenizer_id): fingerprints with none cannot be compared")
        self.root, self.model_id, self.canaries, self.tokenizer_id = root, model_id, canaries, tokenizer_id
        self.draw = ck._check_draw(draw, canaries)   # a beacon draw record when the canaries were drawn
        os.makedirs(os.path.join(root, "fingerprints"), exist_ok=True)
        os.makedirs(os.path.join(root, "plates"), exist_ok=True)
        self.log = os.path.join(root, "log.jsonl")

    # ----------------------------------------------------------------------------------- reading
    def entries(self) -> list[dict]:
        if not os.path.exists(self.log):
            return []
        return [json.loads(l) for l in open(self.log, encoding="utf-8") if l.strip()]

    def _load_fp(self, entry: dict) -> ck.Fingerprint:
        return _load_fp_file(os.path.join(self.root, entry["fingerprint"]))

    def baseline(self) -> dict | None:
        es = self.entries()
        marks = [e for e in es if e.get("kind") in ("baseline", "rebaseline")]
        return marks[-1] if marks else None

    def _append(self, body: dict, prev_hash: str) -> dict:
        body["prev"] = prev_hash
        body["entry_hash"] = _line_hash(prev_hash, {k: v for k, v in body.items() if k != "prev"} | {"prev": prev_hash})
        with open(self.log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
        return body

    # ----------------------------------------------------------------------------------- writing
    def observe(self, probe: Callable, n_null: int = 2, when: str | None = None, note: str = "") -> dict:
        """One observation. `when` is a LABEL (defaults to the clock); `taken` is always the clock."""
        taken = _now()
        when = when or taken
        fps = [ck.fingerprint(probe, self.model_id, self.canaries, tokenizer_id=self.tokenizer_id, draw=self.draw)
               for _ in range(max(2, n_null))]
        floor = ck.null_floor(fps)
        floor_applied = max(floor, ck.RESOLUTION_NATS)
        es = self.entries()
        seq = len(es) + 1
        tag = f"{seq:04d}_{when[:10]}"
        fp_rel = os.path.join("fingerprints", f"{tag}.json")
        fp_path = os.path.join(self.root, fp_rel)
        with open(fp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(fps[0].to_json(), f, separators=(",", ":"))
        fp = _load_fp_file(fp_path)                       # from here on, only the bytes a stranger has
        body = {"seq": seq, "when": when, "taken": taken, "model_id": self.model_id, "canary_sha256": fp.canary_sha256,
                "fingerprint": fp_rel.replace(os.sep, "/"), "fingerprint_sha256": _sha_file(fp_path),
                "coefficients_sha256": coefficients_sha256(coefficients(fp.rdm)),
                "null_floor_nats": floor, "floor_applied_nats": floor_applied, "n_null": len(fps), "note": note}
        base = self.baseline()
        if base is None:
            body["kind"] = "baseline"
        else:
            body["kind"] = "observation"
            body["vs_baseline"] = _dist_body(base["seq"], self._load_fp(base), fp, floor_applied)
            prev = es[-1]
            body["vs_previous"] = _dist_body(prev["seq"], self._load_fp(prev), fp, floor_applied)
        return self._append(body, es[-1]["entry_hash"] if es else "0" * 64)

    def rebaseline(self, reason: str) -> dict:
        """Accept the latest observation as the new baseline. The reason is bound into the chain and
        must be non-empty; nothing here can tell a human from a script, and the docstring says so."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a rebaseline needs a reason; an empty one is a silent absorption of drift")
        es = self.entries()
        if not es:
            raise ValueError("nothing to rebaseline")
        last = es[-1]
        body = {k: last[k] for k in ("seq", "when", "model_id", "canary_sha256", "fingerprint", "fingerprint_sha256",
                                      "coefficients_sha256", "null_floor_nats", "n_null") if k in last}
        if "floor_applied_nats" in last:
            body["floor_applied_nats"] = last["floor_applied_nats"]
        body.update({"seq": len(es) + 1, "kind": "rebaseline", "reason": reason.strip(), "taken": _now()})
        return self._append(body, last["entry_hash"])

    def render(self) -> str | None:
        """Plates for the latest observation: its geometry, and the drift plate against the baseline.
        The drift plate draws belief-geometry disagreement only: a DRIFT verdict from a uniform
        shift of every log-prob leaves the geometry unchanged and the plate empty, and the subtitle
        says so when that happens."""
        from .geoplate import render_grid, render_drift
        es = self.entries()
        if not es:
            return None
        last, base = es[-1], self.baseline()
        Wl = coefficients(self._load_fp(last).rdm); Wb = coefficients(self._load_fp(base).rdm)
        out = os.path.join(self.root, "plates", f"{last['seq']:04d}_{last['when'][:10]}.png")
        vb = last.get("vs_baseline", {})
        sub = (f"vs baseline #{vb['seq']}: {vb['mean_abs_nats']:.4f} nats/token, r = {vb['rdm_r']:.3f}  ({vb['verdict']})"
               if vb else "baseline")
        if vb and vb["verdict"] == "DRIFT" and vb["rdm_r"] > 0.999:
            sub += "  — geometry unchanged, log-probs moved: this plate is empty by construction"
        render_grid([(f"baseline #{base['seq']} ({base['when'][:10]})", "the model when watching began", Wb),
                     (f"observation #{last['seq']} ({last['when'][:10]})", sub, Wl)],
                    out, title=f"observatory: {self.model_id}", ncols=2)
        render_drift([(f"#{base['seq']} → #{last['seq']}", sub, Wb, Wl)], out.replace(".png", "_drift.png"),
                     title="drift plate: belief geometry moved since the baseline (log-prob shifts draw nothing)",
                     ncols=1, size=800)
        return out

    # ---------------------------------------------------------------------------------- checking
    @staticmethod
    def verify(root: str, expect_head: str | None = None, expect_entries: int | None = None) -> dict:
        """Re-derive everything a line claims from the bytes beside it. See the module docstring for
        the list; `problems` names (seq, what) for every failure, and a v0 log (no floor_applied_nats,
        coefficients hashed in memory) fails here by design — that is the finding, not a bug."""
        log = os.path.join(root, "log.jsonl")
        es = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        by_seq = {e["seq"]: e for e in es}
        prev, bad = "0" * 64, []
        for e in es:
            body = {k: v for k, v in e.items() if k != "entry_hash"}
            if e.get("prev") != prev or _line_hash(prev, body) != e["entry_hash"]:
                bad.append((e["seq"], "chain"))
            fp_path = os.path.join(root, e["fingerprint"])
            if not os.path.exists(fp_path) or _sha_file(fp_path) != e["fingerprint_sha256"]:
                bad.append((e["seq"], "fingerprint bytes"))
            else:
                try:
                    fp = _load_fp_file(fp_path)
                except Exception as exc:
                    bad.append((e["seq"], f"fingerprint unreadable: {type(exc).__name__}"))
                    prev = e["entry_hash"]
                    continue
                if coefficients_sha256(coefficients(fp.rdm)) != e.get("coefficients_sha256"):
                    bad.append((e["seq"], "coefficients"))
                if "floor_applied_nats" not in e:
                    bad.append((e["seq"], "floor_applied missing (v0 line)"))
                floor_applied = e.get("floor_applied_nats", max(float(e.get("null_floor_nats", 0.0)), ck.RESOLUTION_NATS))
                for key in ("vs_baseline", "vs_previous"):
                    if key not in e:
                        continue
                    ref = by_seq.get(e[key]["seq"])
                    ref_path = os.path.join(root, ref["fingerprint"]) if ref else None
                    if ref_path is None or not os.path.exists(ref_path):
                        bad.append((e["seq"], f"{key} reference"))
                        continue
                    try:
                        want = _dist_body(e[key]["seq"], _load_fp_file(ref_path), fp, floor_applied)
                    except Exception as exc:
                        bad.append((e["seq"], f"{key} not re-derivable: {type(exc).__name__}"))
                        continue
                    if not all(_same(want[k], e[key].get(k)) for k in ("verdict", "mean_abs_nats", "ci", "rdm_r")):
                        bad.append((e["seq"], key))
            prev = e["entry_hash"]
        if expect_entries is not None and len(es) != expect_entries:
            bad.append((len(es), f"entries: expected {expect_entries} (a prefix of a chained log is a valid chain)"))
        if expect_head is not None and prev != expect_head:
            bad.append(("head", f"pinned head {expect_head[:12]} is not the file's head {prev[:12]}"))
        return {"entries": len(es), "head": prev, "ok": not bad, "problems": bad,
                "checks": ["chain", "fingerprint bytes", "coefficients from file", "verdicts from files",
                           "head pin" if expect_head else "head NOT pinned", "count pin" if expect_entries else "count NOT pinned"],
                "not_checked": ["plate bytes", "whether `when` is true (it is a label; `taken` is the clock)"]}

    @staticmethod
    def status(root: str) -> str:
        es = [json.loads(l) for l in open(os.path.join(root, "log.jsonl"), encoding="utf-8") if l.strip()]
        lines = ["`when` is a label the caller supplied; `taken` is the clock at observation.",
                 "",
                 "| # | when | taken | kind | floor measured | floor applied | vs baseline | vs previous | note |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for e in es:
            vb, vp = e.get("vs_baseline"), e.get("vs_previous")
            f = lambda d: f"{d['verdict']} {d['mean_abs_nats']:.4f} (r {d['rdm_r']:.3f})" if d else "—"
            fa = e.get("floor_applied_nats")
            lines.append(f"| {e['seq']} | {e['when'][:10]} | {e.get('taken', '—')} | {e['kind']} | {e['null_floor_nats']:.2e} | "
                         f"{(f'{fa:.2e}' if fa is not None else '— (v0)')} | {f(vb)} | {f(vp)} | {e.get('note') or e.get('reason') or ''} |")
        return "\n".join(lines)
