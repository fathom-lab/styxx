# -*- coding: utf-8 -*-
"""styxx.observatory — a model, watched.

    obs = Observatory("observatory/qwen2.5-1.5b", model_id="Qwen/Qwen2.5-1.5B")
    obs.observe(probe, n_null=2)          # one observation: floor, fingerprint, distance to the
                                          # baseline and to the previous observation, plates, a
                                          # log line chained to the last one
    Observatory.verify(root)              # every line's hash re-derived; every file's hash checked
    Observatory.status(root)              # a table for humans

Policy, fixed: the BASELINE is the first accepted observation and stays the baseline until a
human calls `rebaseline(reason)` — drift is never absorbed silently. Each observation reports
the distance to the baseline (has the model moved since we started watching?) and to the
previous observation (did it move since yesterday?), each against the floor measured that day
from n_null repeated fingerprints. Verdicts come from styxx.checksum and nothing here re-grades
them.

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


def _sha_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _line_hash(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


class Observatory:
    def __init__(self, root: str, model_id: str, canaries=ck.CANARIES, tokenizer_id: str = ""):
        self.root, self.model_id, self.canaries, self.tokenizer_id = root, model_id, canaries, tokenizer_id
        os.makedirs(os.path.join(root, "fingerprints"), exist_ok=True)
        os.makedirs(os.path.join(root, "plates"), exist_ok=True)
        self.log = os.path.join(root, "log.jsonl")

    # ----------------------------------------------------------------------------------- reading
    def entries(self) -> list[dict]:
        if not os.path.exists(self.log):
            return []
        return [json.loads(l) for l in open(self.log, encoding="utf-8") if l.strip()]

    def _load_fp(self, entry: dict) -> ck.Fingerprint:
        d = json.load(open(os.path.join(self.root, entry["fingerprint"]), encoding="utf-8"))
        return ck.Fingerprint(model_id=d["model_id"], canary_sha256=d["canary_sha256"], tokenizer_id=d["tokenizer_id"],
                              ids=d["ids"], mean_lp=np.asarray(d["mean_lp"]), rdm=np.asarray(d["rdm"]),
                              n_tokens=d["n_tokens"], created=d["created"])

    def baseline(self) -> dict | None:
        es = self.entries()
        marks = [e for e in es if e.get("kind") in ("baseline", "rebaseline")]
        return marks[-1] if marks else None

    # ----------------------------------------------------------------------------------- writing
    def observe(self, probe: Callable, n_null: int = 2, when: str | None = None, note: str = "") -> dict:
        when = when or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fps = [ck.fingerprint(probe, self.model_id, self.canaries, tokenizer_id=self.tokenizer_id) for _ in range(max(2, n_null))]
        floor = ck.null_floor(fps)
        fp = fps[0]
        es = self.entries()
        seq = len(es) + 1
        tag = f"{seq:04d}_{when[:10]}"
        fp_rel = os.path.join("fingerprints", f"{tag}.json")
        json.dump(fp.to_json(), open(os.path.join(self.root, fp_rel), "w"), separators=(",", ":"))
        W = coefficients(fp.rdm)
        body = {"seq": seq, "when": when, "model_id": self.model_id, "canary_sha256": fp.canary_sha256,
                "fingerprint": fp_rel, "fingerprint_sha256": _sha_file(os.path.join(self.root, fp_rel)),
                "coefficients_sha256": coefficients_sha256(W), "null_floor_nats": floor, "n_null": len(fps), "note": note}
        base = self.baseline()
        if base is None:
            body["kind"] = "baseline"
        else:
            body["kind"] = "observation"
            d_base = ck.distance(self._load_fp(base), fp, floor_nats=floor)
            body["vs_baseline"] = {"seq": base["seq"], "verdict": d_base.verdict, "mean_abs_nats": d_base.mean_abs_nats,
                                   "ci": list(d_base.ci_mean_abs), "rdm_r": d_base.rdm_r}
            prev = es[-1]
            d_prev = ck.distance(self._load_fp(prev), fp, floor_nats=floor)
            body["vs_previous"] = {"seq": prev["seq"], "verdict": d_prev.verdict, "mean_abs_nats": d_prev.mean_abs_nats,
                                   "ci": list(d_prev.ci_mean_abs), "rdm_r": d_prev.rdm_r}
        prev_hash = es[-1]["entry_hash"] if es else "0" * 64
        body["prev"] = prev_hash
        body["entry_hash"] = _line_hash(prev_hash, {k: v for k, v in body.items() if k != "prev"} | {"prev": prev_hash})
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
        return body

    def rebaseline(self, reason: str) -> dict:
        es = self.entries()
        if not es:
            raise ValueError("nothing to rebaseline")
        last = es[-1]
        body = {k: last[k] for k in ("seq", "when", "model_id", "canary_sha256", "fingerprint", "fingerprint_sha256",
                                      "coefficients_sha256", "null_floor_nats", "n_null")}
        body.update({"seq": len(es) + 1, "kind": "rebaseline", "reason": reason,
                     "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        prev_hash = last["entry_hash"]
        body["prev"] = prev_hash
        body["entry_hash"] = _line_hash(prev_hash, {k: v for k, v in body.items() if k != "prev"} | {"prev": prev_hash})
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
        return body

    def render(self) -> str | None:
        """Plates for the latest observation: its geometry, and the drift plate against the baseline."""
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
        render_grid([(f"baseline #{base['seq']} ({base['when'][:10]})", "the model when watching began", Wb),
                     (f"observation #{last['seq']} ({last['when'][:10]})", sub, Wl)],
                    out, title=f"observatory: {self.model_id}", ncols=2)
        render_drift([(f"#{base['seq']} → #{last['seq']}", sub, Wb, Wl)], out.replace(".png", "_drift.png"),
                     title="drift plate: sand only where the model moved since the baseline", ncols=1, size=800)
        return out

    # ---------------------------------------------------------------------------------- checking
    @staticmethod
    def verify(root: str) -> dict:
        log = os.path.join(root, "log.jsonl")
        es = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        prev, bad = "0" * 64, []
        for e in es:
            body = {k: v for k, v in e.items() if k != "entry_hash"}
            if e.get("prev") != prev or _line_hash(prev, body) != e["entry_hash"]:
                bad.append((e["seq"], "chain"))
            fp = os.path.join(root, e["fingerprint"])
            if not os.path.exists(fp) or _sha_file(fp) != e["fingerprint_sha256"]:
                bad.append((e["seq"], "fingerprint bytes"))
            prev = e["entry_hash"]
        return {"entries": len(es), "head": prev, "ok": not bad, "problems": bad}

    @staticmethod
    def status(root: str) -> str:
        es = [json.loads(l) for l in open(os.path.join(root, "log.jsonl"), encoding="utf-8") if l.strip()]
        lines = ["| # | when | kind | floor | vs baseline | vs previous |", "|---|---|---|---|---|---|"]
        for e in es:
            vb, vp = e.get("vs_baseline"), e.get("vs_previous")
            f = lambda d: f"{d['verdict']} {d['mean_abs_nats']:.4f} (r {d['rdm_r']:.3f})" if d else "—"
            lines.append(f"| {e['seq']} | {e['when'][:10]} | {e['kind']} | {e['null_floor_nats']:.2e} | {f(vb)} | {f(vp)} |")
        return "\n".join(lines)
