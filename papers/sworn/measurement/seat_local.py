# -*- coding: utf-8 -*-
"""The local family seat: Qwen2.5-7B-Instruct bf16 on CPU, 3B the design's fallback (SPEC §The seat
runners). ``--device cuda`` and ``--quant nf4`` are substrates the design does not name; a seat file
written under either carries ``"named_in_design": false``.

Greedy decoding, so seats 1/2/3 differ only by the committed rotation of the instruction blocks
(``common.block_order``), disclosed in the seat file and in any RESULT. The answer is the earliest
balanced JSON object in the generated text, else ``parsed: false``. ``--throughput-probe`` generates a
fixed number of tokens on a synthetic prompt and prints tokens per second and peak resident memory;
it writes no seat file. ``--smoke`` calls the model over a synthetic packet only.

CLI: ``python papers/sworn/measurement/seat_local.py --packet L --seat 1 [--model ID] [--dtype bf16]
[--device cpu] [--quant nf4] [--dry-run | --smoke --max-items 3 | --throughput-probe] [--dir DIR]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402
import seat_claude as SC                             # noqa: E402

FAMILY = "local"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
NAMED_IN_DESIGN = {(DEFAULT_MODEL, "bf16", "cpu", None), (FALLBACK_MODEL, "bf16", "cpu", None)}
THREADS = 22


class LocalSeat:
    def __init__(self, model_id: str = DEFAULT_MODEL, dtype: str = "bf16", device: str = "cpu",
                 quant: Optional[str] = None, threads: int = THREADS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.dtype, self.device, self.quant = model_id, dtype, device, quant
        torch.set_num_threads(threads)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {"device_map": device}
        if quant == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kw["torch_dtype"] = torch.bfloat16 if dtype == "bf16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        self.load_seconds = round(time.time() - t0, 1)
        self.torch = torch

    def substrate(self, seat: int) -> dict:
        return {"model": self.model_id, "transport": "transformers", "dtype": self.dtype, "device": self.device,
                "quant": self.quant, "named_in_design": (self.model_id, self.dtype, self.device, self.quant) in NAMED_IN_DESIGN,
                "block_order": C.block_order(seat), "threads": THREADS, "load_seconds": self.load_seconds}

    def chat(self, system: str, user: str, max_new_tokens: int = C.LOCAL_MAX_NEW_TOKENS) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        dt = time.time() - t0
        new = out[0][enc["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        return {"text": text, "prompt_tokens": int(enc["input_ids"].shape[1]), "new_tokens": int(new.shape[0]),
                "seconds": round(dt, 2)}


def _rss_gb() -> Optional[float]:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        return None


def throughput_probe(seat: LocalSeat, n_tokens: int = 100) -> dict:
    r = seat.chat("You are a synthetic probe. Continue the list.",
                  "List one hundred synthetic words, one per line, numbered.", max_new_tokens=n_tokens)
    return {"model": seat.model_id, "dtype": seat.dtype, "device": seat.device, "quant": seat.quant,
            "new_tokens": r["new_tokens"], "seconds": r["seconds"],
            "tok_per_s": round(r["new_tokens"] / r["seconds"], 2) if r["seconds"] else None,
            "prompt_tokens": r["prompt_tokens"], "load_seconds": seat.load_seconds, "peak_rss_gb": _rss_gb(),
            "note": "a throughput probe on a synthetic prompt; nothing here is a seat or a measurement"}


def run(panel: str, seat: int, meas_dir=None, model_id: str = DEFAULT_MODEL, dtype: str = "bf16",
        device: str = "cpu", quant: Optional[str] = None, dry_run: bool = False, smoke: bool = False,
        max_items: Optional[int] = None, root=None, local: Optional[LocalSeat] = None) -> dict:
    meas_dir = Path(meas_dir or HERE)
    root = Path(root or C.ROOT)
    packet_path = meas_dir / ("packet_%s.json" % panel)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    synthetic = SC.is_synthetic(meas_dir)
    if smoke and not synthetic:
        raise SystemExit("REFUSED: --smoke runs only over a synthetic packet")
    if not dry_run and not smoke:
        rel = (meas_dir / packet["key_digest_file"]).resolve()
        try:
            rel = rel.relative_to(root.resolve()).as_posix()
        except ValueError:
            raise SystemExit("REFUSED: %s is outside the repository" % rel)
        prereg = C.refuse_unless_prereg(False, [rel], root=root)
    else:
        prereg = None
    seat_dir = meas_dir / "seat_outputs" / FAMILY
    out_path = seat_dir / ("%s-seat%d.json" % (panel, seat))
    if out_path.exists():
        raise SystemExit("REFUSED: %s exists; a seat file is written once" % out_path)
    system = SC.system_for(panel, packet, seat, rotate=True)
    if dry_run:
        import synthetic as S
        substrate = {"model": model_id, "transport": "transformers", "dtype": dtype, "device": device, "quant": quant,
                     "named_in_design": (model_id, dtype, device, quant) in NAMED_IN_DESIGN,
                     "block_order": C.block_order(seat), "dry_run": True}
    else:
        local = local or LocalSeat(model_id, dtype, device, quant)
        substrate = local.substrate(seat)
    header = {"schema": SC.SEAT_SCHEMA, "family": FAMILY, "panel": panel, "seat": seat, "substrate": substrate,
              "packet_sha256": C.sha256_file(packet_path), "prereg": prereg, "dry_run": dry_run,
              "synthetic_packet": synthetic, "contamination_probe": None, "items": [], "unparsed": [],
              "errors": [], "timing": [], "verdict": None}
    if not substrate["named_in_design"]:
        header["note"] = "SUBSTRATE NOT NAMED IN DESIGN v2"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    items = packet["items"][:max_items] if max_items else packet["items"]
    for item in items:
        if dry_run:
            text = S.canned_answer(panel, item, FAMILY, seat)
            err = None
        else:
            try:
                r = local.chat(system, SC.prompt_for(panel, item))
                text, err = r["text"], None
                header["timing"].append({"id": item["id"], "prompt_tokens": r["prompt_tokens"],
                                         "new_tokens": r["new_tokens"], "seconds": r["seconds"]})
            except Exception as e:                   # the substrate failing is recorded, never hidden
                text, err = "", SC._classify(e)      # by TYPE: an exception's message is the library's
        raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        SC.ledger_append(seat_dir, {"item_id": item["id"], "panel": panel, "seat": seat, "raw_sha256": raw_sha,
                                    "ts": ts, "error": err, "dry_run": dry_run, "smoke": smoke})
        parsed, fields = SC.parse_answer(panel, text)
        row = {"id": item["id"], "raw_sha256": raw_sha, "parsed": parsed}
        row.update(fields)
        if not parsed:
            header["unparsed"].append(item["id"])
            row["text_head"] = text[:200]
        if err:
            header["errors"].append({"id": item["id"], "error": err})
        header["items"].append(row)
    header["verdict"] = "DRY-RUN" if dry_run else ("SMOKE-SYNTHETIC" if smoke else "RECORDED")
    C.write_json_lf(out_path, header)
    return header


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packet", choices=("L", "R"), default="L")
    ap.add_argument("--seat", type=int, default=1)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--quant", choices=("nf4",), default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--throughput-probe", action="store_true")
    ap.add_argument("--probe-tokens", type=int, default=100)
    a = ap.parse_args(argv)
    if a.throughput_probe:
        seat = LocalSeat(a.model, a.dtype, a.device, a.quant)
        print(json.dumps(throughput_probe(seat, a.probe_tokens), indent=1))
        return 0
    h = run(a.packet, a.seat, a.dir, a.model, a.dtype, a.device, a.quant, dry_run=a.dry_run, smoke=a.smoke,
            max_items=a.max_items)
    print("seat %s-%d (%s): verdict %s, items %d, unparsed %d, errors %d, substrate %s"
          % (a.packet, a.seat, FAMILY, h["verdict"], len(h["items"]), len(h["unparsed"]), len(h["errors"]),
             json.dumps(h["substrate"])))
    if h.get("timing"):
        secs = sum(t["seconds"] for t in h["timing"])
        toks = sum(t["new_tokens"] for t in h["timing"])
        print("timing: %d new tokens in %.1f s (%.2f tok/s), rss %s GB" % (toks, secs, toks / secs if secs else 0, _rss_gb()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
