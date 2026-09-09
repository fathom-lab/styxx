"""Re-run the published first-verdict battery and compare per-item digests against the cert.

WHAT THIS IS. The published verdict at ``papers/v8/first_verdict_2026_09_09`` has never been
re-run. This script performs the first re-run: same box, same session, same operator, same local
snapshot. It is NOT an independent party and establishes nothing about whether anyone else can
reproduce the verdict. What it can establish is (a) whether the mechanism runs end to end from
the published bytes alone, (b) whether the reference run's per-item output digests come back, and
(c) what a re-run COSTS in wall-clock seconds, which is the number that decides whether a
challenge is affordable and which has never been measured.

WHERE THE INPUTS COME FROM. Everything is read out of the published artifact, nothing is retyped:

* the 64 battery items -- from log entry 0 (the battery pool cert), which carries ``prompt_text``
  per item, so the prompts are recoverable from the published log alone;
* the run order -- from the canonical fingerprint's own ``body.items`` order, which section 3.1
  of the spec fixes as the RUN order (and which is checked here against A.3 order and against the
  cert's own ``nuisance.item_order_sha256``);
* the recipe -- from the canonical cert's own signed ``recipe`` block (decoding + materials), not
  from ``recipe.json`` beside it, so the chat template and decoding this script uses are the ones
  inside the signature;
* the subject -- from the canonical cert's ``subject`` block, and the four Appendix A.2 hashes are
  recomputed here from the snapshot on disk and compared, so a different set of weights cannot be
  silently re-run under the cert's name.

HOW IT RUNS. ``styxx.v8.runner_hf.TransformersRunner`` is imported and called unmodified. The
canonical run is ``batch_size=1``, so this script calls ``runner.run([item], recipe, subject)``
once per item: at batch 1 the runner's own loop already chunks one item per forward pass, seeds
per chunk, and pads a single sequence to its own length, so 64 single-item calls issue the same
forward passes as one 64-item call at batch 1. Doing it per item is what makes a PER-ITEM
wall-clock number available at all.

``logits_to_keep`` is NOT passed. It is the documented mitigation for materialising a full
``(batch, seq, vocab)`` logits tensor -- about 6.5 GB at this model's 256k vocab -- but that is a
batch-size problem and the reference run is batch 1. If this script had needed it, relying on it
would have required re-verifying bitwise identity on one item first. It did not, so no such claim
is made or needed here.

Output: ``rerun_raw.json`` (every per-item comparison, timings, environment) beside this file.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path(r"C:\Users\heyzo\clawd\wt\v8")
ART = WT / "papers" / "v8" / "first_verdict_2026_09_09"
SNAPSHOT = r"C:\Users\heyzo\.cache\huggingface\hub\models--google--gemma-2-2b-it\snapshots\299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"

sys.path.insert(0, str(WT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nvidia_driver() -> str:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    return lines[0] if (p.returncode == 0 and lines) else "unknown"


def main() -> int:
    report: dict = {"kind": "rerun", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- inputs, all out of the published artifact ---------------------------------
    canon_path = next((ART / "fp_bf16").glob("fingerprint-canonical-*.json"))
    canon = load_json(canon_path)
    battery = load_json(ART / "log" / "entries" / "000000" / "00000000.json")
    report["inputs"] = {
        "canonical_cert": canon_path.name,
        "canonical_cert_id": canon["id"],
        "battery_cert_id": battery["id"],
        "battery_size": len(battery["body"]["items"]),
        "snapshot": SNAPSHOT,
    }

    prompts = {it["item_id"]: it for it in battery["body"]["items"]}
    cert_items = canon["body"]["items"]
    run_order = [rec["item_id"] for rec in cert_items]
    by_id = {rec["item_id"]: rec for rec in cert_items}
    recipe = canon["recipe"]
    subject = canon["subject"]
    nuisance = canon["body"]["nuisance"]

    from styxx.v8.fingerprint import _id_key, item_record
    from styxx.v8.jcs import canonical_bytes, sha256_hex

    a3 = sorted(run_order, key=_id_key)
    order_sha = sha256_hex(canonical_bytes(run_order))
    report["order_checks"] = {
        "cert_run_order_is_a3": run_order == a3,
        "recomputed_item_order_sha256": order_sha,
        "cert_item_order_sha256": nuisance.get("item_order_sha256"),
        "item_order_sha256_matches": order_sha == nuisance.get("item_order_sha256"),
        "cert_nuisance": {k: v for k, v in nuisance.items() if k != "item_order_sha256"},
        "recipe_batch_size": recipe["decoding"]["batch_size"],
        "recipe_max_new_tokens": recipe["decoding"]["max_new_tokens"],
        "recipe_seed": recipe["decoding"]["seed"],
        "recipe_padding_side": recipe["decoding"]["padding_side"],
        "recipe_temperature": recipe["decoding"]["temperature"],
    }
    missing = [i for i in run_order if i not in prompts]
    if missing:
        report["FATAL"] = f"{len(missing)} run-order ids absent from the battery cert"
        (HERE / "rerun_raw.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        return 1

    items = [
        {"item_id": i, "prompt_text": prompts[i]["prompt_text"], "role": prompts[i].get("role", "item")}
        for i in run_order
    ]

    # ---- the runner, unmodified ------------------------------------------------------
    from styxx.v8.runner_hf import TransformersRunner, snapshot_hashes

    t_hash = time.perf_counter()
    observed = snapshot_hashes(SNAPSHOT)
    report["subject_check"] = {
        "hash_seconds": round(time.perf_counter() - t_hash, 2),
        "fields": {
            k: {"cert": subject.get(k), "observed": observed[k], "same": subject.get(k) == observed[k]}
            for k in ("weights_sha256", "config_sha256", "generation_config_sha256", "tokenizer_sha256")
        },
    }
    report["subject_check"]["all_same"] = all(
        v["same"] for v in report["subject_check"]["fields"].values()
    )

    runner = TransformersRunner(SNAPSHOT, dtype="bfloat16", device="cuda")

    t_load = time.perf_counter()
    runner._load()
    load_seconds = time.perf_counter() - t_load
    report["load_seconds"] = round(load_seconds, 2)
    report["runner_subject"] = runner.subject(subject)
    report["runner_environment"] = runner.environment()
    report["cert_environment"] = subject.get("environment")
    report["driver_now"] = nvidia_driver()
    report["python"] = platform.python_version()
    try:
        import torch
        import transformers
        report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception as exc:  # pragma: no cover - reporting only
        report["versions"] = {"error": repr(exc)}

    # ---- the re-run ------------------------------------------------------------------
    try:
        import torch as _t
        _t.cuda.reset_peak_memory_stats()
        after_load_bytes = _t.cuda.memory_allocated()
    except Exception:
        _t, after_load_bytes = None, None
    per_item = []
    rerun_records = []
    t_all = time.perf_counter()
    for k, item in enumerate(items):
        t0 = time.perf_counter()
        got = runner.run([item], recipe, subject)[0]
        dt = time.perf_counter() - t0
        rec = item_record(got, redacted=False)
        rerun_records.append(rec)
        ref = by_id[item["item_id"]]
        cmp = {
            "position": k,
            "item_id": item["item_id"],
            "seconds": round(dt, 3),
            "output_sha256_reproduces": rec["output_sha256"] == ref["output_sha256"],
            "token_ids_sha256_reproduces": rec["token_ids_sha256"] == ref["token_ids_sha256"],
            "n_generated_reproduces": rec["n_generated"] == ref["n_generated"],
            "seq_logprob_bitwise": rec["seq_logprob"] == ref["seq_logprob"],
            "topk_bitwise": rec["topk"] == ref["topk"],
        }
        if not cmp["output_sha256_reproduces"]:
            cmp["cert_output_sha256"] = ref["output_sha256"]
            cmp["rerun_output_sha256"] = rec["output_sha256"]
            cmp["cert_output_text"] = ref.get("output_text")
            cmp["rerun_output_text"] = rec.get("output_text")
            cmp["cert_token_ids"] = ref["token_ids"]
            cmp["rerun_token_ids"] = rec["token_ids"]
        if not cmp["seq_logprob_bitwise"]:
            cmp["cert_seq_logprob"] = ref["seq_logprob"]
            cmp["rerun_seq_logprob"] = rec["seq_logprob"]
        if not cmp["topk_bitwise"]:
            cmp["topk_first_diff"] = _first_topk_diff(ref["topk"], rec["topk"])
        per_item.append(cmp)
        mark = "." if cmp["output_sha256_reproduces"] else "X"
        print(f"{mark} {k:2d}/64 {item['item_id']} {dt:6.2f}s", flush=True)
    generate_seconds = time.perf_counter() - t_all

    # ---- phase 2: the same 64 items in ONE runner call --------------------------------
    # The per-item loop above is what makes a per-item number exist, but it pays the runner's
    # per-call setup 64 times. This second pass is the shape a real challenger would run -- one
    # call, batch 1 -- and it is also the control for the claim that the two are the same
    # computation: the digests are compared against the same certificate.
    t_batch = time.perf_counter()
    whole = runner.run(items, recipe, subject)
    whole_seconds = time.perf_counter() - t_batch
    whole_records = [item_record(r, redacted=False) for r in whole]
    whole_ok = sum(
        1 for rec in whole_records if rec["output_sha256"] == by_id[rec["item_id"]]["output_sha256"]
    )
    whole_bitwise = sum(
        1 for rec in whole_records
        if rec["token_ids"] == by_id[rec["item_id"]]["token_ids"]
        and rec["seq_logprob"] == by_id[rec["item_id"]]["seq_logprob"]
        and rec["topk"] == by_id[rec["item_id"]]["topk"]
    )
    report["one_call_pass"] = {
        "seconds": round(whole_seconds, 2),
        "seconds_per_item": round(whole_seconds / len(items), 3),
        "output_sha256_reproduced": whole_ok,
        "fully_bitwise": whole_bitwise,
        "differing_item_ids": [
            rec["item_id"] for rec in whole_records
            if rec["output_sha256"] != by_id[rec["item_id"]]["output_sha256"]
        ],
        "agrees_with_per_item_loop": whole_records == rerun_records,
    }

    # ---- the exact channel over the whole re-run --------------------------------------
    # Appendix A.3: one hash over all 64 item records. A single-item difference moves it, so
    # this is the whole-battery statement the per-item table is the breakdown of.
    from styxx.v8.fingerprint import exact_hash

    report["exact_channel"] = {
        "cert": canon["body"]["channels"]["exact"]["hash"],
        "rerun": exact_hash(rerun_records),
    }
    report["exact_channel"]["same"] = (
        report["exact_channel"]["cert"] == report["exact_channel"]["rerun"]
    )

    ok = sum(1 for c in per_item if c["output_sha256_reproduces"])
    secs = [c["seconds"] for c in per_item]
    report["result"] = {
        "items": len(per_item),
        "output_sha256_reproduced": ok,
        "output_sha256_differed": len(per_item) - ok,
        "differing_item_ids": [c["item_id"] for c in per_item if not c["output_sha256_reproduces"]],
        "token_ids_sha256_reproduced": sum(1 for c in per_item if c["token_ids_sha256_reproduces"]),
        "seq_logprob_bitwise": sum(1 for c in per_item if c["seq_logprob_bitwise"]),
        "topk_bitwise": sum(1 for c in per_item if c["topk_bitwise"]),
    }
    report["timing"] = {
        "model_load_seconds": round(load_seconds, 2),
        "snapshot_hash_seconds": report["subject_check"]["hash_seconds"],
        "generate_seconds_64_items": round(generate_seconds, 2),
        "generate_plus_load_seconds": round(generate_seconds + load_seconds, 2),
        "per_item_mean_seconds": round(sum(secs) / len(secs), 3),
        "per_item_min_seconds": round(min(secs), 3),
        "per_item_max_seconds": round(max(secs), 3),
        "per_item_median_seconds": round(sorted(secs)[len(secs) // 2], 3),
        "first_item_seconds": secs[0],
        "per_item_mean_excluding_first": round(sum(secs[1:]) / len(secs[1:]), 3),
    }
    if _t is not None:
        # What a challenger's card has to hold. `logits_to_keep` was not passed; at batch 1 the
        # full (1, seq, 256k) logits tensor is small enough that it never had to be.
        report["gpu_memory"] = {
            "weights_mib_after_load": round(after_load_bytes / (1 << 20), 1),
            "peak_allocated_mib": round(_t.cuda.max_memory_allocated() / (1 << 20), 1),
            "peak_reserved_mib": round(_t.cuda.max_memory_reserved() / (1 << 20), 1),
            "logits_to_keep_passed": False,
        }
    report["per_item"] = per_item
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (HERE / "rerun_raw.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")

    print("")
    print(f"output_sha256 reproduced : {ok}/{len(per_item)}")
    print(f"token_ids_sha256         : {report['result']['token_ids_sha256_reproduced']}/{len(per_item)}")
    print(f"seq_logprob bitwise      : {report['result']['seq_logprob_bitwise']}/{len(per_item)}")
    print(f"topk bitwise             : {report['result']['topk_bitwise']}/{len(per_item)}")
    print(f"differing                : {report['result']['differing_item_ids']}")
    print(f"exact channel hash same  : {report['exact_channel']['same']}")
    print(f"  cert  {report['exact_channel']['cert']}")
    print(f"  rerun {report['exact_channel']['rerun']}")
    print(f"generate 64 items        : {generate_seconds:.1f}s  ({generate_seconds/64:.2f}s/item)"
          f"  [64 separate runner calls]")
    op = report["one_call_pass"]
    print(f"one runner call, 64 items: {op['seconds']}s  ({op['seconds_per_item']}s/item)"
          f"  reproduced {op['output_sha256_reproduced']}/64"
          f"  identical to per-item loop: {op['agrees_with_per_item_loop']}")
    print(f"model load               : {load_seconds:.1f}s")
    print(f"snapshot hashing         : {report['subject_check']['hash_seconds']}s")
    if "gpu_memory" in report:
        g = report["gpu_memory"]
        print(f"gpu after load / peak    : {g['weights_mib_after_load']} MiB / "
              f"{g['peak_allocated_mib']} MiB allocated, {g['peak_reserved_mib']} MiB reserved")
    return 0


def _first_topk_diff(ref, got):
    if not isinstance(ref, list) or not isinstance(got, list):
        return {"shape": [type(ref).__name__, type(got).__name__]}
    if len(ref) != len(got):
        return {"len": [len(ref), len(got)]}
    for a, b in zip(ref, got):
        if a != b:
            return {"pos": a.get("pos"), "cert": a, "rerun": b}
    return None


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # SystemExit/KeyboardInterrupt are not failures of the re-run
        import traceback
        (HERE / "FAILED_rerun.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        print(traceback.format_exc())
        code = 1
    sys.exit(code)
