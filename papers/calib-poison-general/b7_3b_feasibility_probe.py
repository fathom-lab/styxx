"""B7 feasibility probe -- does Qwen2.5-3B-Instruct FIT the arc's training regime on 8GB?

VRAM/throughput probe ONLY -- no scientific claim, no honesty data, no scored bars. Answers the
question two cycle logs and the strategy panel name as the binding flank: can the erasure arc's
exact training configuration (bf16 + LoRA r16 alpha32 on the seven projection modules, AdamW,
micro-batch 8 x accum 2, output_hidden_states=True erasure-style loss) run at 3B on this 8188 MiB
card -- plain, or with gradient checkpointing -- or does it OOM? A documented OOM is itself the
concrete case for compute (backlog cycle-36/37 "next").

Probes, in order (each isolated, VRAM stats reset between):
  1. audit-path forward: batch 8, output_hidden_states=True, no_grad (the resid_all pattern)
  2. training step, plain: forward+backward+step on an erasure-shaped loss
  3. training step, gradient checkpointing (only if 2 OOMs -- else recorded as skipped)

Content is synthetic fact-length filler (VRAM depends on shapes, not meaning).
Output: b7_3b_feasibility_result.json (feasibility receipt, NOT a RESULT doc).
Usage: python papers/calib-poison-general/b7_3b_feasibility_probe.py
"""
from __future__ import annotations
import json, gc, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
STEPS = 8
MICRO_BATCH = 8
ACCUM = 2
LR = 1e-4
FILLER = [
    f"The capital city of region number {i} is named after the river that crosses it, "
    f"and the local museum was founded in the nineteenth century. This statement is true."
    for i in range(64)
]


def mib(x: int) -> float:
    return round(x / (1024 * 1024), 1)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    out = {"what": "B7 feasibility/VRAM probe of Qwen2.5-3B-Instruct under the erasure arc's exact training regime on 8GB -- no scientific claim",
           "model": MODEL_3B, "gpu_total_mib": None, "probes": {}}
    props = torch.cuda.get_device_properties(0)
    out["gpu_total_mib"] = mib(props.total_memory)
    out["gpu_name"] = props.name

    tok = AutoTokenizer.from_pretrained(MODEL_3B)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def fresh_stats():
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    def record(key, ok, err=None, extra=None):
        d = {"ok": ok,
             "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
             "peak_reserved_mib": mib(torch.cuda.max_memory_reserved())}
        if err:
            d["error"] = err
        if extra:
            d.update(extra)
        out["probes"][key] = d
        print(f"[{key}] ok={ok} peak_alloc={d['peak_allocated_mib']}MiB peak_res={d['peak_reserved_mib']}MiB"
              + (f" err={err[:120]}" if err else "") + (f" {extra}" if extra else ""), flush=True)

    # ---- load (three strategies: the arc default, then two Windows-pagefile mitigations) ----
    model = None
    for key, kwargs, post in [
        ("load_bf16", dict(dtype=torch.bfloat16, device_map="cuda"), None),
        ("load_bf16_low_cpu", dict(dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True), None),
        ("load_bf16_cpu_then_cuda", dict(dtype=torch.bfloat16, low_cpu_mem_usage=True), "cuda"),
    ]:
        fresh_stats()
        try:
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(MODEL_3B, **kwargs)
            if post:
                model = model.to(post)
            model.eval()
            record(key, True, extra={"load_seconds": round(time.time() - t0, 1),
                                     "n_layers": model.config.num_hidden_layers,
                                     "hidden_size": model.config.hidden_size})
            out["load_strategy"] = key
            break
        except Exception as e:
            record(key, False, err=f"{type(e).__name__}: {e}")
            model = None
            gc.collect(); torch.cuda.empty_cache()
    if model is None:
        out["feasibility"] = "LOAD_FAILED_HOST_SIDE"
        (HERE / "b7_3b_feasibility_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print("\nB7 FEASIBILITY: LOAD FAILED (host-side; GPU never saturated) -- see result JSON", flush=True)
        return 0

    enc = tok(FILLER[:MICRO_BATCH], return_tensors="pt", padding=True)
    ids = enc.input_ids.to("cuda"); attn = enc.attention_mask.to("cuda")

    # ---- probe 1: audit-path forward (resid_all pattern) ----
    fresh_stats()
    try:
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        n_hs = len(o.hidden_states)
        del o
        record("audit_forward_hidden_states", True, extra={"n_hidden_states": n_hs})
    except torch.cuda.OutOfMemoryError as e:
        record("audit_forward_hidden_states", False, err=f"OOM: {e}")
    except Exception as e:
        record("audit_forward_hidden_states", False, err=f"{type(e).__name__}: {e}")

    # ---- LoRA wrap (the arc's exact adapter) ----
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)

    def train_probe(key, checkpointing):
        fresh_stats()
        try:
            if checkpointing:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            model.train()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
            times = []
            for step in range(STEPS):
                t0 = time.time()
                opt.zero_grad()
                for _ in range(ACCUM):
                    o = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
                    last = attn.sum(1) - 1
                    l = 0.0
                    # erasure-shaped loss: squared projection of final-token states at six layers
                    n_layers = len(o.hidden_states) - 1
                    scan = [int(n_layers * f) for f in (0.42, 0.5, 0.57, 0.64, 0.71, 0.78)]
                    for L in scan:
                        h = o.hidden_states[L]
                        hlast = h[torch.arange(h.shape[0]), last, :].float()
                        l = l + hlast.pow(2).mean() * 1e-6
                    (l / ACCUM).backward()
                    del o
                opt.step()
                times.append(time.time() - t0)
            del opt
            record(key, True, extra={"steps": STEPS, "sec_per_step": round(sum(times) / len(times), 2),
                                     "est_300_step_minutes": round(sum(times) / len(times) * 300 / 60, 1)})
            return True
        except torch.cuda.OutOfMemoryError as e:
            record(key, False, err=f"OOM: {str(e)[:200]}")
            return False
        except Exception as e:
            record(key, False, err=f"{type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[:300]}")
            return False
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()

    ok_plain = train_probe("train_step_plain", checkpointing=False)
    if not ok_plain:
        train_probe("train_step_grad_checkpointing", checkpointing=True)
    else:
        out["probes"]["train_step_grad_checkpointing"] = {"ok": None, "skipped": "plain training fit; checkpointing not needed"}
        print("[train_step_grad_checkpointing] skipped -- plain fit", flush=True)

    plain = out["probes"].get("train_step_plain", {})
    ckpt = out["probes"].get("train_step_grad_checkpointing", {})
    if plain.get("ok"):
        out["feasibility"] = "FITS_PLAIN"
    elif ckpt.get("ok"):
        out["feasibility"] = "FITS_WITH_GRADIENT_CHECKPOINTING"
    else:
        out["feasibility"] = "OOM_DOCUMENTED"
    (HERE / "b7_3b_feasibility_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nB7 FEASIBILITY: {out['feasibility']}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
