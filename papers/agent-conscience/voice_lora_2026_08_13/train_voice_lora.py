#!/usr/bin/env python
"""
train_voice_lora.py -- QLoRA voice tune of Qwen2.5-7B-Instruct on voice_dataset.jsonl.

Prereg: C:/Users/heyzo/clawd/styxx/papers/agent-conscience/PREREG_voice_lora_honesty_2026_08_11.md
This script is the VOICE-arm training apparatus only. It takes no measurements.

Design decisions (recorded here on purpose):
  * SYSTEM PROMPT EXCLUDED from training text. The Qwen2.5 chat template silently
    injects a default system block ("You are Qwen, created by Alibaba Cloud...")
    whenever no system message is present, so we do NOT use apply_chat_template.
    We build the ChatML string by hand: <|im_start|>user ... <|im_end|> then
    <|im_start|>assistant ... <|im_end|>. No system block at all. Voice lives in
    the replies; at serve time darkflobi's own prefill supplies the system text,
    so the LoRA must not bind the voice to any particular system string.
  * LOSS ONLY ON REPLY TOKENS. User-turn tokens and the assistant header are
    label-masked (-100). The model learns to *produce* the register, not to
    model the operator's prompts.
  * Batch 1 (no padding needed), grad-accum 16 -> effective batch 16.
  * VRAM guard: refuses to start unless >= 7000 MiB free. darkflobi-fast holds
    ~5 GiB while serving; the operator must stop it first (pm2 stop
    darkflobi-fast). This script NEVER touches pm2 itself.

Launch:
  python train_voice_lora.py
  (optional) python train_voice_lora.py --smoke   # 8 examples, 2 steps, plumbing only
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "voice_dataset.jsonl"
OUT_DIR = HERE / "voice_lora"
ADAPTER_DIR = OUT_DIR / "adapter"
RECEIPT = OUT_DIR / "TRAIN_RECEIPT.json"
LOSS_LOG = OUT_DIR / "loss_log.jsonl"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

HP = {
    "quant": "nf4_double_quant_bf16_compute",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "max_seq_len": 1024,
    "per_device_batch": 1,
    "grad_accum": 16,
    "epochs": 2,
    "lr": 2e-4,
    "lr_schedule": "cosine",
    "warmup_ratio": 0.03,
    "optim": "paged_adamw_8bit",
    "seed": 42,
    "system_prompt_in_training_text": False,
    "loss_on": "assistant_reply_tokens_plus_im_end_only",
}

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_no_bom(path: Path, obj) -> None:
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)  # bytes write == no BOM, ever
    back = path.read_bytes()
    if len(back) != len(data) or back[:1] == b"\xef":
        raise RuntimeError(f"write verification failed for {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="8 examples / 2 optimizer steps; plumbing check only. "
                         "Per prereg, any smoke output is INVALID for the verdict.")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available.", file=sys.stderr)
        return 2

    free_b, total_b = torch.cuda.mem_get_info()
    free_mib = free_b // (1 << 20)
    if free_mib < 7000:
        print(f"FATAL: only {free_mib} MiB VRAM free (need >= 7000). "
              f"Stop the serving process first: pm2 stop darkflobi-fast "
              f"(operator action -- this script does not touch pm2).",
              file=sys.stderr)
        return 3

    if not DATASET.exists():
        print(f"FATAL: dataset missing: {DATASET}", file=sys.stderr)
        return 4

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(HP["seed"])

    dataset_sha = sha256_file(DATASET)
    started = utc_now()
    t0 = time.time()

    # ---- data -------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

    records = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    n_pairs = len(records)
    if args.smoke:
        records = records[:8]

    examples, truncated = [], 0
    for rec in records:
        msgs = rec["messages"]
        assert len(msgs) == 2 and msgs[0]["role"] == "user" \
            and msgs[1]["role"] == "assistant", f"bad record shape: {msgs}"
        u, a = msgs[0]["content"], msgs[1]["content"]
        # ChatML by hand -- NO system block (see module docstring).
        prompt_text = f"{IM_START}user\n{u}{IM_END}\n{IM_START}assistant\n"
        ids_p = tok(prompt_text, add_special_tokens=False)["input_ids"]
        ids_r = tok(a + IM_END, add_special_tokens=False)["input_ids"]
        input_ids = ids_p + ids_r
        labels = [-100] * len(ids_p) + list(ids_r)
        if len(input_ids) > HP["max_seq_len"]:
            input_ids = input_ids[: HP["max_seq_len"]]
            labels = labels[: HP["max_seq_len"]]
            truncated += 1
        examples.append({"input_ids": input_ids, "labels": labels,
                         "attention_mask": [1] * len(input_ids)})

    lens = sorted(len(e["input_ids"]) for e in examples)
    print(f"examples={len(examples)} truncated={truncated} "
          f"tok_p50={lens[len(lens)//2]} tok_max={lens[-1]}")

    def collate(feats):
        assert len(feats) == 1  # batch 1 -> no padding path exists on purpose
        f = feats[0]
        return {k: torch.tensor([f[k]], dtype=torch.long) for k in f}

    # ---- model ------------------------------------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lcfg = LoraConfig(
        r=HP["lora_r"], lora_alpha=HP["lora_alpha"],
        lora_dropout=HP["lora_dropout"], bias="none",
        task_type="CAUSAL_LM", target_modules=HP["target_modules"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # ---- train ------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        per_device_train_batch_size=HP["per_device_batch"],
        gradient_accumulation_steps=HP["grad_accum"],
        num_train_epochs=HP["epochs"] if not args.smoke else 1,
        max_steps=2 if args.smoke else -1,
        learning_rate=HP["lr"],
        lr_scheduler_type=HP["lr_schedule"],
        warmup_ratio=HP["warmup_ratio"],
        optim=HP["optim"],
        bf16=True,
        logging_steps=5,
        save_strategy="no",          # single final adapter save below
        report_to=[],
        seed=HP["seed"],
        dataloader_num_workers=0,    # Windows
        max_grad_norm=1.0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=examples,
                      data_collator=collate)

    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()

    # ---- artifacts --------------------------------------------------------
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        print("FATAL: adapter_model.safetensors not written", file=sys.stderr)
        return 5

    with open(LOSS_LOG, "wb") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")

    losses = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    receipt = {
        "prereg": "C:/Users/heyzo/clawd/styxx/papers/agent-conscience/"
                  "PREREG_voice_lora_honesty_2026_08_11.md",
        "smoke": bool(args.smoke),
        "smoke_note": "smoke output is INVALID for any verdict (prereg smoke_verdict)"
                      if args.smoke else None,
        "base_model": MODEL_ID,
        "dataset_path": str(DATASET),
        "dataset_sha256": dataset_sha,
        "dataset_pairs_total": n_pairs,
        "examples_trained": len(examples),
        "examples_truncated_at_seq_len": truncated,
        "hyperparams": HP,
        "started_utc": started,
        "ended_utc": utc_now(),
        "wall_seconds": round(time.time() - t0, 1),
        "train_loss_mean": result.metrics.get("train_loss"),
        "final_logged_loss": losses[-1] if losses else None,
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() // (1 << 20),
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() // (1 << 20),
        "adapter_dir": str(ADAPTER_DIR),
        "versions": _versions(),
    }
    write_json_no_bom(RECEIPT, receipt)
    print(f"receipt -> {RECEIPT}")
    print(f"adapter -> {ADAPTER_DIR}")
    return 0


def _versions():
    import bitsandbytes, peft, torch, transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__}


if __name__ == "__main__":
    sys.exit(main())
