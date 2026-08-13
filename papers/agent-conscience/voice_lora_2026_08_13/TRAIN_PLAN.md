# TRAIN_PLAN — voice LoRA, Qwen2.5-7B-Instruct, RTX 4070 Laptop 8GB

apparatus plan only. prereg is binding:
`C:\Users\heyzo\clawd\styxx\papers\agent-conscience\PREREG_voice_lora_honesty_2026_08_11.md`
this document licenses no verdict. smoke runs are INVALID per prereg.

## readiness audit (all measured 2026-08-13, receipts inline)

| check | result |
|---|---|
| torch | 2.5.1+cu121, `torch.cuda.is_available()` = True |
| GPU | RTX 4070 Laptop, 8188 MiB, compute capability 8.9 |
| base weights (HF) | ON DISK: `C:\Users\heyzo\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28` (15 GB, 4 safetensors shards + tokenizer — no download needed) |
| C: free space | 11.7 GiB free (no 15 GB download needed, adapter+checkpoints ~0.5 GiB — fits) |
| bitsandbytes | 0.49.2, Windows-native, `python -m bitsandbytes` self-test: **SUCCESS** (CUDA callable) |
| transformers / peft / accelerate / datasets | 4.57.3 / 0.18.1 / 1.13.0 / 4.8.4 |
| trl | NOT installed — deliberately not used; plain `transformers.Trainer` with explicit label masking is more auditable and avoids new dependency risk on a live box |
| dataset | `voice_dataset.jsonl` — 1,215 records, all exactly [user, assistant], 0 malformed; sha256 `584392022129c163313eb5cd7ef30a249d269def605899a7ec4522d8342a0ec1` |
| dataset token lengths (real tokenizer) | p50 **212**, p95 **293**, max **526** tokens; **0 of 1,215** exceed the 1024 cap; total 2-epoch training tokens = **517,474** |
| llama-server LoRA support | `C:\Users\heyzo\llama-glimmer\llama-server.exe` (build b10355) `--help` lists `--lora FNAME`, `--lora-scaled`, `--lora-init-without-apply` — **VERIFIED** |
| conversion script | llama.cpp source cloned at matching tag **b10355** → `C:\Users\heyzo\llama.cpp\convert_lora_to_gguf.py`; `--help` runs clean |

installed during readiness: `gguf` 0.19.0 (pip). it dragged huggingface-hub to
1.27.0 which broke transformers; **fixed** by re-pinning huggingface-hub to
0.36.2 and re-verifying `import transformers` + the conversion script. pip will
print a resolver warning about gguf wanting hub>=1.0 — cosmetic, both import fine.

## training text decision (recorded)

**System prompt is EXCLUDED from the training text.** Qwen2.5's chat template
silently injects "You are Qwen, created by Alibaba Cloud…" when no system
message is present, so the script builds ChatML by hand (no `apply_chat_template`)
and asserts the default system string is absent. Loss is computed ONLY on
assistant-reply tokens + closing `<|im_end|>`; user tokens are masked -100.
Voice lives in the replies; the serve-time prefill supplies system text, and the
LoRA must not bind the register to any particular system string.

## hyperparameters (frozen in `train_voice_lora.py`)

QLoRA NF4 (double-quant, bf16 compute) · LoRA r=16 α=32 dropout=0.05 on
q/k/v/o/gate/up/down projections (40.4M trainable params) · seq len 1024 ·
per-device batch 1 · grad-accum 16 (effective 16) · 2 epochs · cosine LR 2e-4,
warmup 3% · paged_adamw_8bit · seed 42 · gradient checkpointing on ·
sdpa attention · `local_files_only=True` (no network at train time).

2,430 examples / 16 = **~152 optimizer steps**.

## VRAM math vs 8,188 MiB

| component | estimate |
|---|---|
| base weights NF4 (≈6.53B quantized linear params × ~0.55 B/param) | ~3.6 GiB |
| embed_tokens + lm_head kept bf16 (2 × 545M × 2 B — vocab 152,064, untied) | ~2.0 GiB |
| LoRA weights fp32 + grads fp32 (40.4M × 4 B × 2) | ~0.31 GiB |
| optimizer (paged AdamW 8-bit, states paged to host) | ~0.08 GiB |
| activations w/ checkpointing @ seq 526 (boundary states + recompute workspace) | ~0.35 GiB |
| logits + CE grad spike (526 × 152,064 × fp32 × 2) — the 152k-vocab tax | ~0.6 GiB |
| **peak @ observed max seq 526** | **~7.0 GiB** |
| worst case if a 1024-token example existed (none do) | ~7.8 GiB |

8,188 MiB total minus ~0.3–0.5 GiB Windows/WDDM overhead ≈ 7.7 GiB usable.
**Fits at the observed sequence lengths — but ONLY with the serving process down.**

### HARD RULE: darkflobi-fast MUST be stopped during training

nvidia-smi at audit time: 5,173 MiB used with serving up → training cannot
coexist. The script enforces this: it aborts unless ≥ 7,000 MiB VRAM is free.
Stopping/starting pm2 is an **operator action** (`pm2 stop darkflobi-fast`,
afterwards `pm2 start darkflobi-fast`); the training script never touches pm2.

## expected wall-clock

517,474 training tokens (2 epochs) at a conservative 400–800 tok/s for QLoRA
fwd+bwd with checkpointing on this card → 11–22 min compute, plus ~3–5 min
model load/quantize and overhead. **Expect ~20–45 minutes total; budget 1 hour.**

## launch

```powershell
# operator: pm2 stop darkflobi-fast   (script will refuse to start otherwise)
C:\Users\heyzo\AppData\Local\Programs\Python\Python312\python.exe C:\Users\heyzo\.styxx\glimmer-day-zero\train_voice_lora.py
# optional plumbing check first (INVALID for any verdict per prereg):
#   ...\python.exe ...\train_voice_lora.py --smoke
```

outputs: `voice_lora\adapter\` (PEFT adapter) · `voice_lora\loss_log.jsonl` ·
`voice_lora\TRAIN_RECEIPT.json` (dataset sha256, hyperparams, start/end UTC,
wall seconds, final loss, peak VRAM, package versions).

## adapter → servable next to the Q4_K_M GGUF

```powershell
# 1. convert PEFT adapter to GGUF (CPU work, safe while serving is up)
cd C:\Users\heyzo\llama.cpp
C:\Users\heyzo\AppData\Local\Programs\Python\Python312\python.exe convert_lora_to_gguf.py `
  --base C:\Users\heyzo\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28 `
  --outfile C:\Users\heyzo\.styxx\glimmer-day-zero\voice_lora\voice-lora-qwen25-7b-f16.gguf `
  --outtype f16 `
  C:\Users\heyzo\.styxx\glimmer-day-zero\voice_lora\adapter

# 2. VOICE arm serving = BASE serving command + one flag:
C:\Users\heyzo\llama-glimmer\llama-server.exe `
  -m C:\Users\heyzo\models\darkflobi-fast\Qwen2.5-7B-Instruct-Q4_K_M.gguf `
  --lora C:\Users\heyzo\.styxx\glimmer-day-zero\voice_lora\voice-lora-qwen25-7b-f16.gguf `
  <same flags as the current darkflobi-fast pm2 config>
```

caveat, stated up front: the adapter is trained against the NF4-quantized base
and applied at serve time over the Q4_K_M base — two different 4-bit
quantizations. this is standard practice and the mismatch is small, but it is a
real approximation; G1 (voice acquired, blinded 0.75 bar) is the gate that
catches it if the transfer fails. prereg arms require everything except the
`--lora` flag identical between BASE and VOICE.

## rejected alternatives (for the record)

- **trl SFTTrainer**: would add a new dependency to a live environment for
  functionality (chat formatting) we need to override anyway to keep the system
  prompt out. plain Trainer + hand-rolled masking is fully inspectable.
- **llama.cpp native finetune**: no LoRA-on-quantized training path in the
  b10355 Windows release binaries; the HF/bnb path is measured working on this box.
- **unsloth**: needs triton, which is not installed and is fragile on native
  Windows; bitsandbytes already passes its CUDA self-test here, so the standard
  stack wins on credibility.
