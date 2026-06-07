"""Instrumentation ONLY for the wall-dynamics experiment (PREREG_wall_dynamics_2026_06_07.md):
raw across-layer logit-lens trajectories of a teacher-forced answer. NO feature selection,
NO labels, NO kill-gate here -- pure measurement, so it can be positive-controlled before the
prereg is locked. For a GREEDY-decoded answer, teacher-forcing that exact answer reproduces the
same per-position hidden states generation produced (deterministic greedy + causal attention),
so these trajectories equal the free-run-decode trajectories by construction.

trajectory(tok, model, prompt, answer) -> dict of per-(layer,position) measurements:
  emit_pos : np.ndarray (n_layers+1, n_ans)  log-prob of each emitted answer token, per layer
  ent_pos  : np.ndarray (n_layers+1, n_ans)  next-token Shannon entropy (nats), per layer
  am_first : list[int] (n_layers+1)          logit-lens argmax token id at the FIRST answer
                                             predict-position, per layer (for top1_churn)
  n_layers, alen
All read at the predict-answer positions (teacher-forced), exactly as run_wall_read.loglens but
across EVERY layer and keeping per-position detail (so token-1-dropped is a free recompute).
"""
from __future__ import annotations
import numpy as np
import torch
from spec_exec_logprob import build_input


@torch.no_grad()
def trajectory(tok, model, prompt, answer):
    pids = tok(build_input(tok, prompt), return_tensors="pt").to(model.device).input_ids
    aids = tok(answer, add_special_tokens=False, return_tensors="pt").to(model.device).input_ids
    if aids.shape[1] == 0:
        return None
    full = torch.cat([pids, aids], 1)
    hs = model(full, output_hidden_states=True).hidden_states          # tuple (n_layers+1) of (1,T,d)
    p = pids.shape[1]
    pos = torch.arange(p - 1, full.shape[1] - 1, device=full.device)    # predict-answer positions
    toks = full[0, p:]                                                  # answer token ids
    norm = model.model.norm
    head = model.lm_head
    n_ans = int(aids.shape[1])
    emit_pos = np.empty((len(hs), n_ans), dtype=np.float64)
    ent_pos = np.empty((len(hs), n_ans), dtype=np.float64)
    am_first = []
    ar = torch.arange(n_ans, device=full.device)
    for li, h in enumerate(hs):
        logits = head(norm(h[0])).float()                              # (T, vocab)
        lp = torch.log_softmax(logits, -1)
        sel = lp[pos]                                                  # (n_ans, vocab)
        emit_pos[li] = sel[ar, toks].cpu().numpy()
        ent_pos[li] = (-(sel.exp() * sel).sum(-1)).cpu().numpy()
        am_first.append(int(sel[0].argmax().item()))
    return {"emit_pos": emit_pos, "ent_pos": ent_pos, "am_first": am_first,
            "n_layers": len(hs) - 1, "alen": n_ans}


if __name__ == "__main__":
    from spec_exec_local import load_model, free
    from spec_exec_harness import load_truthfulqa
    tok, model = load_model("Qwen/Qwen2.5-1.5B-Instruct")
    items = load_truthfulqa(4, 11)
    for it in items[:3]:
        g = it["correct"][0] if it.get("correct") else None
        if not g:
            continue
        tr = trajectory(tok, model, it["prompt"], g)
        nl = tr["n_layers"]
        lp = tr["emit_pos"].mean(1); en = tr["ent_pos"].mean(1)
        print(f"\nprompt {it['prompt'][:50]!r} alen={tr['alen']} L={nl}")
        print(f"  emit_lp  L1={lp[1]:+.2f} Lmid={lp[nl//2]:+.2f} Lfinal={lp[-1]:+.2f}")
        print(f"  entropy  L1={en[1]:.2f} Lmid={en[nl//2]:.2f} Lfinal={en[-1]:.2f}")
        flips = sum(tr['am_first'][i] != tr['am_first'][i-1] for i in range(2, nl-2))
        print(f"  am_first flips (mid-stack) = {flips}")
    free(model)
    print("\n[instrument smoke OK]")
