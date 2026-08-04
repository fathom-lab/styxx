"""One-command replication of the mutual-legibility matrix (B37).

    python papers/disjoint-worlds/replicate_legibility.py

What it does:
  1. Ensures the four concept extractions exist (banked .npz used if present; otherwise
     extracts Llama-3.2-3B / Llama-3.2-1B / gemma-2-2b / Qwen2.5-1.5B on your GPU —
     ~90 min on an 8 GB card, each banked on completion so a crash resumes).
  2. Runs the frozen matrix apparatus (`run_b37.py`) — 12 directed pairs, label-free
     discovery + read, scored by styxx.protocol against the git-frozen gates.
  3. Compares your matrix to the canonical committed receipt (`b37_result.json`).

Replication bar (from REPLICATIONS.md): the clique/island TOPOLOGY must reproduce —
every clique pair (llama_3b, llama_1b, gemma_2b) discovers at >= 0.30 and every qwen pair
at <= 0.25 in your run. Exact seed_acc equality is expected ONLY if you use our banked
extractions (CPU-deterministic); fresh GPU extraction varies by hardware, hence the
topology bar. A run that BREAKS the topology is more valuable than one that matches —
either way, open an