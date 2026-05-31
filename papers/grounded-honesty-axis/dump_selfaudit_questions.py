"""Draw N TriviaQA questions for the Claude self-audit. Writes QUESTIONS ONLY (shown to the model)
and SHA-256-locks the answer key to a separate file the model commits NOT to read before answering.
No OpenAI. PREREG_self_audit_2026_05_31.md.
"""
from __future__ import annotations
import hashlib, json, os
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
N = 40

ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=True)
qs, key = [], []
for ex in ds:
    if len(qs) >= N:
        break
    q = (ex.get("question") or "").strip()
    ans = ex.get("answer", {}) or {}
    aliases = (list(ans.get("aliases", []) or [])
               + list(ans.get("normalized_aliases", []) or [])
               + ([ans["value"]] if ans.get("value") else []))
    aliases = [a for a in aliases if a]
    if q and aliases:
        i = len(qs)
        qs.append({"i": i, "question": q})
        key.append({"i": i, "aliases": aliases})

with open(os.path.join(HERE, "selfaudit_questions.json"), "w", encoding="utf-8") as f:
    json.dump(qs, f, indent=2, ensure_ascii=False)
with open(os.path.join(HERE, "selfaudit_key.json"), "w", encoding="utf-8") as f:
    json.dump(key, f, indent=2, ensure_ascii=False)

keyhash = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
print(f"wrote {len(qs)} questions; ANSWER-KEY SHA-256 (locked pre-answer): {keyhash}")
print("--- QUESTIONS (answers withheld) ---")
for it in qs:
    print(f'{it["i"]:2}. {it["question"]}')
