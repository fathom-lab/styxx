"""Build the darkflobi voice fine-tuning dataset from his own transcripts.

Why this exists: prompting a 7B to "be darkflobi" hits a hard ceiling — the
base model's assistant priors live in the weights, and a system prompt can dent
them but not erase them. His voice is in 26k of his own turns; the honest way
to give it to a local model is to train on it, not to describe it.

Output: voice_dataset.jsonl in chat format —
    {"messages": [{"role":"user",...},{"role":"assistant",...}]}
one operator->darkflobi exchange per line, ready for an unsloth/peft SFT run.

The styxx angle (why this is an experiment and not just a chore): a LoRA is a
weight-channel intervention, the same channel the frame-locality work bounds as
a dose. So the question is pre-registerable and measurable: does teaching a
model a VOICE also move its honesty properties? Baseline the register vitals
and the know-say gap BEFORE the tune, re-measure after, and the delta is the
finding — including, and especially, if the delta is null.

    python build_voice_dataset.py --min 400 --max 6000
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIST = HERE / "darkflobi_history.sqlite"
OUT = HERE / "voice_dataset.jsonl"
STATS = HERE / "voice_dataset_stats.json"

ME = {"darkflobi", "assistant", "me", "bot"}
THEM = {"operator", "user", "flobi", "human"}

# turns that are machinery, not speech
BAD = re.compile(
    r"```|\[\[\w+\]\]|\[message_id|HEARTBEAT|NO_REPLY|"
    r"<system-reminder|<command-name|Caveat: The messages|"
    r"\bTraceback\b|Request interrupted", re.I)


def fts_table(con):
    names = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table'")]
    return next((n for n in names
                 if any(k in n.lower() for k in ("sess", "hist", "messag", "turn"))
                 and not n.lower().endswith(
                     ("_data", "_idx", "_content", "_docsize", "_config"))), None)


def col(cols, *cands):
    return next((c for c in cols if c.lower() in cands), None)


def build(min_len, max_len, since="2026-03"):
    con = sqlite3.connect(f"file:{HIST.as_posix()}?mode=ro", uri=True)
    tbl = fts_table(con)
    if not tbl:
        con.close()
        return [], {"error": "no fts table"}
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{tbl}")')]
    c_text = col(cols, "text", "content", "body")
    c_role = col(cols, "speaker", "role", "who")
    c_date = col(cols, "date", "ts", "timestamp")
    c_sess = col(cols, "session_id", "session", "conversation_id")
    rows = list(con.execute(
        f'select "{c_date}", "{c_role}", "{c_sess}", "{c_text}" from "{tbl}"'))
    con.close()

    pairs, seen = [], set()
    for i in range(1, len(rows)):
        d, role, sid, text = rows[i]
        pd, prole, psid, ptext = rows[i - 1]
        role = (role or "").strip().lower()
        prole = (prole or "").strip().lower()
        if role not in ME or prole not in THEM or sid != psid:
            continue
        if isinstance(d, str) and d < since:      # mature register only
            continue
        t, p = (text or "").strip(), (ptext or "").strip()
        if not (min_len <= len(t) <= max_len) or not (5 <= len(p) <= 1200):
            continue
        if BAD.search(t) or BAD.search(p):
            continue
        key = t[:120]
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"messages": [{"role": "user", "content": p},
                                   {"role": "assistant", "content": t}],
                      "date": d})
    stats = {
        "pairs": len(pairs),
        "date_range": [pairs[0]["date"], pairs[-1]["date"]] if pairs else None,
        "mean_reply_chars": round(
            sum(len(x["messages"][1]["content"]) for x in pairs) / len(pairs))
        if pairs else 0,
        "source_table": tbl,
        "filters": {"min_len": min_len, "max_len": max_len, "since": since},
    }
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=400)
    ap.add_argument("--max", type=int, default=6000)
    ap.add_argument("--since", default="2026-03")
    args = ap.parse_args()

    pairs, stats = build(args.min, args.max, args.since)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(json.dumps({"messages": p["messages"]},
                               ensure_ascii=False) + "\n")
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
