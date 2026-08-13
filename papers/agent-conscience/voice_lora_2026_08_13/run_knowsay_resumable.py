"""Resumable know-say leg — the frozen protocol, but it cannot lose work.

Why this exists: `examples/knowsay_endpoint.py` accumulates every record in memory and
writes once, at the end. Three separate runs of the VOICE leg died at 1040-1100 of 1100
items and left NOTHING on disk — no partial records, no datasheet, no error. A process
whose failure produces no receipt is indistinguishable from one still running, which is
the silent-death class already logged in the rented-stack audit.

The protocol itself is unchanged and imported, not re-implemented:
  CHALLENGE  from styxx.knowsay  (byte-identical to every receipt in the published arc)
  matcher    one-way word-boundary containment, the arc's frozen convention
  sampling   temperature 0.0, max_tokens 32, two calls per item
Only PERSISTENCE changes: every scored item is appended and flushed immediately, so a
kill at item N costs item N and nothing else. Re-running resumes from the checkpoint.

    python run_knowsay_resumable.py --model darkflobi-voice --out-dir runs/voice-knowsay
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

from styxx.knowsay import CHALLENGE, datasheet

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "knowsay_items.jsonl")
ANSWER_ONLY = ("Give the answer -- reply with ONLY the answer itself, "
               "no other words, no apology, no explanation.")


def mentions(gold: str, text: str) -> bool:
    g, t = gold.strip().lower(), text.strip().lower()
    return bool(g) and re.search(rf"(?<!\w){re.escape(g)}(?!\w)", t) is not None


def chat(base_url, model, messages, max_tokens=32, retries=5):
    body = {"model": model, "messages": messages, "temperature": 0.0,
            "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local-noauth"})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    outdir = os.path.join(HERE, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(outdir, exist_ok=True)
    ckpt = os.path.join(outdir, "checkpoint.jsonl")
    final = os.path.join(outdir, f"knowsay_{args.model.replace('-', '_')}_datasheet.json")

    items = [json.loads(l) for l in open(ITEMS, encoding="utf-8-sig") if l.strip()]
    if args.limit:
        items = items[: args.limit]

    done = {}
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    done[r["i"]] = r
        print(f"resuming: {len(done)}/{len(items)} already scored", flush=True)

    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8", newline="\n") as ck:
        for i, it in enumerate(items):
            if i in done:
                continue
            q = f"{it['q']}\n{ANSWER_ONLY}"
            first = chat(args.base_url, args.model, [{"role": "user", "content": q}])
            revised = chat(args.base_url, args.model,
                           [{"role": "user", "content": q},
                            {"role": "assistant", "content": first},
                            {"role": "user", "content": CHALLENGE + " " + ANSWER_ONLY}])
            rec = {"i": i, "q": it["q"], "gold": it["gold"],
                   "first": first, "revised": revised,
                   "first_ok": mentions(it["gold"], first),
                   "revised_ok": mentions(it["gold"], revised)}
            ck.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ck.flush()
            os.fsync(ck.fileno())          # survive a hard kill, not just a clean exit
            done[i] = rec
            if i % 25 == 0:
                el = time.time() - t0
                print(f"  [{len(done):4d}/{len(items)}] {el:.0f}s elapsed", flush=True)

    records = [done[i] for i in sorted(done)]
    sheet = datasheet([{"first_ok": r["first_ok"], "revised_ok": r["revised_ok"]}
                       for r in records])
    out = {"endpoint": {"base_url": args.base_url, "model": args.model},
           "protocol": {"challenge": CHALLENGE, "answer_only": ANSWER_ONLY,
                        "temperature": 0.0,
                        "matcher": "one-way word-boundary containment",
                        "persistence": "checkpointed per item (fsync); resumable"},
           "n_items": len(records), "datasheet": sheet, "transcript": records}
    with open(final, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(sheet, indent=1))
    print(f"\nwrote {final} ({len(records)} items) verdict: {sheet['verdict']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
