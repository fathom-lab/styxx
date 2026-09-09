#!/usr/bin/env python3
"""recomputable.py -- is the floor recomputable from the stored bytes, or do the
certs record only summaries?

Answers it by rederiving, from the per-item raw payloads alone, every digest the
fingerprint certs carry, per Appendix A.3:

  token_ids_sha256      = sha256(UTF-8(JCS(ids)))
  output_sha256         = sha256(UTF-8(output_text))
  channels.exact.hash   = sha256(concatenation, in item_id order, of the raw
                                 32-byte token_ids_sha256 digests)

and by counting the raw quantities each channel's distance needs.  A cert that
carried only summaries would fail here.  Imports nothing from styxx.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from floor2 import load_entries, sorted_items  # noqa: E402


def jcs_ints(ids):
    """JCS of an array of integers: no whitespace, integers rendered minimally."""
    return "[" + ",".join(str(int(v)) for v in ids) + "]"


def main():
    log_dir = os.path.join(HERE, "..", "first_verdict_2026_09_09", "log")
    entries = load_entries(log_dir)
    fps = [(m["index"], c) for _, m, c in entries if m["type"] == "fingerprint"]

    bad = 0
    for idx, cert in fps:
        items = sorted_items(cert)
        cat = b""
        for it in items:
            h = hashlib.sha256(jcs_ints(it["token_ids"]).encode("utf-8")).hexdigest()
            if h != it["token_ids_sha256"]:
                bad += 1
                print("  entry %d item %s token_ids_sha256 MISMATCH" % (idx, it["item_id"]))
            o = hashlib.sha256(it["output_text"].encode("utf-8")).hexdigest()
            if o != it["output_sha256"]:
                bad += 1
                print("  entry %d item %s output_sha256 MISMATCH" % (idx, it["item_id"]))
            if len(it["token_ids"]) != it["n_generated"]:
                bad += 1
                print("  entry %d item %s n_generated != len(token_ids)" % (idx, it["item_id"]))
            cat += bytes.fromhex(it["token_ids_sha256"])
        eh = hashlib.sha256(cat).hexdigest()
        stored = cert["body"]["channels"]["exact"]["hash"]
        ok = eh == stored
        if not ok:
            bad += 1
        print("entry %d  run %d  items=%d  exact.hash %s"
              % (idx, cert["body"]["run_index"], len(items), "RECOMPUTES" if ok else "MISMATCH"))

    print()
    it0 = sorted_items(fps[0][1])[0]
    print("raw payload per item, as stored: %s" % sorted(it0.keys()))
    print("  exact needs token_ids           -> present (%d ids on item 0)" % len(it0["token_ids"]))
    print("  seqlp needs seq_logprob         -> present (%r)" % it0["seq_logprob"])
    print("  topk  needs per-position ids+lps-> present (%d positions, k=%d)"
          % (len(it0["topk"]), len(it0["topk"][0]["ids"])))
    print()
    print("blobs/ directory contents: %r  (body.items_blob not used; items are inline)"
          % sorted(os.listdir(os.path.join(log_dir, "blobs"))))
    print()
    print("VERDICT: the floor is recomputable from the stored bytes. %d mismatches." % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
