"""Verify the PARRHESIA receipt on this announcement — trust nothing, re-derive everything.

    pip install styxx
    python examples/parrhesia/verify.py

It re-hashes the announcement (catches a swapped message) and RE-RUNS the deterministic audit (catches
a forged verdict). Prints VERIFIED only if both reproduce. You are not trusting us — you are re-running
the auditor on a content-addressed message.
"""
import json
from pathlib import Path

import styxx

HERE = Path(__file__).resolve().parent
prompt = (HERE / "prompt.txt").read_text(encoding="utf-8")
message = (HERE / "announcement.txt").read_text(encoding="utf-8")
receipt = json.loads((HERE / "receipt.json").read_text(encoding="utf-8"))

result = styxx.verify_receipt(receipt, prompt, message)
print(f"styxx {styxx.__version__}")
print(f"receipt schema   : {receipt['schema']}")
print(f"passed audit     : {receipt['verdict']['passed_register_audit']}")
print(f"certifies        : {receipt['certifies']}")
print(f"message digest   : {receipt['message_sha256'][:16]}…  match={result.message_digest_match}")
print(f"audit reproduces : {result.audit_reproduces}")
print(f"\n>>> {result.status} <<<")
raise SystemExit(0 if result.status == "VERIFIED" else 1)
