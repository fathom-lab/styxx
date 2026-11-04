#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_vectors.py — run the conformance vectors with a Python verifier (by default the gate's own ledger.py).

    py -3.12 -B run_vectors.py [--vectors vectors.json] [--ledger-py <path to a ledger.py>]

Exit 0 when every case answers as the file says. A verifier that is not ledger.py is conformant when it does.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vectorlib  # noqa: E402


def load(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("ledger_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def answer(L, text: bytes, withheld, anchors):
    return vectorlib.answer(L, text, withheld, anchors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=str(HERE / "vectors.json"))
    ap.add_argument("--ledger-py", default="C:/Users/heyzo/clawd/gate/ledger.py")
    a = ap.parse_args(argv)
    vpath = Path(a.vectors)
    doc = json.loads(vpath.read_text("utf-8"))
    base = (vpath.parent / doc["base"]).read_bytes()
    if hashlib.sha256(base).hexdigest() != doc["base_sha256"]:
        print("the base file is not the one the vectors were made over")
        return 2
    L = load(Path(a.ledger_py))
    failed = 0
    for c in doc["cases"]:
        text = vectorlib.apply(base, c["transform"], L)
        if hashlib.sha256(text).hexdigest() != c["input_sha256"]:
            print("FAIL %s: the transform did not produce the vector's input bytes" % c["name"])
            failed += 1
            continue
        got = answer(L, text, c.get("withheld"), c.get("anchors"))
        exp = c["expect"]
        if c["transform"]["op"] == "dup_key":
            same = got["ok"] == exp["ok"]
        else:
            same = all(got.get(k) == exp.get(k) for k in exp)
        print("%s %s" % ("ok  " if same else "FAIL", c["name"]))
        if not same:
            failed += 1
            print("   expected %s" % json.dumps(exp, sort_keys=True))
            print("   got      %s" % json.dumps(got, sort_keys=True))
    print("%d of %d vectors answered as the file says" % (len(doc["cases"]) - failed, len(doc["cases"])))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
