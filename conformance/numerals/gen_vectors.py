# -*- coding: utf-8 -*-
"""Generate the numerals conformance vectors from `texts.py`, replay them, write `vectors_v2.json`.

    py -3.12 -B conformance/numerals/gen_vectors.py --v1-recording PATH [--out vectors_v2.json]
    py -3.12 -B conformance/numerals/gen_vectors.py --check            # replay the committed set, write nothing

Four families. `reader`: `extract_numbers` and `uncovered_spans` on one text under the shipped flags,
minus the `context` field (a function of the text). `binding`: `certify_doc` on one text and its
receipts under the shipped flags: the verdict, its class, the counts, every ledger row's token, status
and receipt_ref, and the counted band. `v1_reader` and `v1_binding`: the same texts and cases under
`V15_READER_V2 = False`, whose expectations are READ FROM A RECORDING made at HEAD before the reader v2
existed (`--v1-recording`): the generator re-derives them with the master flag off and refuses to write
if any differ. That recording is the machine form of "flag off == HEAD".

`id = sha256(utf8(jcs({"family": f, "text": t[, "receipts": r]})))`. A receipt file is named by its
index, so `receipt_ref` reads `receipt[j]:path` in every expectation. `set_sha256` is the sha256 of the
JCS text of the index minus `set_sha256` and minus `provenance`.

A RECEIPT IS HISTORY TOO. The generator refuses to write a vector whose id already stands in any
committed `conformance/numerals/vectors_*.json` with a different `expect`. A later reader records its
expectations in a new file, never by editing this one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.certify as C  # noqa: E402
import texts as T  # noqa: E402
from styxx.attestation import jcs  # noqa: E402

SCHEMA = "styxx-oath/numerals-vectors/v1"
READER_SCHEMA = "styxx-oath/numerals/v2"
OUT = HERE / "vectors_v2.json"
STUB = {"schema": "stub", "head": None, "receipts": []}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def vector_id(family: str, text: str, receipts=None) -> str:
    key = {"family": family, "text": text}
    if receipts is not None:
        key["receipts"] = receipts
    return _sha(jcs(key))


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "context"}


def read(text: str) -> dict:
    return {"numbers": [_strip(n) for n in C.extract_numbers(text)],
            "uncovered": [_strip(u) for u in C.uncovered_spans(text)]}


def bind(text: str, receipts: list, tmp: Path, tag: str) -> dict:
    doc = tmp / f"{tag}.md"
    doc.write_text(text + "\n", encoding="utf-8")
    rps = []
    for j, r in enumerate(receipts):
        rp = tmp / f"{tag}_r{j}.json"
        rp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        rps.append(rp)
    old = C._receipt_binding_block
    C._receipt_binding_block = lambda rp, near: STUB      # reads git; not part of the reader
    try:
        cert = C.certify_doc(doc, rps)
    finally:
        C._receipt_binding_block = old
    names = {rp.name: f"receipt[{j}]" for j, rp in enumerate(rps)}

    def ref(r):
        if not isinstance(r, str):
            return r
        for name, short in names.items():
            if r.startswith(name + ":"):
                return short + r[len(name):]
            r = r.replace(name + ":", short + ":")
        return r

    return {"verdict": cert["verdict"], "verdict_class": cert["verdict"].split(",")[0].strip(),
            "counts": cert["counts"],
            "rows": [{"token": e["token"], "status": e["status"], "receipt_ref": ref(e.get("receipt_ref"))}
                     for e in cert["ledger"]],
            "uncovered": [_strip(u) for u in cert["uncovered_items"]]}


def set_flags(master: bool) -> None:
    C.V15_READER_V2 = master
    C._v15_reset()


def committed_sets() -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "conformance/numerals/vectors_*.json"],
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        out = []
    return [ROOT / p for p in out]


def existing_expectations() -> dict:
    seen: dict = {}
    for p in committed_sets():
        doc = json.loads(p.read_text(encoding="utf-8"))
        for fam in doc.get("families", {}).values():
            for v in fam.get("vectors", []):
                seen[v["id"]] = (p.name, v["expect"])
    return seen


def generate(v1_recording: Path | None) -> dict:
    shipped = C.V15_READER_V2
    families: dict = {"reader": [], "binding": [], "v1_reader": [], "v1_binding": []}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        set_flags(True)
        for t in T.all_texts():
            families["reader"].append({"id": vector_id("reader", t), "text": t, "expect": read(t)})
        for i, (name, text, rec) in enumerate(T.BINDING_CASES):
            recs = rec if isinstance(rec, list) else [rec]
            families["binding"].append({"id": vector_id("binding", text, recs), "name": name, "text": text,
                                        "receipts": recs, "expect": bind(text, recs, tmp, f"b{i}")})
        set_flags(False)
        for t in T.all_texts():
            families["v1_reader"].append({"id": vector_id("v1_reader", t), "text": t, "expect": read(t)})
        for i, (name, text, rec) in enumerate(T.BINDING_CASES):
            recs = rec if isinstance(rec, list) else [rec]
            families["v1_binding"].append({"id": vector_id("v1_binding", text, recs), "name": name, "text": text,
                                           "receipts": recs, "expect": bind(text, recs, tmp, f"v{i}")})
        set_flags(shipped)
    # two cases with the same text and receipts are one vector (the id is over the text and the
    # receipts, never over a case's name); the earlier case's name is kept
    for name, vs in families.items():
        seen_ids: dict = {}
        for v in vs:
            seen_ids.setdefault(v["id"], v)
        families[name] = list(seen_ids.values())
    provenance = {"generated_by": "conformance/numerals/gen_vectors.py", "texts": "conformance/numerals/texts.py"}
    if v1_recording is not None:
        rec = json.loads(v1_recording.read_text(encoding="utf-8"))
        by_text = {r["text"]: r for r in rec["reader"]}
        diffs = []
        for v in families["v1_reader"]:
            exp = by_text.get(v["text"])
            if exp is None or {"numbers": exp["numbers"], "uncovered": exp["uncovered"]} != v["expect"]:
                diffs.append(("reader", v["text"]))
        by_case = {(r["text"], jcs(r["receipts"])): (i, r) for i, r in enumerate(rec["binding"])}
        for v in families["v1_binding"]:
            found = by_case.get((v["text"], jcs(v["receipts"])))
            if found is None:
                diffs.append(("binding", v["text"]))
                continue
            i, exp = found
            rows = []
            for r in exp["rows"]:
                ref = r["receipt_ref"]
                if isinstance(ref, str):
                    # the recorder named receipts c{i}_r{j}.json; the vectors name them receipt[j]
                    for j in range(len(v["receipts"])):
                        ref = ref.replace(f"c{i}_r{j}.json:", f"receipt[{j}]:")
                rows.append({"token": r["token"], "status": r["status"], "receipt_ref": ref})
            want = {"verdict": exp["verdict"], "verdict_class": exp["verdict"].split(",")[0].strip(),
                    "counts": exp["counts"], "rows": rows, "uncovered": exp["uncovered"]}
            if want != v["expect"]:
                diffs.append(("binding", v["text"], want, v["expect"]))
        if diffs:
            raise SystemExit(f"REFUSED: the master-off reader differs from the HEAD recording on {len(diffs)} "
                             f"vector(s): {diffs[:3]}")
        provenance["v1_recording"] = {"head": rec["head"], "certify_sha256_lf": rec["certify_sha256_lf"],
                                      "n_reader": rec["n_reader"], "n_binding": rec["n_binding"],
                                      "note": "the v1_* families reproduce this recording exactly"}
    certify_bytes = (ROOT / "styxx" / "certify.py").read_bytes()
    doc = {
        "schema": SCHEMA,
        "reader": {"schema": READER_SCHEMA,
                   "certify_sha256_lf": _sha(certify_bytes.replace(b"\r\n", b"\n").decode("utf-8")),
                   "rules": sorted(k for k in C._V15_RULE_FLAGS if getattr(C, k)),
                   "master": "V15_READER_V2"},
        "id_rule": "sha256(utf8(jcs({family, text[, receipts]})))",
        "families": {name: {"n": len(vs), "vectors": sorted(vs, key=lambda v: v["id"])}
                     for name, vs in families.items()},
    }
    doc["set_sha256"] = _sha(jcs(doc))
    doc["provenance"] = provenance
    return doc


def replay(doc: dict) -> list:
    """Every vector of *doc* through the installed reader; the mismatches, never a skip."""
    bad = []
    core = {k: v for k, v in doc.items() if k not in ("set_sha256", "provenance")}
    if _sha(jcs(core)) != doc.get("set_sha256"):
        bad.append(("set_sha256", "does not re-derive"))
    shipped = C.V15_READER_V2
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for fam, body in doc["families"].items():
                set_flags(not fam.startswith("v1_"))
                for k, v in enumerate(body["vectors"]):
                    if fam.endswith("reader"):
                        if vector_id(fam, v["text"]) != v["id"]:
                            bad.append((fam, v["text"], "id"))
                        got = read(v["text"])
                    else:
                        if vector_id(fam, v["text"], v["receipts"]) != v["id"]:
                            bad.append((fam, v["text"], "id"))
                        got = bind(v["text"], v["receipts"], tmp, f"{fam}{k}")
                    if got != v["expect"]:
                        bad.append((fam, v["text"], got, v["expect"]))
    finally:
        set_flags(shipped)
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-recording", type=Path, default=None,
                    help="JSON recorded from HEAD before the build; the v1_* families must reproduce it")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="replay the committed set and write nothing")
    a = ap.parse_args(argv)
    if a.check:
        bad = []
        for p in committed_sets() or [OUT]:
            bad += replay(json.loads(p.read_text(encoding="utf-8")))
        print("replay:", "ok" if not bad else f"{len(bad)} mismatch(es): {bad[:3]}")
        return 1 if bad else 0
    doc = generate(a.v1_recording)
    seen = existing_expectations()
    clashes = [(v["id"], where) for fam in doc["families"].values() for v in fam["vectors"]
               if v["id"] in seen and seen[v["id"]][1] != v["expect"] for where in [seen[v["id"]][0]]
               if where != a.out.name]
    if clashes:
        raise SystemExit(f"REFUSED: {len(clashes)} vector id(s) already stand in a committed set with a "
                         f"different expectation (a receipt is history too): {clashes[:3]}")
    bad = replay(doc)
    if bad:
        raise SystemExit(f"REFUSED: the generated set does not replay: {bad[:3]}")
    a.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", a.out, {k: v["n"] for k, v in doc["families"].items()}, "set_sha256", doc["set_sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
