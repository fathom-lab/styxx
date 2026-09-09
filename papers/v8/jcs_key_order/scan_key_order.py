# -*- coding: utf-8 -*-
"""Blast radius of the JCS key-order defect: does ANY committed artifact contain
an object whose sibling keys sort differently by Unicode code point (what
styxx.attestation._jcs does) than by UTF-16 code unit (what RFC 8785 section
3.2.3 requires)?

Denominator: every git-tracked *.json and *.jsonl in the repository, plus every
base64 blob in conformance/sworn/blobs.json that decodes to JSON (those blobs
ARE the payloads the sworn conformance vectors take digests over).

Every object at every depth is walked. Duplicate keys are preserved
(object_pairs_hook), so nothing is collapsed before it is measured.

Run:  python papers/v8/jcs_key_order/scan_key_order.py
Writes scan_key_order.json next to this file (UTF-8, LF).
"""
from __future__ import annotations

import base64
import binascii
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
_OUT = _HERE.parent / "scan_key_order.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:  # pragma: no cover
    pass


def utf16_key(s: str) -> bytes:
    """The RFC 8785 section 3.2.3 sort key: the string's UTF-16 code units,
    big-endian, so bytewise comparison IS code-unit comparison. surrogatepass
    lets a lone surrogate (illegal but representable in a Python str) through
    rather than crashing the scan."""
    return s.encode("utf-16-be", errors="surrogatepass")


def classify(keys: list[str]) -> dict:
    """Classify one object's sibling key list."""
    non_ascii = [k for k in keys if any(ord(c) > 0x7F for c in k)]
    non_bmp = [k for k in keys if any(ord(c) > 0xFFFF for c in k)]
    cp_order = sorted(keys)
    cu_order = sorted(keys, key=utf16_key)
    return {
        "non_ascii": non_ascii,
        "non_bmp": non_bmp,
        "differs": cp_order != cu_order,
        "cp_order": cp_order,
        "cu_order": cu_order,
    }


class Walker:
    def __init__(self) -> None:
        self.objects = 0
        self.keys = 0
        self.obj_non_ascii = 0
        self.obj_non_bmp = 0
        self.obj_differs = 0
        self.max_key_code_point = 0
        self.keys_ge_e000 = 0
        self.non_ascii_keys: dict[str, int] = {}
        self.divergent_sites: list[dict] = []
        self.non_bmp_sites: list[dict] = []
        self.eq_docs = 0
        self.eq_same = 0
        self.eq_diff = 0
        self.eq_outside_domain = 0
        self.eq_diff_sources: list[str] = []
        self.eq_domain_examples: list[dict] = []

    def pairs_hook(self, pairs):
        """json object_pairs_hook: measure, then return a dict (duplicates are
        measured before any collapse)."""
        self.objects += 1
        keys = [k for k, _ in pairs]
        self.keys += len(keys)
        c = classify(keys)
        for k in keys:
            for ch in k:
                o = ord(ch)
                if o > self.max_key_code_point:
                    self.max_key_code_point = o
                if o >= 0xE000:
                    self.keys_ge_e000 += 1
                    break
        for k in c["non_ascii"]:
            self.non_ascii_keys[k] = self.non_ascii_keys.get(k, 0) + 1
        if c["non_ascii"]:
            self.obj_non_ascii += 1
        if c["non_bmp"]:
            self.obj_non_bmp += 1
            self._pending_non_bmp = c
        if c["differs"]:
            self.obj_differs += 1
            self._pending_differs = c
        return dict(pairs)

    def _equivalence(self, docs: list, source: str) -> None:
        """Canonicalize each parsed document under the shipped sort and under the
        proposed (UTF-16) sort and compare the bytes."""
        for doc in docs:
            self.eq_docs += 1
            try:
                a = _shipped_jcs(doc)
                b = _jcs_repaired(doc)
            except Exception as exc:
                self.eq_outside_domain += 1
                if len(self.eq_domain_examples) < 10:
                    self.eq_domain_examples.append(
                        {"source": source, "error": f"{type(exc).__name__}: {exc}"[:200]}
                    )
                continue
            if a == b:
                self.eq_same += 1
            else:
                self.eq_diff += 1
                self.eq_diff_sources.append(source)

    def scan_text(self, text: str, source: str, jsonl: bool) -> str | None:
        """Returns an error string, or None on success."""
        before_bmp, before_diff = self.obj_non_bmp, self.obj_differs
        docs = []
        try:
            if jsonl:
                for i, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        docs.append(json.loads(line, object_pairs_hook=self.pairs_hook))
            else:
                docs.append(json.loads(text, object_pairs_hook=self.pairs_hook))
        except Exception as exc:  # malformed by design in some attack fixtures
            return f"{type(exc).__name__}: {exc}"
        self._equivalence(docs, source)
        if self.obj_non_bmp > before_bmp:
            self.non_bmp_sites.append(
                {"source": source, "objects": self.obj_non_bmp - before_bmp,
                 "example_keys": self._pending_non_bmp["non_bmp"]}
            )
        if self.obj_differs > before_diff:
            self.divergent_sites.append(
                {"source": source, "objects": self.obj_differs - before_diff,
                 "cp_order": self._pending_differs["cp_order"],
                 "cu_order": self._pending_differs["cu_order"]}
            )
        return None


# ---------------------------------------------------------------------------
# Repair equivalence. The proposed repair changes exactly one thing: the sort
# key. This pass canonicalizes every parsed document BOTH ways and compares the
# bytes, so "no committed digest moves" is measured, not inferred.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_REPO))
from styxx.attestation import _jcs as _shipped_jcs  # noqa: E402
from styxx.attestation import _es_number_to_string  # noqa: E402


def _jcs_repaired(obj):
    """styxx.attestation._jcs with the ONE proposed change: the sort key."""
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _es_number_to_string(obj)
    if isinstance(obj, list):
        return "[" + ",".join(_jcs_repaired(x) for x in obj) + "]"
    if isinstance(obj, dict):
        parts = (
            json.dumps(k, ensure_ascii=False) + ":" + _jcs_repaired(v)
            for k, v in sorted(obj.items(), key=lambda kv: utf16_key(kv[0]))
        )
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"not JCS-serializable: {type(obj).__name__}")


def tracked_json_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.json", "*.jsonl"],
        cwd=str(_REPO), capture_output=True, check=True,
    ).stdout.decode("utf-8")
    return sorted(p for p in out.split("\0") if p)


GROUPS = [
    ("papers/**", lambda p: p.startswith("papers/")),
    ("conformance/sworn/**", lambda p: p.startswith("conformance/sworn/")),
    ("*.certificate.json", lambda p: p.endswith(".certificate.json")),
    ("*.sworn.json", lambda p: p.endswith(".sworn.json")),
    ("*.sworn-receipt.json", lambda p: p.endswith(".sworn-receipt.json")),
    ("*.jsonl", lambda p: p.endswith(".jsonl")),
]


def main() -> int:
    files = tracked_json_files()
    w = Walker()
    errors: list[dict] = []
    read_errors: list[dict] = []
    jsonl_fallback: list[str] = []
    per_group = {name: 0 for name, _ in GROUPS}

    for rel in files:
        for name, pred in GROUPS:
            if pred(rel):
                per_group[name] += 1
        path = _REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            read_errors.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        err = w.scan_text(text, rel, jsonl=rel.endswith(".jsonl"))
        if err and not rel.endswith(".jsonl"):
            # Several vendored benchmark files carry a .json extension but hold
            # JSON Lines. Retry that way so the denominator has no hole.
            err2 = w.scan_text(text, rel, jsonl=True)
            if err2 is None:
                jsonl_fallback.append(rel)
                err = None
            else:
                err = f"{err} | as-jsonl: {err2}"
        if err:
            errors.append({"file": rel, "error": err})

    files_parsed = len(files) - len(errors) - len(read_errors)

    # ---- second pass: the sworn conformance blobs (the digested payloads) ----
    blobs_path = _REPO / "conformance/sworn/blobs.json"
    blobs_total = blobs_json = blobs_notjson = blobs_malformed = blobs_undecodable = 0
    if blobs_path.exists():
        blobs = json.loads(blobs_path.read_text(encoding="utf-8"))
        for digest, b64 in sorted(blobs.items()):
            blobs_total += 1
            try:
                raw = base64.b64decode(b64, validate=True)
                text = raw.decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                blobs_undecodable += 1
                continue
            stripped = text.lstrip()
            if not stripped[:1] in "{[":
                # A document blob (markdown, text): hashed as raw bytes by
                # sha256, never routed through jcs, so key order cannot apply.
                blobs_notjson += 1
                continue
            err = w.scan_text(text, f"conformance/sworn/blobs.json#{digest}", jsonl=False)
            if err:
                # Deliberately malformed fixtures: the verifier must REJECT
                # these, so no canonical form is ever taken over them.
                blobs_malformed += 1
            else:
                blobs_json += 1

    charon = _REPO / "papers/charon/charon.log.jsonl"
    result = {
        "schema": "styxx.v8.jcs-key-order-scan/v1",
        "label": "a scan, not a result; no claim beyond the counts below",
        "repo": str(_REPO),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO), capture_output=True, check=True
        ).stdout.decode().strip(),
        "rule": (
            "an object is AFFECTED iff sorted(keys) (Unicode code point, what "
            "styxx.attestation._jcs does) != sorted(keys, key=utf-16-be) (UTF-16 "
            "code unit, what RFC 8785 section 3.2.3 requires)"
        ),
        "denominator": {
            "files_enumerated": len(files),
            "files_parsed": files_parsed,
            "files_unparseable": len(errors),
            "files_unreadable": len(read_errors),
            "files_parsed_via_jsonl_fallback": len(jsonl_fallback),
            "files_by_group": per_group,
            "charon_log_present": charon.exists(),
            "blobs_total": blobs_total,
            "blobs_scanned_as_json": blobs_json,
            "blobs_document_text_not_json": blobs_notjson,
            "blobs_json_shaped_but_malformed_by_design": blobs_malformed,
            "blobs_undecodable": blobs_undecodable,
        },
        "counts": {
            "objects_walked": w.objects,
            "keys_walked": w.keys,
            "objects_with_any_non_ascii_key": w.obj_non_ascii,
            "objects_with_any_non_bmp_key": w.obj_non_bmp,
            "objects_where_the_two_orders_differ": w.obj_differs,
            "keys_containing_any_char_ge_U+E000": w.keys_ge_e000,
            "max_code_point_over_all_keys": hex(w.max_key_code_point),
        },
        "repair_equivalence": {
            "rule": (
                "canonicalize each parsed document with styxx.attestation._jcs and "
                "with the same function differing ONLY in sort key "
                "(sorted(..., key=lambda kv: kv[0].encode('utf-16-be'))), compare bytes"
            ),
            "documents_compared": w.eq_docs,
            "byte_identical": w.eq_same,
            "bytes_differ": w.eq_diff,
            "outside_the_jcs_domain_both_ways": w.eq_outside_domain,
            "differing_sources": w.eq_diff_sources,
            "domain_error_examples": w.eq_domain_examples,
        },
        "distinct_non_ascii_keys": {
            k: n for k, n in sorted(w.non_ascii_keys.items(), key=lambda kv: -kv[1])
        },
        "non_bmp_sites": w.non_bmp_sites,
        "divergent_sites": w.divergent_sites,
        "unparseable_files": errors,
        "unreadable_files": read_errors,
        "files_parsed_via_jsonl_fallback": jsonl_fallback,
    }

    _OUT.write_text(
        json.dumps(result, indent=1, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    d, c = result["denominator"], result["counts"]
    print("styxx JCS key-order blast-radius scan")
    print(f"  commit                              : {result['commit']}")
    print(f"  files enumerated (git-tracked json) : {d['files_enumerated']}")
    print(f"  files parsed                        : {d['files_parsed']}")
    print(f"  files unparseable / unreadable      : {d['files_unparseable']} / {d['files_unreadable']}")
    print(f"  files parsed via JSONL fallback     : {d['files_parsed_via_jsonl_fallback']}")
    for name, n in d["files_by_group"].items():
        print(f"    of which {name:24s}: {n}")
    print(f"  sworn blobs decoded and scanned     : {d['blobs_scanned_as_json']} of {d['blobs_total']}")
    print(f"    blob remainder: {d['blobs_document_text_not_json']} document text, "
          f"{d['blobs_json_shaped_but_malformed_by_design']} malformed by design, "
          f"{d['blobs_undecodable']} undecodable")
    print(f"  charon.log.jsonl present            : {d['charon_log_present']}")
    print(f"  objects walked                      : {c['objects_walked']}")
    print(f"  keys walked                         : {c['keys_walked']}")
    print(f"  objects with any non-ASCII key      : {c['objects_with_any_non_ascii_key']}")
    print(f"  objects with any non-BMP key        : {c['objects_with_any_non_bmp_key']}")
    print(f"  objects where the two orders DIFFER : {c['objects_where_the_two_orders_differ']}")
    print(f"  keys with any char >= U+E000        : {c['keys_containing_any_char_ge_U+E000']}")
    print(f"  max code point over all keys        : {c['max_code_point_over_all_keys']}")
    e = result["repair_equivalence"]
    print(f"  --- repair equivalence (shipped sort vs proposed UTF-16 sort) ---")
    print(f"  documents canonicalized both ways   : {e['documents_compared']}")
    print(f"  byte-identical                      : {e['byte_identical']}")
    print(f"  bytes DIFFER                        : {e['bytes_differ']}")
    print(f"  outside the jcs domain both ways    : {e['outside_the_jcs_domain_both_ways']}")
    print(f"  distinct non-ASCII keys seen        : {len(result['distinct_non_ascii_keys'])}")
    for k, n in list(result["distinct_non_ascii_keys"].items())[:20]:
        print(f"    {n:6d}  {k!r}  code points {[hex(ord(ch)) for ch in k]}")
    print(f"  written                             : {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
