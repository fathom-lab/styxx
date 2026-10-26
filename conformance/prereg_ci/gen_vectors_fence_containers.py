# -*- coding: utf-8 -*-
"""Generate vectors_fence_containers.json: the B3 review's shapes for the fence and RESULT readers.

  python -B conformance/prereg_ci/gen_vectors_fence_containers.py          write the file where none is committed
  python -B conformance/prereg_ci/gen_vectors_fence_containers.py --check  regenerate in memory; exit 1 if it differs

A new file beside vectors_fence.json, which is never written again (a receipt is history too). The B3 review found
two classes past the readers that vectors_fence.json pins: a ```result or ```locks fence inside a container that
CommonMark renders and the B3 shadow rule did not count (ADV1: a list item's continuation line 4 or more columns in
or behind a tab, a footnote's, a nested list marker 4 columns in, a tab after a block quote marker), and RESULT
bytes in a wide encoding that declared nothing and were not refused (M1). These vectors pin, for "result" and
"locks", that each such shape refuses (core._fence_json) and where the rule stops (indented code with no list item
or footnote opened above it is code; below a list that has ended it refuses, an over-refusal stated in advance);
and, for "result", what core._result_reading and core._result_candidate give bytes in UTF-16 and UTF-32 with a
byte-order mark, with none, with a NUL byte, and binary bytes with a marker planted in them (ADV2's reading of any
file a pull request writes rests on core._result_candidate).

The renderer evidence for the container shapes is markdown-it-py (CommonMark) with mdit_py_plugins' footnote plugin,
run in the fix round's case files, not here. Every case is synthetic and states its outcome in advance; a case that
does not come out as stated is a refusal: printed, exit 1, nothing written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci.gen_vectors import _WORD_RX, _dump  # noqa: E402
from conformance.prereg_ci.gen_vectors_fence import BODIES, _json_leg, _outcome  # noqa: E402
from conformance.prereg_ci.replay import blob_entry, content_id, sha256  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402

OUT = ROOT / "conformance" / "prereg_ci" / "vectors_fence_containers.json"
SCHEMA = "styxx.prereg-ci.conformance.fence-containers/v0"
LABEL = "conformance vectors for prereg-ci's fence and RESULT readers on the B3 review's shapes; no claim"
F = "```"
INFOS = ("result", "locks")


def body(info: str, verdict: str = "PASS") -> str:
    b = dict(BODIES[info])
    if info == "result":
        b["verdict"] = verdict
    return json.dumps(b)


def block(info: str, verdict: str = "PASS") -> str:
    return f"{F}{info}\n{body(info, verdict)}\n{F}\n"


def indented(text: str, pad: str) -> str:
    return "".join(pad + ln + "\n" for ln in text.splitlines())


def contained(info: str) -> Dict[str, tuple]:
    """name -> (why, document): shapes a CommonMark renderer shows as an <info> fence and the B3 rule did not
    count. The names are the ones tests/test_prereg_ci_result_review.py holds."""
    b = block(info, "FAIL")
    opener, rest = b.splitlines()[0], "\n".join(b.splitlines()[1:])
    return {
        "list-continuation-4": ("a list item's continuation, 4 columns in",
                                "# d\n\n-   the verdict:\n\n" + indented(b, "    ")),
        "ordered-continuation-4": ("an ordered item's continuation, 4 columns in",
                                   "# d\n\n10. the verdict:\n" + indented(b, "    ")),
        "list-continuation-tab": ("a list item's continuation behind a tab",
                                  "# d\n\n- the verdict:\n\n" + indented(b, "\t")),
        "list-continuation-7": ("a 4-column item's continuation, 7 columns in",
                                "# d\n\n-   the verdict:\n\n" + indented(b, "       ")),
        "nested-list-continuation": ("a nested item's continuation", "# d\n\n- a\n  - b\n\n" + indented(b, "      ")),
        "lazy-paragraph-then-4": ("below a lazy paragraph line of an item", "# d\n\n- a\nlazy\n" + indented(b, "    ")),
        "quoted-list-continuation": ("an item's continuation inside a block quote",
                                     "# d\n\n> - a\n>\n" + "".join(">     " + ln + "\n" for ln in b.splitlines())),
        "nested-marker-at-4": ("a nested list marker 4 columns in, the fence on its line",
                               "# d\n\n-   a\n\n    - " + opener + "\n" + indented(rest, "      ")),
        "footnote-continuation": ("a footnote's continuation, 4 columns in",
                                  "# d\n\ntext[^1]\n\n[^1]: note\n\n" + indented(b, "    ")),
        "footnote-label-line": ("the fence on a footnote label's line",
                                "# d\n\ntext[^1]\n\n[^1]: " + opener + "\n" + indented(rest, "    ")),
        "quote-then-tab": ("a tab after a block quote marker", "# d\n\n" + "".join("> \t" + ln + "\n"
                                                                                    for ln in b.splitlines())),
    }


def fence_cases(info: str) -> List[tuple]:
    """(name, why, text, want): want is the strict reader's outcome ("object", "none" or "refuses")."""
    rows = [(f"{info}/contained/{name}", why, text, "refuses") for name, (why, text) in contained(info).items()]
    rows += [
        (f"{info}/contained/beside-the-plain-fence", "the plain fence, and a list item's continuation fence below it",
         f"# d\n\n{block(info)}\n-   but read this one:\n\n" + indented(block(info, "FAIL"), "    "), "refuses"),
        (f"{info}/boundary/indented-code-below-prose", "4 columns in with no list item or footnote above: code",
         "# d\n\ntext\n\n" + indented(block(info), "    "), "none"),
        (f"{info}/boundary/indented-code-behind-a-tab", "a tab in with no list item or footnote above: code",
         "# d\n\n" + indented(block(info), "\t"), "none"),
        (f"{info}/boundary/indented-code-below-a-quote", "4 columns in below a block quote: code",
         "# d\n\n> quoted\n\n" + indented(block(info), "    "), "none"),
        (f"{info}/boundary/below-a-list-that-has-ended", "4 columns in below a list a paragraph has ended: a renderer "
         "prints code, and the reader refuses (over-refusal, stated)",
         "# d\n\n- a\n\npara\n\n" + indented(block(info), "    "), "refuses"),
        (f"{info}/boundary/the-plain-fence-below-a-list", "the one plain fence at column 0 below a list: read",
         f"# d\n\n- a\n- b\n\n{block(info)}", "object"),
    ]
    return rows


def encoding_cases() -> List[tuple]:
    """(name, why, bytes, candidate, code, n): what core._result_candidate and core._result_reading give."""
    fenced = f"# RESULT a\n\n{block('result', 'FAIL')}"
    plain = "# notes\n\nThe verdict was FAIL.\n\n" + F + "python\nprint(1)\n" + F + "\n"
    # the tag is built from pieces, so this file's bytes hold none (core._result_candidate reads any file a
    # pull request writes)
    span = '# RESULT a\n\nVerdict: <' + 'sworn r="path:p/receipt.json#/verdict" k="quote">`FAIL`</' + 'sworn>\n'
    confusable = "# RESULT a\n\n" + F + "r" + chr(0x435) + "sult\n{}\n" + F + "\n"
    boms = {"utf-16-le": b"\xff\xfe", "utf-16-be": b"\xfe\xff", "utf-32-le": b"\xff\xfe\x00\x00",
            "utf-32-be": b"\x00\x00\xfe\xff"}
    refuse = "RESULT_REFUSES"
    rows = []
    for enc, bom in boms.items():
        rows.append((f"result/encoding/{enc}-bom-fence", f"a result fence in {enc} with its byte-order mark: refused",
                     bom + fenced.encode(enc), True, refuse, 0))
        rows.append((f"result/encoding/{enc}-bom-prose", f"prose and a python fence in {enc}: nothing",
                     bom + plain.encode(enc), False, None, 0))
    rows += [
        ("result/encoding/utf-16-le-bom-span", "a sworn verdict span in UTF-16 with its mark: refused",
         b"\xff\xfe" + span.encode("utf-16-le"), True, refuse, 0),
        ("result/encoding/utf-16-le-bom-confusable", "a confusable info string in UTF-16 with its mark: refused",
         b"\xff\xfe" + confusable.encode("utf-16-le"), True, refuse, 0),
        ("result/encoding/utf-16-le-no-bom", "UTF-16 with no mark is read as the bytes it is: no fence there",
         fenced.encode("utf-16-le"), False, None, 0),
        ("result/encoding/utf-8-with-a-nul", "a UTF-8 RESULT holding a NUL byte is read as UTF-8, as at B3",
         fenced.encode("utf-8") + b"\x00", True, None, 1),
    ]
    rng = random.Random(20261025)
    noise = bytes(rng.getrandbits(8) for _ in range(8192))
    rows += [
        ("result/binary/noise", "8192 seeded random bytes: no candidate", noise, False, None, 0),
        ("result/binary/fence-run-and-a-high-byte", "binary bytes holding a backtick run and a high byte: no candidate "
         "(the narrow marker for bytes that are not UTF-8)", b"\x89PNG\r\n\x1a\n" + noise[:4096] + b"```\xc3\xa9"
         + noise[4096:], False, None, 0),
        ("result/binary/a-result-marker", "binary bytes holding a result fence's marker: refused, as at B3",
         b"\x89PNG\r\n\x1a\n" + noise[:4096] + b"\n```result\n" + noise[4096:], True, refuse, 0),
    ]
    return rows


def build() -> dict:
    problems: List[str] = []
    cases: List[dict] = []
    for info in INFOS:
        for name, why, text, want in fence_cases(info):
            cases.append({"name": name, "why": why, "info": info, "data": text.encode("utf-8"), "want": want})
    for name, why, data, cand, code, n in encoding_cases():
        cases.append({"name": name, "why": why, "info": "result", "data": data, "want_candidate": cand,
                      "want_code": code, "want_n": n})
    if len({c["name"] for c in cases}) != len(cases):
        problems.append("two cases share a name")
    blobs: Dict[str, dict] = {}
    vectors = []
    for c in cases:
        data, info = c["data"], c["info"]
        expect: Dict[str, Any] = {"candidate": core._result_candidate(data)}
        try:
            text: Optional[str] = core._decision_text(data)
        except UnicodeDecodeError:
            text = None
        if text is not None:
            expect["fence_json"] = _json_leg(text, info)
        if info == "result":
            expect["reading"] = json.loads(json.dumps(core._result_reading(data)))
        if "want" in c:
            got = _outcome(expect["fence_json"])
            if got != c["want"]:
                problems.append(f"{c['name']}: the strict reader reads {got}, stated {c['want']}")
            if info == "result" and (expect["reading"]["code"] is not None) != (c["want"] == "refuses"):
                problems.append(f"{c['name']}: the reading's code is {expect['reading']['code']}, stated "
                                f"{'a refusal' if c['want'] == 'refuses' else 'none'}")
            if info == "result" and c["want"] != "none" and not expect["candidate"]:
                problems.append(f"{c['name']}: a document the reading refuses or reads is no candidate")
        if "want_code" in c:
            r = expect["reading"]
            if (expect["candidate"], r["code"], len(r["declarations"])) != (c["want_candidate"], c["want_code"],
                                                                             c["want_n"]):
                problems.append(f"{c['name']}: reads candidate {expect['candidate']}, code {r['code']} with "
                                f"{len(r['declarations'])} declarations, stated {c['want_candidate']}, "
                                f"{c['want_code']} with {c['want_n']}")
        key = sha256(data)
        blobs.setdefault(key, blob_entry(data))
        inputs = {"text": key}
        vectors.append({"id": content_id(c["name"], {"info": info, **inputs}), "name": c["name"], "why": c["why"],
                        "info": info, "inputs": inputs, "expect": expect})
    if problems:
        raise SystemExit("REFUSED\n" + "\n".join(problems))
    return {
        "schema": SCHEMA, "label": LABEL,
        "rules": {
            "id": "sha256 of the sorted, compact UTF-8 JSON of {name, inputs: {info, text}}",
            "text": "inputs.text names bytes in this file's blobs by sha256 ({utf8} or {b64}); fence_json reads them "
                    "decoded as UTF-8 with CRLF and a lone CR read as LF, and is absent for bytes that are not UTF-8; "
                    "candidate and reading are handed the bytes",
            "candidate": "styxx.prereg_ci.core._result_candidate(bytes)",
            "fence_json": "styxx.prereg_ci.core._fence_json(text, info): {line, object}, {none: true}, or "
                          "{raises, message}",
            "reading": "result only: styxx.prereg_ci.core._result_reading(bytes), as JSON",
        },
        "vectors": vectors,
        "blobs": dict(sorted(blobs.items())),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory; exit 1 if a byte differs")
    args = ap.parse_args(argv)
    doc = build()
    hits = sorted({m.group(0) for m in _WORD_RX.finditer(json.dumps(doc, ensure_ascii=False))})
    if hits:
        print("REFUSED the file prints a word the lab does not: " + ", ".join(hits))
        return 1
    raw = _dump(doc)
    if args.check or OUT.exists():
        if OUT.exists() and OUT.read_bytes() == raw:
            print(f"SAME {OUT.name}={hashlib.sha256(raw).hexdigest()[:16]}; nothing written")
            return 0
        print(f"DIFFERS {OUT.name}: the run does not regenerate the committed bytes; nothing written. New "
              f"vectors go in new files.")
        return 1
    OUT.write_bytes(raw)
    if OUT.read_bytes() != raw:
        print(f"REFUSED {OUT.name} did not read back as written")
        return 1
    print(f"WROTE {OUT.name} {len(doc['vectors'])} vectors, {len(doc['blobs'])} blobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
