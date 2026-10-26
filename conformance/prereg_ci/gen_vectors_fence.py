# -*- coding: utf-8 -*-
"""Generate vectors_fence.json: the fence and RESULT vectors of the prereg-ci conformance set (step B3).

  python -B conformance/prereg_ci/gen_vectors_fence.py           write the file where none is committed
  python -B conformance/prereg_ci/gen_vectors_fence.py --check   regenerate in memory; exit 1 if a byte differs

A new file beside the committed set, which is never written again (a receipt is history too). PLAN K4:
styxx.protocol._select_gates_block hard-codes its info string, so styxx.prereg_ci.core holds the scanner
with the info string as a parameter (core._fence_at). These vectors pin, for "gates", that it decides as
_select_gates_block does on each attack class tests/test_protocol_v4.py names, on the fences CommonMark
renders that the scanner does not count, and on plain cases; for "result" and "locks", that each attack
class refuses, that those shadow fences refuse too (core._fence_json), and that duplicate and non-ASCII
keys refuse; and for "result", what core._result_reading declares: the ```result fence's three keys and
the sworn span on a receipt's verdict leaf, with each check styxx.sworn makes from the bytes alone.

Every case is synthetic and states its outcome in advance; a case that does not come out as stated is a
refusal: printed, exit 1, nothing written. Reads no history; the bytes of every vector are in the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci.gen_vectors import _WORD_RX, _dump  # noqa: E402
from conformance.prereg_ci.replay import blob_entry, content_id, sha256  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.protocol import GateSpecError, _select_gates_block  # noqa: E402

OUT = ROOT / "conformance" / "prereg_ci" / "vectors_fence.json"
SCHEMA = "styxx.prereg-ci.conformance.fence/v0"
LABEL = "conformance vectors for prereg-ci's fence scanner and RESULT readers; no claim"
F = "```"
BOM = chr(0xFEFF)
RLO = chr(0x202E)
INFOS = ("gates", "result", "locks")
BODIES = {
    "gates": {"gates": {"G": {"metric": "m", "op": "<=", "value": 0.2}},
              "outcomes": [{"when": {"G": True}, "verdict": "PASS"}, {"when": {"G": False}, "verdict": "FAIL"}],
              "smoke_verdict": "SMOKE"},
    "result": {"prereg": "PREREG_a.md", "receipt": "receipt.json", "verdict": "PASS"},
    "locks": {"scorer": "score.py", "reads": ["data.csv"], "outcomes": ["PASS", "FAIL"], "withdraws": {}},
}
# one letter of each info string swapped for a Cyrillic look-alike (U+0430, U+0435, U+043E)
CONFUSABLE = {"gates": "g" + chr(0x430) + "tes", "result": "r" + chr(0x435) + "sult",
              "locks": "l" + chr(0x43E) + "cks"}
NO_GATES = "prereg has no ```gates block"
RP = "papers/x/receipt.json"


def honest(info: str) -> str:
    return json.dumps(BODIES[info])


def decoy(info: str) -> str:
    body = json.loads(honest(info))
    if info == "gates":
        body["gates"]["G"]["value"] = 99
    elif info == "result":
        body["verdict"] = "FAIL"
    else:
        body["scorer"] = "other.py"
    return json.dumps(body)


def fence_cases(info: str) -> List[dict]:
    """(name, why, text, want): want is the scanner's outcome ("block", "none" or "refuses") and, for
    "result" and "locks", the strict reader's ("object", "none" or "refuses")."""
    h, d = honest(info), decoy(info)
    lead = h[1:h.index(":")]
    dup = h.replace(lead + ":", lead + ": 1, " + lead + ":", 1)
    homoglyph = h.replace(lead + ":", lead.replace(lead[1], chr(0x441), 1) + ": 1, " + lead + ":", 1)
    block = f"{F}{info}\n{h}\n{F}\n"
    plain = "block" if info == "gates" else ("block", "object")
    refused = "refuses" if info == "gates" else ("refuses", "refuses")
    keys = "block" if info == "gates" else ("block", "refuses")
    shadow = "none" if info == "gates" else ("none", "refuses")
    nothing = "none" if info == "gates" else ("none", "none")
    rows = [
        ("plain", "the one plain fence", f"# d\n\n{block}", plain),
        ("plain-crlf", "the plain fence with CRLF newlines, read as LF", ("# d\n\n" + block).replace("\n", "\r\n"),
         plain),
        ("no-fence", "prose only", "# d\n\nno fence here.\n", nothing),
        ("indented-code", "four columns of indent is code a renderer prints as text",
         "# d\n\n" + "".join("    " + ln + "\n" for ln in block.splitlines()), nothing),
        ("attack/decoy-in-comment", "a decoy fence in an HTML comment beside the visible one",
         f"# d\n\n<!--\n{F}{info}\n{d}\n{F}\n-->\n\n{block}", refused),
        ("attack/tilde-fence", "a tilde fence beside a hidden backtick decoy",
         f"# d\n\n<!--\n{F}{info}\n{d}\n{F}\n-->\n\n~~~{info}\n{h}\n~~~\n", refused),
        ("attack/case-variant", "an upper-case info string", f"# d\n\n{F}{info.upper()}\n{h}\n{F}\n", refused),
        ("attack/fence-only-in-comment", "the one fence sits in an HTML comment",
         f"# d\n\n<!--\n{block}-->\n", refused),
        ("attack/confusable-info", "a Cyrillic letter in an info string, beside a decoy",
         f"# d\n\n{F}{CONFUSABLE[info]}\n{h}\n{F}\n\n{F}{info}\n{d}\n{F}\n", refused),
        ("attack/tab-indented-opener", "a tab-indented decoy opener a renderer prints as text, the honest fence "
         "in a comment", f"# d\n\n<!--\n{block}-->\n\n\t{F}{info}\n{d}\n{F}\n", refused),
        ("attack/unterminated-comment", "an unterminated HTML comment hides the fence to the end",
         f"# d\n\n<!--\n\n{block}", refused),
        ("attack/unclosed-fence", "the fence is never closed", f"# d\n\n{F}{info}\n{h}\n", refused),
        ("attack/two-fences", "two plain fences", f"# d\n\n{block}\n{F}{info}\n{d}\n{F}\n", refused),
        ("attack/indented-opener", "a two-space opener a renderer reads", f"# d\n\n  {F}{info}\n{h}\n{F}\n", refused),
        ("keys/duplicate-key", "a key declared twice", f"# d\n\n{F}{info}\n{dup}\n{F}\n", keys),
        ("keys/non-ascii-key", "a key with a Cyrillic letter beside its ASCII twin",
         f"# d\n\n{F}{info}\n{homoglyph}\n{F}\n", keys),
        ("shadow/bom-on-line-1", "a byte-order mark before the opener on line 1", BOM + block, shadow),
        ("shadow/block-quote", "the fence in a block quote",
         "# d\n\n" + "".join("> " + ln + "\n" for ln in block.splitlines()), shadow),
        ("shadow/list-item", "the fence as a list item's content",
         "# d\n\n" + "".join(("- " if i == 0 else "  ") + ln + "\n" for i, ln in enumerate(block.splitlines())),
         shadow),
        ("shadow/ordered-list-item", "the fence as an ordered list item's content",
         "# d\n\n" + "".join(("1. " if i == 0 else "   ") + ln + "\n" for i, ln in enumerate(block.splitlines())),
         shadow),
        ("shadow/quote-in-list", "a block quote inside a list item",
         "# d\n\n" + "".join(("- > " if i == 0 else "  > ") + ln + "\n" for i, ln in enumerate(block.splitlines())),
         shadow),
        ("shadow/second-word", "an info string with a second word", f"# d\n\n{F}{info} json\n{h}\n{F}\n", shadow),
    ]
    if info != "gates":
        rows.append(("shadow/beside-the-plain-fence", "the plain fence, and a quoted one beside it",
                     f"# d\n\n{block}\n> {F}{info}\n> {d}\n> {F}\n", ("block", "refuses")))
    return rows


def span(verdict: str, ref: str = f"path:{RP}#/verdict", kind: str = "quote") -> str:
    return f'<sworn r="{ref}" k="{kind}">`{verdict}`</sworn>'


def result_cases() -> List[dict]:
    """(name, why, text, code, declarations): what core._result_reading reads from a RESULT."""
    fence = lambda body: f"# RESULT\n\n{F}result\n{body}\n{F}\n"            # noqa: E731
    good = BODIES["result"]
    refuse = "RESULT_REFUSES"
    rows = [
        ("result/fence/three-keys", "a fence naming prereg, receipt and verdict", fence(json.dumps(good)), None, 1),
        ("result/fence/lacks-prereg", "a fence without its prereg",
         fence(json.dumps({"receipt": "receipt.json", "verdict": "PASS"})), refuse, 0),
        ("result/fence/extra-key", "a fence with a key the reader does not read",
         fence(json.dumps({**good, "gates": {"G": True}})), refuse, 0),
        ("result/fence/verdict-not-a-string", "a number for the verdict", fence(json.dumps({**good, "verdict": 5})),
         refuse, 0),
        ("result/fence/verdict-empty", "an empty verdict", fence(json.dumps({**good, "verdict": ""})), refuse, 0),
        ("result/fence/not-an-object", "a list in the fence", fence(json.dumps([good])), refuse, 0),
        ("result/fence/not-strict-json", "a trailing comma", fence(json.dumps(good)[:-1] + ",}"), refuse, 0),
        ("result/sworn/verdict-leaf", "a sworn span on the receipt's verdict leaf",
         "# RESULT\n\nVerdict: " + span("PASS") + ".\n", None, 1),
        ("result/sworn/other-leaves", "spans on other leaves declare nothing",
         "# RESULT\n\n" + " ".join(span("PASS", ref=f"path:{RP}#{f}") for f in ("/gates_sha256", "/verdict/x", "L3"))
         + "\n", None, 0),
        ("result/sworn/in-a-code-fence", "a span inside a fenced code block, which sworn does not read",
         f"# RESULT\n\n{F}\n" + span("PASS") + f"\n{F}\n", None, 0),
        ("result/sworn/in-inline-code", "a span inside an inline code span, which sworn does not read",
         "# RESULT\n\n`` " + span("PASS") + " ``\n", None, 0),
        ("result/sworn/rn-form", "a span naming its receipt by a manifest id: printed, not paired",
         "# RESULT\n\n" + span("PASS", ref="r1#/verdict") + "\n", None, 0),
        ("result/sworn/unlexable-names-the-leaf-in-prose", "a document whose fence lines sworn cannot balance, "
         "naming the verdict leaf in an inline code span and in no tag: nothing is declared and nothing refused",
         "# notes\n\n  " + F + "locks in a wrapped line\n\nThe leaf `path:<receipt>#/verdict` is named in prose.\n",
         None, 0),
        ("result/both/agree", "the fence and a span on the same receipt, agreeing",
         fence(json.dumps({**good, "receipt": RP})) + "\nVerdict: " + span("PASS") + ".\n", None, 2),
        ("result/both/disagree", "the fence and a span that disagree: both declarations are read here, and "
         "core._declared refuses the pair", fence(json.dumps({**good, "receipt": RP})) + "\nVerdict: "
         + span("HOLD") + ".\n", None, 2),
    ]
    attacks = {
        "hidden-in-a-comment": "Verdict: HOLD. <!-- " + span("PASS") + " -->",
        "behind-an-unterminated-comment": "Verdict: HOLD.\n<!--\n" + span("PASS"),
        "nested": f'<sworn r="path:{RP}#/verdict" k="quote">' + span("PASS") + "</sworn>",
        "unclosed": f'<sworn r="path:{RP}#/verdict" k="quote">`PASS`',
        "a-stray-closer-elsewhere": span("PASS") + " and later </sworn>",
        "single-quoted-attributes": f"<sworn r='path:{RP}#/verdict' k='quote'>`PASS`</sworn>",
        "upper-case-tag": f'<SWORN r="path:{RP}#/verdict" k="quote">`PASS`</SWORN>',
        "kind-numeric": span("PASS", kind="numeric"),
        "kind-unknown": span("PASS", kind="quot"),
        "kind-absent": span("PASS", kind="absent"),
        "two-needles": f'<sworn r="path:{RP}#/verdict" k="quote">`PASS` or `HOLD`</sworn>',
        "blank-needle": span(" "),
        "directional-override": span(RLO + "PASS"),
        "over-the-length-cap": span("A" * 301),
        "a-path-sworn-cannot-name": span("PASS", ref="path:/abs/receipt.json#/verdict"),
        "unbalanced-fences": F + "\n" + span("PASS"),
    }
    for name, body in attacks.items():
        rows.append(("result/sworn-attack/" + name, "refused: " + name.replace("-", " "),
                     "# RESULT\n\n" + body + "\n", refuse, 0))
    return rows


def _select(text: str, info: str) -> dict:
    try:
        got = core._select_fence(text, info)
    except GateSpecError as e:
        return {"raises": "GateSpecError", "message": str(e)}
    return {"none": True} if got is None else {"block": got}


def _protocol(text: str) -> dict:
    try:
        return {"block": _select_gates_block(text)}
    except GateSpecError as e:
        if str(e).startswith(NO_GATES):
            return {"none": True}
        return {"raises": "GateSpecError", "message": str(e)}


def _json_leg(text: str, info: str) -> dict:
    try:
        got = core._fence_json(text, info)
    except GateSpecError as e:
        return {"raises": "GateSpecError", "message": str(e)}
    return {"none": True} if got is None else {"line": got[0], "object": got[1]}


def _outcome(leg: dict) -> str:
    return "refuses" if "raises" in leg else ("none" if "none" in leg else ("object" if "object" in leg else "block"))


def build() -> dict:
    problems: List[str] = []
    cases: List[dict] = []
    for info in INFOS:
        for name, why, text, want in fence_cases(info):
            cases.append({"name": f"{info}/{name}", "why": why, "info": info, "text": text, "want": want})
    for name, why, text, code, n in result_cases():
        cases.append({"name": name, "why": why, "info": "result", "text": text, "want_code": code, "want_n": n})
    if len({c["name"] for c in cases}) != len(cases):
        problems.append("two cases share a name")
    blobs: Dict[str, dict] = {}
    vectors = []
    for c in cases:
        data = c["text"].encode("utf-8")
        text = core._decision_text(data)
        info = c["info"]
        expect: Dict[str, Any] = {"select": _select(text, info)}
        if info == "gates":
            expect["protocol"] = _protocol(text)
            if expect["protocol"] != expect["select"]:
                problems.append(f"{c['name']}: the scanner and _select_gates_block decide differently")
        else:
            expect["fence_json"] = _json_leg(text, info)
        if info == "result":
            expect["reading"] = json.loads(json.dumps(core._result_reading(data)))
        if "want" in c:
            want = c["want"] if isinstance(c["want"], tuple) else (c["want"],)
            got = (_outcome(expect["select"]),) + ((_outcome(expect["fence_json"]),) if info != "gates" else ())
            if got != want:
                problems.append(f"{c['name']}: reads {got}, stated {want}")
        if "want_code" in c:
            r = expect["reading"]
            if r["code"] != c["want_code"] or len(r["declarations"]) != c["want_n"]:
                problems.append(f"{c['name']}: reads code {r['code']} with {len(r['declarations'])} declarations, "
                                f"stated {c['want_code']} with {c['want_n']}")
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
            "text": "inputs.text names bytes in this file's blobs by sha256; each leg reads them decoded as UTF-8 "
                    "with CRLF and a lone CR read as LF, except reading, which is handed the bytes",
            "select": "styxx.prereg_ci.core._select_fence(text, info): {block}, {none: true}, or {raises, message}",
            "protocol": "gates only: styxx.protocol._select_gates_block(text), its refusal 'prereg has no ```gates "
                        "block ...' read as {none: true}; equal to select on every vector",
            "fence_json": "result and locks: styxx.prereg_ci.core._fence_json(text, info): {line, object}, "
                          "{none: true}, or {raises, message}",
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
    # every byte of this file is the generator's own (no carried history), so all of it is scanned
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
