"""differential.py — do the two sworn verifiers agree on inputs nobody chose?

SPEC: ``papers/sworn/SPEC_differential_agreement_v01_2026_09_05.md``, frozen before this file.

The conformance set asks whether a second implementation agrees where the lab looked. Every one of
its 1689 vectors was recorded from a call some author wrote, and the JavaScript was repaired five
times against those very vectors until it matched. This asks the other question: a seeded grammar
composes documents out of the format's decision boundaries, both shipped implementations verify
each one, and the two verdict core digests are compared byte for byte.

Neither side is instrumented (D1). The generator never consults either implementation (D2). Every
case is a pure function of (seed, index), so a disagreement is reproducible from two integers (D3).

    python conformance/sworn/differential.py --cases 100000 --seed 20260905 --out RESULT.json

Node is spawned once per BATCH, not once per case, because a process launch costs more than a
thousand verifications.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx import sworn                                  # noqa: E402
from styxx.attestation import jcs                        # noqa: E402

JS = ROOT / "styxx" / "_data" / "sworn_verify.js"
OUTSIDE_CORE = ("verifier", "coverage")


# ============================================================ the grammar (D2)

KINDS = (["numeric"] * 6 + ["quote"] * 6 + ["hash"] * 4 + ["absent"] * 4
         + ["exec", "Numeric", "", "numeri c"])          # invalid kinds stay reachable, rare
SOURCE_KINDS = (["tool_stdout"] * 4 + ["test_report"] * 4 + ["file_read"] * 3
                + ["tool_stderr", "http_fetch", "harness_note", "attestation"]
                + ["agent_output", "agent_file_write", "agent_message", "made_up_kind", ""])
RUNGS = ["L1"] * 5 + ["L2"] * 4 + [None] * 2 + ["L3", "L0", "l1", 7]

NUMBERS = [
    "0", "1", "-1", "+1", "0.5", "0.50", ".5", "5.", "1,234", "1,234.5", "12,34",
    "1e5", "1E5", "1e-5", "0.0000001", "100%", "-0.0", "+0.0", "0.4211", "0.421", "0.42",
    "23,247.", "−0.0154", "٣.٥", "1_000", "0x1f", "1.2.3", "9" * 40, "1" + "0" * 330,
    "3.14159265358979", "0.1", "0.2", "0.3", "2.675", "1.005", "0.615",
    # the values the manifest payloads actually carry, so the HELD path of every kind is
    # reachable and the decimal canonicaliser is compared on agreement as well as on failure
    "12", "12.0", "12.00", "1.2e1", "0.4211", "0.42110", "1", "2", "3", "-0.0", "100000",
]
WORDS = ["the", "battery", "passed", "checks", "precision", "of", "a", "bar", "n=", "and",
         "recall", "over", "seeds", "rate", "note", "value", "score"]
NEEDLES = (["the harness wrote this"] * 4 + ["STRUCT-1 = 16/38 (A-share 0.4211)"] * 3
           + ["sixteen bytes!!"] * 3 + ["line one\nline two"] * 2 + ["plain bytes, not json"] * 2
           + ["a needle long enough to count", "seventeen bytes!!", "not present anywhere here",
              "ok", "passed", "x", "", "   ", "All checks passed!", "nan", "éééééééé",
              # --- APERTURE_PAYLOAD: a needle that overlaps itself distinguishes an overlapping
              # count from a non-overlapping one; nothing here did before.
              "aa", "abab", "aaa", "one\r\ntwo", "one\rtwo", "\U0001f600", "e\u0301"])
HEXES = ["a" * 64, "0" * 64, "deadbeef" * 8, "A" * 64, "b" * 32, "c" * 40, "d" * 96, "e" * 128,
         "f" * 63, "1234", ""]
RECEIPTS = (["r1"] * 10 + ["r2"] * 4 + ["r1#/passed"] * 6 + ["r1#/x"] * 4 + ["r1#/y"] * 3
            + ["r1#L1"] * 3 + ["r1#L1-L3"] * 2
            # --- APERTURE_PAYLOAD, the receipt grammar half. A slice that starts past line 1, a
            # pointer through an escaped key, and an index at an array's last element were all
            # absent, so the code that handles each of them was compared against nothing.
            + ["r1#L2"] * 2 + ["r1#L2-L4"] * 2 + ["r1#/a~0b", "r1#/c~1d", "r1#/~1", "r1#/~0",
                                                  "r1#/a/b/2", "r1#/a/b/0", "r1#/e/0", "r1#/o",
                                                  "r1#/p", "r1#/t1", "r1#/tiny", "r1#/nan",
                                                  "prereg:" + "A" * 64]
            + ["r1#/a/b", "r1#/0", "r1#/missing", "r1#L9",
                                                  "r1#/n", "r1#/t", "r1#/s", "r1#/big", "r1#/neg",
                                                  "r1#/e", "r1#/dup", "r2#/passed", "r3", "r10"]
            # the grammar's minority: forms that cannot resolve or cannot parse
            + ["r0", "r01", "R1", "r", "r1#L0", "r1#", "r1#/~0", "r1#/~2",
               "path:a.json", "path:a.json#/x", "path:/a.json", "path::a.json", "path:a b.json",
               "path:../a.json", "path:a/../b.json", "path:", "prereg:" + "a" * 64, "prereg:zz",
               "", "nonsense"])

FILLERS = ["", " ", "\n", "\r\n", "\n\n", " \t ", "\r"]
FENCES = ["```", "~~~", "````", "   ```", "    ```"]


def _rng(seed: int, index: int) -> random.Random:
    """A generator is a pure function of (seed, index) — D3."""
    return random.Random("%d:%d" % (seed, index))


def _number_sentence(r: random.Random) -> str:
    n = r.choice(NUMBERS)
    body = " ".join(r.choice(WORDS) for _ in range(r.randint(0, 4)))
    tail = r.choice(["", ".", ",", ":", "%", " checks", " of 38"])
    parts = [body, n + tail] if body else [n + tail]
    if r.random() < 0.15:                       # a second digit-bearing token: number_count
        parts.append(r.choice(NUMBERS))
    r.shuffle(parts)
    return " ".join(p for p in parts if p)


def _quote_sentence(r: random.Random) -> str:
    needle = r.choice(NEEDLES)
    ticks = "`" * r.choice([1, 1, 1, 2, 3])
    n_spans = r.choice([1, 1, 1, 0, 2])
    body = " ".join(r.choice(WORDS) for _ in range(r.randint(0, 3)))
    spans = " ".join(ticks + needle + ticks for _ in range(n_spans))
    return (body + " " + spans).strip()


def _hash_sentence(r: random.Random) -> str:
    h = r.choice(HEXES)
    extra = r.choice(["", " " + r.choice(HEXES), " " + r.choice(WORDS)])
    return ("it hashes to " + h + extra).strip()


def _inner_for(kind: str, r: random.Random) -> str:
    if kind == "numeric":
        return _number_sentence(r)
    if kind in ("quote", "absent"):
        return _quote_sentence(r)
    if kind == "hash":
        return _hash_sentence(r)
    return r.choice(["", " ", _number_sentence(r), _quote_sentence(r)])


def _span(r: random.Random) -> str:
    kind = r.choice(KINDS)
    receipt = r.choice(RECEIPTS)
    inner = _inner_for(kind if kind in ("numeric", "quote", "hash", "absent") else "numeric", r)
    if r.random() < 0.02:
        inner = inner * r.randint(20, 60)              # over the code-point cap
    opener = '<sworn r="%s" k="%s">' % (receipt, kind)
    if r.random() < 0.02:                              # near-miss tag shapes
        opener = r.choice([
            '<sworn r="%s" k="%s" >' % (receipt, kind),
            "<sworn r='%s' k='%s'>" % (receipt, kind),
            '<SWORN r="%s" k="%s">' % (receipt, kind),
            '<swornx r="%s" k="%s">' % (receipt, kind),
            '<sworn  r="%s" k="%s">' % (receipt, kind),
            '<sworn r="%s">' % receipt,
        ])
    closer = "</sworn>" if r.random() > 0.02 else r.choice(["</sworn >", "</SWORN>", "", "</sworn"])
    return opener + inner + closer


def _document(r: random.Random) -> bytes:
    parts = []
    for _ in range(r.randint(1, 6)):
        pick = r.random()
        if pick < 0.72:
            parts.append(_span(r))
        elif pick < 0.82:
            parts.append(" ".join(r.choice(WORDS) for _ in range(r.randint(1, 8))) + ".")
        elif pick < 0.90:
            fence = r.choice(FENCES)
            body = _span(r) if r.random() < 0.5 else "code " + r.choice(NUMBERS)
            parts.append(fence + "\n" + body + "\n" + (fence.strip() if r.random() < 0.8 else ""))
        elif pick < 0.96:
            parts.append("<!--" + (_span(r) if r.random() < 0.5 else "hidden") +
                         ("-->" if r.random() < 0.8 else ""))
        elif pick < 0.98:
            parts.append("</sworn>")                    # stray closer
        else:
            parts.append(_span(r).replace("</sworn>", _span(r) + "</sworn>"))   # nesting
        parts.append(r.choice(FILLERS))
    text = "".join(parts)
    raw = text.encode("utf-8")
    hazard = r.random()
    if hazard < 0.01:
        raw = b"\xef\xbb\xbf" + raw                     # BOM
    elif hazard < 0.02:
        raw = raw + b"\xff\xfe"                         # invalid UTF-8
    elif hazard < 0.03:
        raw = raw.replace(b"\n", b"\r\n")
    elif hazard < 0.035:
        raw = raw + b"\x00"
    return raw


def _manifest(r: random.Random) -> dict | None:
    if r.random() < 0.12:
        return None
    receipts = {}
    for i in range(1, r.randint(2, 5)):
        rid = "r%d" % i
        payload = r.choice([
            b'{"passed": 12, "note": "the harness wrote this"}\n',
            # --- APERTURE_PAYLOAD. The strict JSON parser is reached ONLY through these bytes, so
            # every parser behaviour absent from this list was invisible at any case count.
            b'\xef\xbb\xbf{"passed": 12}\n',                   # a BOM the reader must refuse
            b'{"nan": NaN, "inf": Infinity, "ninf": -Infinity}\n',
            # values sitting exactly on a rounding tie at the printed precision: without one of
            # these, half-even and half-up are the same function as far as this harness can tell
            b'{"t1": 0.25, "t2": 1.5, "t3": 2.5, "t4": 0.05, "t5": -0.05, "t6": 2.675}\n',
            b'{"tiny": 1e-7, "tinier": 1e-320, "huge": 1e320, "edge": 0.00001}\n',
            b'{"s": "\\ud83d\\ude00", "lone": "\\ud800", "pair": "a\\ud83d\\ude00b"}\n',
            '{"astral": "\U0001f600\U0001f4a1", "combining": "e\u0301"}\n'.encode("utf-8"),
            b'{"a~b": 1, "c/d": 2, "~1": 3, "~0": 4, "a~1b": 5}\n',   # pointer-escape keys
            b'{"o": "aaaa", "p": "abababab", "q": "aaa"}\n',      # needles that overlap themselves
            b'{"a": {"b": {"c": {"d": {"e": [1, [2, [3, [4]]]]}}}}}\n',
            b'{"bad": "\xff\xfe not utf-8"}\n',                   # invalid UTF-8 inside a receipt
            b'{"HEX": "DEADBEEF' + b"0" * 56 + b'"}\n',           # uppercase hex to fold
            b'{"crlf": "one\r\ntwo", "cr": "one\rtwo", "tab": "a\tb"}\n',
            b'line one\nline two\nline three\nline four\nline five\n',
            b'{"passed": 12.0, "a": {"b": [1, 2, 3]}}\n',
            b'{"x": 0.4211, "y": "STRUCT-1 = 16/38 (A-share 0.4211)"}\n',
            b'{"n": null, "t": true, "s": "sixteen bytes!!"}\n',
            b"plain bytes, not json at all\n",
            b'{"dup": 1, "dup": 2}\n',
            b'{"big": ' + b"9" * 400 + b'}\n',
            b'{"neg": -0.0, "e": 1e5}\n',
            b'line one\nline two\nline three\n',
            b'',
        ])
        entry = {"id": rid, "sha256": hashlib.sha256(payload).hexdigest(),
                 "kind_of_source": r.choice(SOURCE_KINDS),
                 # one constant timestamp meant every timestamp path was compared on one value
                 "captured_at": r.choice(["2026-09-01T00:00:00Z", "2026-09-01T00:00:00.500Z",
                                          "2026-09-01T00:00:00+02:00", "2026-09-01T00:00:60Z",
                                          "not a timestamp", ""])}
        if r.random() < 0.06:
            entry["sha256"] = entry["sha256"].upper()   # a digest the reader must case-fold
        if r.random() > 0.08:
            entry["complete"] = r.choice([True, True, True, False])
        if r.random() > 0.04:
            entry["bytes"] = base64.b64encode(payload).decode("ascii")
        if r.random() < 0.02:
            entry["sha256"] = "0" * 64                  # integrity break
        if r.random() < 0.02:
            entry = r.choice([5, "not an entry", None, []])
        receipts[rid] = entry
    m = {"spec": r.choice(["sworn/manifest/0.2", "sworn/manifest/0.2", "sworn/manifest/0.1"]),
         # a JCS string escape is only observable on a string that needs escaping
         "harness": r.choice(["pytest", "", "harness with é", "two\nlines", 'quote"inside',
                              "back\\slash", "tab\there", "\U0001f600"]),
         "turn": r.choice(["t", "turn-1", ""]),
         "minted_at": "2026-09-01T00:00:00Z",
         "authored_sha256": [], "receipts": receipts}
    if m["spec"] == "sworn/manifest/0.2":
        m["rung"] = r.choice(RUNGS)
    if r.random() < 0.04:                               # the agent swearing to itself
        for e in receipts.values():
            if isinstance(e, dict) and "bytes" in e:
                m["authored_sha256"] = [e["sha256"]]
                break
    # a one-element list can never show an ordering bug, and a lowercase-only list can never show a
    # case-folding one; both were true of every manifest the old grammar built
    if r.random() < 0.10:
        digs = [e["sha256"] for e in receipts.values() if isinstance(e, dict) and "sha256" in e]
        if len(digs) > 1:
            m["authored_sha256"] = list(reversed(digs))
        if digs and r.random() < 0.5:
            m["authored_sha256"] = [d.upper() for d in m["authored_sha256"]]
    if r.random() < 0.03:
        m["digest"] = "0" * 64                          # a digest that does not re-derive
    else:
        try:
            core = {k: v for k, v in m.items() if k != "digest"}
            m["digest"] = hashlib.sha256(jcs(core).encode("utf-8")).hexdigest()
        except Exception:                               # noqa: BLE001
            pass
    return m


def case(seed: int, index: int) -> dict:
    r = _rng(seed, index)
    return {"index": index, "document": _document(r), "manifest": _manifest(r),
            "name": r.choice(["d.md", "", "a name with é.md"]),
            "commit": r.choice([None, None, "a" * 40, "b" * 64])}


# ============================================================ the two sides (D1)

def python_digest(c: dict):
    """(digest, error, census) — the shipped verifier, uninstrumented."""
    try:
        man = sworn.Manifest.from_dict(c["manifest"]) if c["manifest"] is not None else None
    except SystemExit as e:
        return None, "manifest-refused: %s" % type(e).__name__, {}
    except Exception as e:                               # noqa: BLE001
        return None, "manifest-%s" % type(e).__name__, {}
    try:
        core = sworn.verify(c["document"], name=c["name"], manifest=man, commit=c["commit"])
    except SystemExit as e:
        return None, "refused: %s" % str(e)[:80], {}
    except Exception as e:                               # noqa: BLE001
        return None, type(e).__name__, {}
    census = {"document_verdict": core["document_verdict"],
              "counts": dict(core["counts"]),
              "reasons": [s.get("reason") for s in core["spans"] if s.get("reason")],
              "kinds": [s.get("kind") for s in core["spans"]],
              "document_malformed": (core.get("document_malformed") or {}).get("reason")}
    body = {k: v for k, v in core.items() if k not in OUTSIDE_CORE}
    try:
        return hashlib.sha256(jcs(body).encode("utf-8")).hexdigest(), None, census
    except Exception as e:                               # noqa: BLE001
        return None, "jcs-%s" % type(e).__name__, census


_NODE_RUNNER = r"""
const fs = require('fs');
const api = require(process.argv[2]);
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = [];
for (const c of cases) {
  let digest = null, error = null;
  try {
    const doc = new Uint8Array(Buffer.from(c.document_b64, 'base64'));
    const man = c.manifest === null ? null : api.jsonPlain(JSON.stringify(c.manifest));
    const core = api.swornVerify(doc, man, { name: c.name, commit: c.commit });
    digest = api.coreDigest(core);
  } catch (e) {
    error = (e && e.name ? e.name : 'Error') + ': ' + String(e && e.message ? e.message : e).slice(0, 90);
  }
  out.push({ index: c.index, digest, error });
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def js_digests(batch: list, workdir: Path) -> dict:
    """One node process per batch — a launch costs more than a thousand verifications."""
    runner = workdir / "runner.js"
    runner.write_bytes(_NODE_RUNNER.encode("utf-8"))
    payload = [{"index": c["index"],
                "document_b64": base64.b64encode(c["document"]).decode("ascii"),
                "manifest": c["manifest"], "name": c["name"], "commit": c["commit"]}
               for c in batch]
    inp = workdir / "in.json"
    outp = workdir / "out.json"
    inp.write_bytes(json.dumps(payload).encode("utf-8"))
    r = subprocess.run(["node", str(runner), str(JS), str(inp), str(outp)],
                       capture_output=True, timeout=900)
    if r.returncode != 0 or not outp.exists():
        raise SystemExit("REFUSED: the node side did not run: %s"
                         % r.stderr.decode("utf-8", "replace")[-400:])
    return {row["index"]: row for row in json.loads(outp.read_text(encoding="utf-8"))}


# ============================================================ the run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "differential_agreement.json"))
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    if out.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(out)],
                           capture_output=True)
        if r.returncode == 0:
            print("REFUSED: %s is tracked; a run is history — write a new file (D6)" % out.name,
                  file=sys.stderr)
            return 2

    agree = disagree = 0
    py_err = js_err = both_err = 0
    verdicts, reasons, kinds, doc_malformed = Counter(), Counter(), Counter(), Counter()
    span_counts = Counter()
    disagreements = []
    error_pairs = Counter()

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        done = 0
        while done < a.cases:
            n = min(a.batch, a.cases - done)
            batch = [case(a.seed, done + i) for i in range(n)]
            js = js_digests(batch, work)
            for c in batch:
                pd, pe, census = python_digest(c)
                row = js.get(c["index"], {})
                jd, je = row.get("digest"), row.get("error")
                if census:
                    verdicts[census["document_verdict"]] += 1
                    for k, v in census["counts"].items():
                        span_counts[k] += v
                    for why in census["reasons"]:
                        reasons[why] += 1
                    for k in census["kinds"]:
                        kinds[str(k)] += 1
                    if census["document_malformed"]:
                        doc_malformed[census["document_malformed"]] += 1
                if pe and je:
                    both_err += 1
                    error_pairs[(pe.split(":")[0], je.split(":")[0])] += 1
                    continue
                if pe or je:
                    if pe:
                        py_err += 1
                    else:
                        js_err += 1
                    disagreements.append({
                        "seed": a.seed, "index": c["index"], "why": "one side raised",
                        "python": {"digest": pd, "error": pe},
                        "javascript": {"digest": jd, "error": je},
                        "document_b64": base64.b64encode(c["document"]).decode("ascii"),
                        "manifest": c["manifest"], "name": c["name"], "commit": c["commit"]})
                    continue
                if pd == jd:
                    agree += 1
                else:
                    disagree += 1
                    disagreements.append({
                        "seed": a.seed, "index": c["index"], "why": "the digests differ",
                        "python": {"digest": pd}, "javascript": {"digest": jd},
                        "document_b64": base64.b64encode(c["document"]).decode("ascii"),
                        "manifest": c["manifest"], "name": c["name"], "commit": c["commit"]})
            done += n
            print("  %d/%d compared, %d agree, %d disagree, %d one-sided errors"
                  % (done, a.cases, agree, disagree, py_err + js_err), file=sys.stderr)

    compared = agree + disagree
    gates = {
        "G-N": {"quantity": "cases compared", "value": compared, "bar": ">= 100000",
                "pass": compared >= 100000},
        "G-A": {"quantity": "cases whose core digests are equal", "value": agree,
                "share": round(agree / compared, 6) if compared else None,
                "disagreements": disagree},
        "G-C": {"quantity": "verdict vocabulary reached",
                "value": {k: span_counts.get(k, 0) for k in
                          ("HELD", "FAILED", "UNRESOLVED", "MALFORMED", "WITHHELD")},
                "document_malformed": sum(doc_malformed.values()),
                "pass": all(span_counts.get(k, 0) > 0
                            for k in ("HELD", "FAILED", "UNRESOLVED", "MALFORMED"))
                        and sum(doc_malformed.values()) > 0},
        "G-R": {"quantity": "distinct MALFORMED reasons exercised", "value": len(reasons),
                "bar": ">= 12", "pass": len(reasons) >= 12},
        "G-E": {"quantity": "cases where either side raised",
                "python_only": py_err, "javascript_only": js_err, "both": both_err,
                "pairs": {"%s | %s" % k: v for k, v in error_pairs.most_common(12)}},
    }
    void = not gates["G-C"]["pass"]
    result = {
        "schema": "styxx.sworn.differential-agreement/v1",
        "spec": "papers/sworn/SPEC_differential_agreement_v01_2026_09_05.md",
        "seed": a.seed, "cases_requested": a.cases,
        "implementations": {
            "python": {"module": "styxx/sworn.py",
                       "sha256": hashlib.sha256(
                           (ROOT / "styxx" / "sworn.py").read_bytes().replace(b"\r\n", b"\n")
                       ).hexdigest()},
            "javascript": {"module": "styxx/_data/sworn_verify.js",
                           "sha256": hashlib.sha256(
                               JS.read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
            "note": "content identity modulo newlines, the corpus doctrine",
        },
        "compared": compared, "agree": agree, "disagree": disagree,
        "void": void,
        "gates": gates,
        "census": {"document_verdicts": dict(verdicts), "span_verdicts": dict(span_counts),
                   "malformed_reasons": dict(reasons.most_common()),
                   "document_malformed": dict(doc_malformed), "kinds": dict(kinds.most_common())},
        "disagreements": disagreements[:50],
        "disagreements_total": len(disagreements),
        "reading": ("agreement here is agreement on inputs nobody chose, which is strictly more "
                    "than agreement on the conformance vectors and strictly less than "
                    "correctness: both implementations may be wrong in the same way, and the same "
                    "hands wrote both."),
    }
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("compared %d | agree %d | disagree %d | one-sided errors %d | reasons %d%s"
          % (compared, agree, disagree, py_err + js_err, len(reasons),
             " | VOID (vocabulary not reached)" if void else ""))
    print("-> %s" % out)
    return 1 if (disagree or py_err or js_err) else 0


if __name__ == "__main__":
    sys.exit(main())
