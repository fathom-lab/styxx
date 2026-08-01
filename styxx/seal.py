# -*- coding: utf-8 -*-
"""styxx.seal — the trust seal for agent work.

Generation is commoditizing; verification is the scarce primitive of the agent economy. An
agent hands over a deliverable and the receiver either trusts it or spends hours checking.
The seal makes that check one call, and portable: it composes the program's shipped
verifiers over one document and either SEALS it or REFUSES with the failing claim named.

What one ``seal()`` runs:

  1. **OATH** (`styxx.certify`) — every numeric claim in the document verified against the
     receipt JSONs; one UNGROUNDED refuses the seal.
  2. **Protocol** (`styxx.protocol`) — for each referenced prereg, the freeze is checked
     against git history and the result dict is re-scored through the FROZEN gates block;
     a verdict mismatch between what the document claims and what the frozen table returns
     refuses the seal. (No prereg references → this layer records "none declared".)
  3. **Attestation** (`styxx.attestation`) — the composite (document sha, OATH counts,
     protocol verdicts, gates hashes, prereg commits) is bound into a machine-checkable
     seal object with its own content hash, re-derivable by ``verify_seal``.

The seal never verifies prose truth — that boundary is the OATH's and it is inherited
loudly: numeric claims, receipt grounding, frozen-gate verdicts, nothing more. A document
with zero checkable claims seals VACUOUSLY and says so on the seal.

CLI::

    python -m styxx.seal DOC.md receipt1.json [receipt2.json ...] \
        [--prereg PREREG.md=RESULT.json ...] [--out SEAL.json]

Exit 0 = SEALED (or VACUOUS), 1 = REFUSED — a drop-in CI gate for agent deliverables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .certify import certify_doc
from .protocol import Experiment, PrologueError, GateSpecError

__all__ = ["seal", "verify_seal", "Seal"]

_SEAL_VERSION = "seal/v0"


@dataclass
class Seal:
    verdict: str                    # SEALED | SEALED_VACUOUS | REFUSED
    document: str
    document_sha256: str
    oath: dict                      # counts + verdict from certify
    protocol: list = field(default_factory=list)   # per-prereg verdict records
    refusals: list = field(default_factory=list)   # the named failures
    seal_sha256: str = ""

    def to_dict(self) -> dict:
        d = {"seal": _SEAL_VERSION, "verdict": self.verdict, "document": self.document,
             "document_sha256": self.document_sha256, "oath": self.oath,
             "protocol": self.protocol, "refusals": self.refusals}
        d["seal_sha256"] = hashlib.sha256(
            json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
        return d


def seal(doc_path: str | Path, receipts: list[str | Path],
         preregs: list[tuple[str | Path, str | Path]] | None = None) -> Seal:
    """Seal one document: OATH over its numeric claims + protocol over its preregs.

    ``preregs`` is a list of (prereg_path, result_json_path) pairs: each result is
    re-scored through its prereg's FROZEN gates block, and the verdict recorded there
    must appear verbatim in the result file's own ``verdict`` field (the document's
    claimed verdict is the result file's — a mismatch is a refusal).
    """
    doc_path = Path(doc_path)
    cert = certify_doc(doc_path, [Path(r) for r in receipts])
    refusals = []
    for u in cert["ungrounded"]:
        refusals.append({"layer": "oath", "line": u["line"], "token": u["token"],
                         "why": "numeric claim grounds in no provided receipt"})

    protocol_records = []
    for prereg_path, result_path in (preregs or []):
        rec = {"prereg": str(prereg_path), "result": str(result_path)}
        try:
            exp = Experiment(prereg_path)
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
            v = exp.score(result, smoke=bool(result.get("smoke")))
            rec.update({"frozen_verdict": v.verdict, "gates": v.gates,
                        "gates_sha256": v.gates_sha256, "prereg_commit": v.prereg_commit})
            claimed = result.get("verdict")
            if claimed != v.verdict:
                refusals.append({"layer": "protocol", "prereg": str(prereg_path),
                                 "why": f"claimed verdict {claimed!r} != frozen-table "
                                        f"verdict {v.verdict!r}"})
        except (PrologueError, GateSpecError, OSError, json.JSONDecodeError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            refusals.append({"layer": "protocol", "prereg": str(prereg_path),
                             "why": rec["error"]})
        protocol_records.append(rec)

    checkable = sum(cert["counts"].values()) + len(protocol_records)
    if refusals:
        verdict = "REFUSED"
    elif checkable == 0:
        verdict = "SEALED_VACUOUS"      # nothing checkable — the seal says so, loudly
    else:
        verdict = "SEALED"
    return Seal(verdict=verdict, document=doc_path.name,
                document_sha256=cert["document_sha256"],
                oath={"counts": cert["counts"], "verdict": cert["verdict"]},
                protocol=protocol_records, refusals=refusals)


def verify_seal(seal_dict: dict) -> bool:
    """Re-derive the seal's content hash; True iff untampered."""
    d = dict(seal_dict)
    claimed = d.pop("seal_sha256", "")
    return hashlib.sha256(
        json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest() == claimed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.seal",
                                 description="Seal an agent deliverable or refuse it.")
    ap.add_argument("doc")
    ap.add_argument("receipts", nargs="*")
    ap.add_argument("--prereg", action="append", default=[],
                    metavar="PREREG.md=RESULT.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pairs = []
    for spec in a.prereg:
        if "=" not in spec:
            print(f"--prereg wants PREREG.md=RESULT.json, got {spec!r}", file=sys.stderr)
            return 2
        p, r = spec.split("=", 1)
        pairs.append((p, r))
    s = seal(a.doc, a.receipts, pairs)
    payload = s.to_dict()
    out = Path(a.out) if a.out else Path(a.doc).with_suffix(".seal.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{s.verdict}  oath={s.oath['counts']}  protocol={len(s.protocol)} "
          f"refusals={len(s.refusals)}  -> {out.name}")
    for r in s.refusals:
        print(f"  [REFUSED:{r['layer']}] {r.get('why')}")
    return 0 if s.verdict.startswith("SEALED") else 1


if __name__ == "__main__":
    sys.exit(main())
