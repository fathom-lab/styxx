"""SCOPE-1's footprint on BENCH-2's 299 `only_touches` claims, before any change ships.

Committed in AMENDMENT A of `PREREG_scope1_the_empty_scope_2026_09_18.md`, which was frozen
(sha256 445d4349..., amended 5b6986ee...) before this ran. The question it answers is the one that
decides whether SCOPE-1 may ship at all:

    of the accusations the shipped instrument makes on `only_touches`, which ones have NO changed
    path inside the claimed scope — and are they anything other than the six items the SCOPE-1
    rule was read off in the first place?

Nothing here re-implements a verdict or a containment test. `gate_diff_text`, `_path_inside`,
`_norm` and `_prefix_is_path_shaped` are imported from the shipped instrument, so the emptiness
question is asked in exactly the terms the accusation was made in.

    python papers/closed-model-frontier/scope1_footprint.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.diffgate import (                                   # noqa: E402
    gate_diff_text, parse_unified_diff, _path_inside, _norm, _prefix_is_path_shaped,
)

DIFFS = HERE / "_diffs"
DATASET = HERE / "bench2_dataset.jsonl"
AUDIT = HERE / "bench2_audit.json"
OUT = HERE / "scope1_footprint.json"

# The six items the rule was read off, by PR url. Listed here so the answer to "is the footprint
# larger than the derivation set" is computed against a set fixed in the preregistration, not one
# assembled after seeing the output.
DERIVED_FROM = {
    "https://github.com/Albeoris/Memoria/pull/1142",
    "https://github.com/Albeoris/Memoria/pull/1145",
    "https://github.com/Albeoris/Memoria/pull/1147",
    "https://github.com/mikepenz/release-changelog-builder-action/pull/1458",
    "https://github.com/open-policy-agent/cert-controller/pull/415",
    "https://github.com/fern-api/fern/pull/9898",
}
KNOWN_CORRECT = {
    "https://github.com/Azure/autorest.typescript/pull/3252",
    "https://github.com/microsoft/wassette/pull/442",
}


def rows() -> list[dict]:
    out = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith('{"_header"'):
            continue
        out.append(json.loads(line))
    return out


def scopes(detail: dict, status) -> list[str]:
    """The prefixes the instrument itself would compare against, by its own admission tests."""
    prefs = [_norm(detail["prefix"]).rstrip("/.")]
    if detail.get("prefix2"):
        prefs.append(_norm(detail["prefix2"]).rstrip("/."))
        if not _prefix_is_path_shaped(detail["prefix2"], status):
            prefs = prefs[:1]
    return prefs


def main() -> int:
    missing = 0
    verdicts = {"VERIFIED": 0, "CONTRADICTED": 0, "UNCHECKABLE": 0, "(none)": 0}
    accusations = []
    for r in rows():
        if r["kind"] != "only_touches":
            continue
        p = DIFFS / f"{r['pr_id']}.diff"
        if not p.exists():
            missing += 1
            continue
        diff = p.read_text(encoding="utf-8", errors="replace")
        g = gate_diff_text(r["claim_text"], diff, run=None, strict=False)
        hit = next((c for c in g.claims if c.kind == "only_touches"), None)
        v = hit.verdict if hit else "(none)"
        verdicts[v] = verdicts.get(v, 0) + 1
        if v != "CONTRADICTED":
            continue
        status, _ = parse_unified_diff(diff)
        prefs = scopes(r.get("claim_detail") or {}, status)
        inside = [q for q in status if any(_path_inside(q, x) for x in prefs)]
        outside = [q for q in status if not any(_path_inside(q, x) for x in prefs)]
        accusations.append({
            "url": r["url"],
            "claim": r["claim_text"].strip(),
            "scopes": prefs,
            "changed_paths": len(status),
            "inside": len(inside),
            "outside": len(outside),
            "inside_examples": inside[:3],
            "why": hit.why,
            "scope1_suppresses": not inside,
            "in_derivation_set": r["url"] in DERIVED_FROM,
            "known_correct": r["url"] in KNOWN_CORRECT,
        })

    suppressed = [a for a in accusations if a["scope1_suppresses"]]
    kept = [a for a in accusations if not a["scope1_suppresses"]]
    novel = [a for a in suppressed if not a["in_derivation_set"]]

    report = {
        "prereg": "PREREG_scope1_the_empty_scope_2026_09_18.md",
        "prereg_sha256_frozen": "445d43493a82e0e70378dd01c284f2a2cdde7b64040b2b56296e3f4b4f985629",
        "only_touches_rows": sum(1 for r in rows() if r["kind"] == "only_touches"),
        "diffs_missing": missing,
        "verdicts": verdicts,
        "accusations": len(accusations),
        "scope1_would_suppress": len(suppressed),
        "scope1_would_keep": len(kept),
        "suppressed_outside_derivation_set": len(novel),
        "kept_are_exactly_the_two_known_correct":
            sorted(a["url"] for a in kept) == sorted(KNOWN_CORRECT),
        "detail": accusations,
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"only_touches rows        {report['only_touches_rows']}   (diffs missing: {missing})")
    print(f"verdicts                 {verdicts}")
    print(f"accusations              {len(accusations)}")
    print(f"SCOPE-1 would suppress   {len(suppressed)}")
    print(f"SCOPE-1 would keep       {len(kept)}")
    print(f"suppressed NOT in the derivation set   {len(novel)}   <-- the whole question")
    for a in accusations:
        tag = "SUPPRESS" if a["scope1_suppresses"] else "keep    "
        src = "derived" if a["in_derivation_set"] else ("known-correct" if a["known_correct"] else "NEW")
        print(f"  {tag}  inside={a['inside']:3d} outside={a['outside']:3d}  [{src:13s}] {a['url']}")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
