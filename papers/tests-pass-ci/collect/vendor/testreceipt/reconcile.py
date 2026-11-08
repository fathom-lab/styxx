"""A pull request's claims about its tests, checked against what its tests did.

| Verdict | Meaning |
|---|---|
| CONTRADICTED | It says the tests pass; the tests failed at the same commit |
| SCOPED | It says some tests pass (a package, the new tests, a platform); the suite failed |
| COUNT | It says N tests pass; the tests failed, and N may or may not be the whole suite |
| UNSTATED | It lists test commands without saying how they went; the tests failed |
| CONSISTENT | It says the tests pass, and they did |
| UNVERIFIED | It says the tests pass; there is no test result to check it against |
| REPORTS FAILURES | It says tests fail |
| NO CLAIM | It says nothing about test results |

A test run's total (from `--run`) settles COUNT: N at least the total is CONTRADICTED, fewer is
SCOPED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .claims import Claim, claims


@dataclass
class Reconciled:
    verdict: str
    message: str
    claims: list[Claim] = field(default_factory=list)
    evidence: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "message": self.message,
            "evidence": self.evidence,
            "claims": [
                {
                    "kind": c.kind,
                    "scope": c.scope,
                    "why_part": c.why_part,
                    "count": c.count,
                    "text": c.text,
                }
                for c in self.claims
            ],
        }


def reconcile(
    description: str, outcome: str, evidence: str = "", suite_total: int | None = None
) -> Reconciled:
    """`outcome` is what the tests did: "tests failed", "tests passed", "no test result" or
    "no CI" (see `ci.CiResult.tests`)."""
    found = claims(description)
    passes = [c for c in found if c.kind == "pass" and c.explicit]
    implied = [c for c in found if c.kind == "pass" and not c.explicit]
    fails = [c for c in found if c.kind == "fail"]
    if fails and not passes:
        return Reconciled(
            "REPORTS FAILURES", "the description reports failing tests", found, evidence
        )
    if not passes and not implied:
        return Reconciled("NO CLAIM", "the description says nothing about test results", found)
    shown = passes or implied
    first = shown[0].text.strip()
    if outcome == "tests passed":
        return Reconciled(
            "CONSISTENT", f"claims {_quote(first)}; the tests passed", found, evidence
        )
    if outcome != "tests failed":
        return Reconciled(
            "UNVERIFIED", f"claims {_quote(first)}; no test result to check it against", found
        )
    if not passes:
        message = f"lists {_quote(first)} without a result; the tests failed"
        return Reconciled("UNSTATED", message, found, evidence)
    full = [c for c in passes if c.scope == "full"]
    counted = [c for c in passes if c.scope == "count"]
    if suite_total is not None:
        full += [c for c in counted if c.count is not None and c.count >= suite_total]
        counted = [c for c in counted if c not in full]
    if full:
        message = f"claims {_quote(full[0].text.strip())}; the tests failed at the same commit"
        return Reconciled("CONTRADICTED", message, found, evidence)
    if counted and suite_total is None:
        claim = counted[0]
        message = (
            f"claims {_quote(claim.text.strip())}; the tests failed, and whether {claim.count} "
            "is the whole suite is not known"
        )
        return Reconciled("COUNT", message, found, evidence)
    part = (counted or passes)[0]
    if part.scope == "count" and suite_total:
        narrowed = f"{part.count} of {suite_total} tests"
    else:
        narrowed = part.why_part or "some of the tests"
    message = f"claims {_quote(part.text.strip())}, which covers only {narrowed}; the suite failed"
    return Reconciled("SCOPED", message, found, evidence)


def _quote(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return "“" + (text if len(text) <= limit else text[: limit - 1] + "…") + "”"
