# -*- coding: utf-8 -*-
"""styxx.attestation — Verifiable Cognometric Attestation.

Binds three existing styxx primitives into one content-addressed artifact a
THIRD PARTY can re-verify against the substrate without trusting the agent
that produced it:

  1. ``agent_audit.extract_claims`` — deterministic checkable claims from an
     agent's free-text self-report (no LLM; closed regex template set).
  2. ``agent_audit.AgentClaimAuditor`` — verifies each claim against the
     substrate (a git repo), producing PASS / FAIL / ERROR verdicts.
  3. ``styxx.compliance`` — maps the audit evidence onto EU AI Act Article 15
     clauses, and carries the explicit uncovered-requirements boundary.

The artifact is SELF-DESCRIBING: per claim it stores the checker name, args,
expected value, the verdict, and the substrate evidence — plus a SHA-256
digest over the canonical payload. ``verify_attestation`` re-runs each
checker against the repo and compares the RE-DERIVED verdict to the embedded
one; it never reads the embedded verdict as truth. Trust the substrate, not
the agent.

Honest boundary (pre-registered in scripts/dogfood/PREREG_verifiable_attestation.md):
an attestation proves only substrate-checkable claims. Interpretive, causal,
and generalization claims are out of scope and reported as uncovered. This is
NOT legal advice and NOT a substitute for an organization's own AI Act
conformity assessment.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .agent_audit import AgentClaimAuditor, Claim, checkers, extract_claims

try:
    from . import __version__ as _STYXX_VERSION
except Exception:  # pragma: no cover - version import is best-effort provenance
    _STYXX_VERSION = "unknown"


__all__ = [
    "attest",
    "verify_attestation",
    "Attestation",
    "VerificationResult",
    "ATTESTATION_VERSION",
]

ATTESTATION_VERSION = "1.0"

# Security boundary: verify_attestation resolves a stored checker name ONLY
# against this allowlist of public, read-only methods on the trusted _Checkers
# singleton. An untrusted artifact can never name an arbitrary attribute to
# call — an unknown name is refused (verdict ERROR), never executed.
_CHECKER_ALLOWLIST: dict[str, Any] = {
    name: getattr(checkers, name)
    for name in dir(checkers)
    if not name.startswith("_") and callable(getattr(checkers, name))
}


@dataclass
class Attestation:
    """The verifiable artifact. ``artifact`` is the JSON-serializable dict."""

    artifact: dict[str, Any]

    @property
    def digest(self) -> str:
        return self.artifact["digest"]["value"]

    @property
    def passed(self) -> bool:
        s = self.artifact["summary"]
        return s["failed"] == 0 and s["errored"] == 0 and s["passed"] > 0

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.artifact, indent=indent, ensure_ascii=False, sort_keys=True)


@dataclass
class VerificationResult:
    """Outcome of independently re-verifying an attestation against a repo."""

    digest_ok: bool
    reproduced: list[dict[str, Any]] = field(default_factory=list)
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    unknown_checkers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff the digest matched AND every embedded verdict reproduced."""
        return self.digest_ok and not self.mismatches and not self.unknown_checkers

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "digest_ok": self.digest_ok,
            "n_reproduced": len(self.reproduced),
            "n_mismatches": len(self.mismatches),
            "mismatches": self.mismatches,
            "unknown_checkers": self.unknown_checkers,
        }


# ---------------------------------------------------------------------------
# Canonicalization + digest. The digest covers everything EXCEPT the volatile
# `generated_at` field and the `digest` field itself. Two attest() runs on an
# identical substrate therefore produce the same digest (the determinism the
# whole protocol rests on).
# ---------------------------------------------------------------------------

def _canonical_payload(artifact: dict[str, Any]) -> str:
    core = {k: v for k, v in artifact.items() if k not in ("generated_at", "digest")}
    return json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_digest(artifact: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(artifact).encode("utf-8")).hexdigest()


def _head_commit(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            capture_output=True, check=False,
        )
        sha = r.stdout.decode("utf-8", errors="replace").strip()
        return sha or "unknown"
    except Exception:  # pragma: no cover - non-git substrate
        return "unknown"


_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _resolve_commit(repo: Path, ref: str) -> str:
    """Resolve a ref to a full commit SHA. Raises on an unknown/invalid ref."""
    r = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=str(repo), capture_output=True, check=False,
    )
    sha = r.stdout.decode("utf-8", errors="replace").strip()
    if not sha:
        raise ValueError(f"unknown git ref in substrate: {ref!r}")
    return sha


@contextmanager
def _materialized_tree(repo: Path, commit_sha: str) -> Iterator[Path]:
    """Yield a throwaway dir holding the repo tree at ``commit_sha``.

    Read-only: uses ``git archive`` (which never touches the working tree or the
    ``.git`` worktree registry) and extracts the tar in-process. ``commit_sha``
    is validated as a hex SHA before it reaches git, so an untrusted artifact
    cannot smuggle an argument (e.g. ``--upload-pack``) through this path.
    """
    if not _SHA_RE.match(commit_sha):
        raise ValueError(f"refusing non-hex commit in substrate: {commit_sha!r}")
    r = subprocess.run(
        ["git", "archive", "--format=tar", commit_sha],
        cwd=str(repo), capture_output=True, check=False,
    )
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git archive failed for {commit_sha[:12]}: {msg}")
    tmp = Path(tempfile.mkdtemp(prefix="styxx_attest_"))
    try:
        with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:") as tar:
            # archive is our own hex-pinned git tree; the `data` filter (3.12+)
            # is a belt-and-suspenders guard against path-escaping members and
            # silences the 3.14 default-filter deprecation.
            if sys.version_info >= (3, 12):
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)  # noqa: S202 — trusted, hex-pinned git tree
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextmanager
def _substrate(repo: Path, ref: str | None) -> Iterator[tuple[Path, str]]:
    """Yield ``(audit_path, commit_sha)`` for the substrate to audit against.

    Unpinned (``ref is None``): the live working tree, paired with the current
    HEAD sha. Pinned: a throwaway materialization of ``ref``'s commit.
    """
    if ref is None:
        yield repo, _head_commit(repo)
    else:
        sha = _resolve_commit(repo, ref)
        with _materialized_tree(repo, sha) as tree:
            yield tree, sha


def _clauses_for_primitive(primitive_prefix: str) -> list[str]:
    """Article 15 clauses whose styxx-primitive list includes this primitive."""
    from .compliance.eu_ai_act import ARTICLE_15_REGISTRY

    return sorted(
        clause
        for clause, m in ARTICLE_15_REGISTRY.items()
        if any(p.primitive.startswith(primitive_prefix) for p in m.styxx_primitives)
    )


def _uncovered_boundary() -> list[dict[str, str]]:
    from .compliance.eu_ai_act import uncovered_requirements

    return [
        {"clause": u.clause, "reason": u.reason, "alternative": u.alternative}
        for u in uncovered_requirements()
    ]


def attest(
    report_text: str,
    repo_path: str | Path,
    *,
    id_prefix: str = "A",
    ref: str | None = None,
) -> Attestation:
    """Produce a Verifiable Cognometric Attestation for an agent self-report.

    Args:
        report_text: the agent's free-text self-report (claims about substrate).
        repo_path: path to the substrate git repository the claims are about.
        id_prefix: claim-id prefix in the artifact (default "A").
        ref: optional git ref (branch, tag, or SHA). When given, the claims are
            verified against the repo tree AT THAT COMMIT — read-only, via
            ``git archive`` — and the resolved SHA is recorded so the
            attestation becomes immutable as-of-date provenance. When ``None``
            (default), the live working tree is used.

    Returns:
        An :class:`Attestation` whose ``.artifact`` is a JSON-serializable,
        content-addressed dict. The verdicts are produced by re-running each
        extracted claim's checker against the substrate — the same mechanical
        audit any third party reproduces via :func:`verify_attestation`.
    """
    repo = Path(repo_path).resolve()
    report = extract_claims(report_text, id_prefix=id_prefix)
    with _substrate(repo, ref) as (audit_path, commit_sha):
        results = AgentClaimAuditor(audit_path).run(report.claims)

    # The audit is itself the `styxx.agent_audit` primitive; every audited
    # claim is evidence under the Article 15 clauses the compliance map already
    # assigns to that primitive. This is an honest, derived mapping — not a
    # per-claim guess.
    audit_clauses = _clauses_for_primitive("styxx.agent_audit")

    by_id = {c.id: c for c in report.claims}
    claim_entries: list[dict[str, Any]] = []
    for r in results:
        c = by_id[r.id]
        claim_entries.append(
            {
                "id": r.id,
                "text": r.text,
                "checker": c.checker.__name__,
                "args": c.args,
                "expected": r.expected,
                "actual": r.actual,
                "verdict": r.verdict,
                "evidence": r.evidence,
                "error": r.error,
                "clauses": audit_clauses,
            }
        )

    passed = sum(1 for r in results if r.verdict == "PASS")
    failed = sum(1 for r in results if r.verdict == "FAIL")
    errored = sum(1 for r in results if r.verdict == "ERROR")

    artifact: dict[str, Any] = {
        "styxx_attestation_version": ATTESTATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": {"styxx_version": _STYXX_VERSION},
        "substrate": {
            "repo": repo.name,
            "commit": commit_sha,
            "pinned_ref": ref,
        },
        "claims": claim_entries,
        "summary": {
            "claims_extracted": len(report.claims),
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "sentences_total": report.sentences_total,
            "sentences_matched": report.sentences_matched,
            "coverage": round(report.coverage, 4),
        },
        "compliance": {
            "framework": "EU AI Act Article 15 (styxx v0.1 measurement map)",
            "evidence_clauses": audit_clauses,
            "uncovered_boundary": _uncovered_boundary(),
            "disclaimer": (
                "Substrate-checkable claims only. Not legal advice; not a "
                "substitute for an organization's own AI Act conformity "
                "assessment. Interpretive, causal, and generalization claims "
                "are out of scope (see uncovered_boundary)."
            ),
        },
    }
    artifact["digest"] = {"alg": "sha256", "value": _compute_digest(artifact)}
    return Attestation(artifact=artifact)


def verify_attestation(
    artifact: dict[str, Any] | Attestation,
    repo_path: str | Path,
) -> VerificationResult:
    """Independently re-verify an attestation against the substrate.

    Re-runs each claim's checker against ``repo_path`` and compares the
    RE-DERIVED verdict to the verdict embedded in the artifact. The embedded
    verdict is NEVER taken as truth — this is what makes the artifact
    agent-independent. Also recomputes the SHA-256 digest over the canonical
    payload and confirms it matches the embedded digest (tamper-evidence).

    A mismatch means the embedded verdict disagrees with what the substrate
    actually says (the agent — or a tamperer — misreported). An unknown
    checker name is refused, never executed.

    If the attestation is commit-pinned, verification re-materializes the EXACT
    recorded commit SHA (not the ref name, which could have moved) and
    reproduces against that historical tree — so as-of-date provenance is
    reproduced as-of that date, not against the repo's current state.
    """
    art = artifact.artifact if isinstance(artifact, Attestation) else artifact
    repo = Path(repo_path).resolve()

    digest_ok = (
        "digest" in art
        and art["digest"].get("value") == _compute_digest(art)
    )

    substrate = art.get("substrate", {})
    pinned_ref = substrate.get("pinned_ref")
    pinned_commit = substrate.get("commit")

    reproduced: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unknown: list[str] = []

    @contextmanager
    def _audit_path() -> Iterator[Path]:
        if pinned_ref is None:
            yield repo
        else:
            with _materialized_tree(repo, str(pinned_commit)) as tree:
                yield tree

    try:
        cm = _audit_path()
        audit_path = cm.__enter__()
    except Exception as e:  # pinned commit missing / unresolvable in this repo
        for entry in art.get("claims", []):
            rec = {
                "id": entry.get("id"),
                "embedded_verdict": entry.get("verdict"),
                "reproduced_verdict": "ERROR",
                "evidence": "",
                "error": f"cannot materialize pinned commit: {type(e).__name__}: {e}",
            }
            reproduced.append(rec)
            mismatches.append(rec)
        return VerificationResult(
            digest_ok=digest_ok, reproduced=reproduced,
            mismatches=mismatches, unknown_checkers=unknown,
        )

    try:
        auditor = AgentClaimAuditor(audit_path)
        for entry in art.get("claims", []):
            name = entry.get("checker")
            checker = _CHECKER_ALLOWLIST.get(name)
            if checker is None:
                unknown.append(name)
                continue
            claim = Claim(
                id=entry["id"],
                text=entry.get("text", ""),
                checker=checker,
                args=entry.get("args", {}),
                expected=entry.get("expected", True),
            )
            (res,) = auditor.run([claim])
            rec = {
                "id": entry["id"],
                "embedded_verdict": entry.get("verdict"),
                "reproduced_verdict": res.verdict,
                "evidence": res.evidence,
                "error": res.error,
            }
            reproduced.append(rec)
            if res.verdict != entry.get("verdict"):
                mismatches.append(rec)
    finally:
        cm.__exit__(None, None, None)

    return VerificationResult(
        digest_ok=digest_ok,
        reproduced=reproduced,
        mismatches=mismatches,
        unknown_checkers=unknown,
    )
