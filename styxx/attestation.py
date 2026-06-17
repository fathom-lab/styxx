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
    "attest_chain",
    "verify_chain",
    "AttestationChain",
    "ChainVerificationResult",
    "CHAIN_VERSION",
]

ATTESTATION_VERSION = "1.0"
CHAIN_VERSION = "1.0"
_CHAIN_GENESIS = "styxx-attestation-chain-v1"

# Security boundary: verify_attestation resolves a stored checker name ONLY
# against this allowlist of public, read-only methods on the trusted _Checkers
# singleton. An untrusted artifact can never name an arbitrary attribute to
# call — an unknown name is refused (verdict ERROR), never executed.
_CHECKER_ALLOWLIST: dict[str, Any] = {
    name: getattr(checkers, name)
    for name in dir(checkers)
    if not name.startswith("_") and callable(getattr(checkers, name))
}

# Checkers that must IMPORT substrate code to evaluate a claim — importing runs
# the module's top-level code. Re-verifying an untrusted third-party receipt must
# never execute attacker-controlled substrate, so verify_attestation REFUSES
# these by default and only runs them when the caller passes trust_substrate=True
# (i.e. "I trust the code in this repo"). A refused checker is recorded as unsafe
# and forces VerificationResult.ok to False — fail-closed, never silently skipped.
_CODE_EXECUTION_CHECKERS: frozenset[str] = frozenset(
    {"python_attr_in_iterable", "python_attr_equals"}
)


@dataclass
class Attestation:
    """The verifiable artifact. ``artifact`` is the JSON-serializable dict."""

    artifact: dict[str, Any]
    # In-memory only (NEVER serialized into the artifact): when the attestation
    # was produced with redactable=True, this carries the secret salt map needed
    # to disclose individual fields later. The public artifact holds only the
    # redactable ROOT under digest.redactable.
    redaction: dict[str, Any] | None = None

    @property
    def digest(self) -> str:
        return self.artifact["digest"]["value"]

    def disclose(self, pointers: list[str]) -> dict[str, Any]:
        """Selectively disclose attested fields (by JSON-pointer) and prove they
        are bound to digest.redactable.root, revealing nothing else.

        Requires the attestation to have been produced with ``redactable=True``
        (so the secret salts are in memory). See :mod:`styxx.redact`.
        """
        if self.redaction is None:
            raise ValueError(
                "this attestation is not redactable — produce it with "
                "attest(..., redactable=True) to enable selective disclosure."
            )
        from . import redact
        core = {k: v for k, v in self.artifact.items() if k not in ("generated_at", "digest")}
        return redact.disclose(core, self.redaction, pointers)

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
    # Code-execution checkers refused because the substrate was not trusted
    # (verify_attestation(trust_substrate=False)). Non-empty => not ok.
    unsafe_checkers: list[str] = field(default_factory=list)
    # Portable (cross-language) digest. None when the artifact predates it
    # (no digest.portable field); True/False once present.
    portable_present: bool = False
    portable_ok: bool = True
    # Vitals reproduction (only populated when the artifact embeds vitals).
    vitals_present: bool = False
    vitals_mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def vitals_ok(self) -> bool:
        """True iff no embedded vitals score diverged from its re-derivation."""
        return not self.vitals_mismatches

    @property
    def ok(self) -> bool:
        """True iff digest matched AND every embedded verdict AND every embedded
        cognometric score reproduced from the substrate."""
        return (
            self.digest_ok
            and self.portable_ok
            and not self.mismatches
            and not self.unknown_checkers
            and not self.unsafe_checkers
            and not self.vitals_mismatches
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "digest_ok": self.digest_ok,
            "portable_present": self.portable_present,
            "portable_ok": self.portable_ok,
            "n_reproduced": len(self.reproduced),
            "n_mismatches": len(self.mismatches),
            "mismatches": self.mismatches,
            "unknown_checkers": self.unknown_checkers,
            "unsafe_checkers": self.unsafe_checkers,
            "vitals_present": self.vitals_present,
            "vitals_ok": self.vitals_ok,
            "vitals_mismatches": self.vitals_mismatches,
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


# ---------------------------------------------------------------------------
# Portable (cross-language) content address. The legacy digest above uses
# Python's json number repr, which is not language-portable (e.g. Python emits a
# saturating score as `1.0`, JavaScript as `1`). The portable digest below is an
# ADDITIVE second address over an RFC 8785 / ECMAScript-canonical payload so the
# content address reproduces byte-for-byte in any language (see
# docs/attestation-content-address.md and web/styxx_verify.js). The legacy
# `digest.value` is left untouched — already-issued receipts stay valid.
# ---------------------------------------------------------------------------

def _es_number_to_string(num: float) -> str:
    """Serialize a finite double per ECMAScript Number::toString (RFC 8785).

    This is what JavaScript's String(n) / JSON.stringify(n) produces, so the
    portable canonical form is identical across languages.
    """
    if num != num or num in (float("inf"), float("-inf")):
        raise ValueError("portable digest is defined for finite numbers only")
    if num == 0:
        return "0"
    if num < 0:
        return "-" + _es_number_to_string(-num)
    # Python repr() yields the shortest round-tripping decimal; reformat its
    # digits to the ECMAScript rules.
    s = repr(num)
    if "e" in s or "E" in s:
        mant, _, exp_s = s.lower().partition("e")
        exp = int(exp_s)
    else:
        mant, exp = s, 0
    if "." in mant:
        intp, frac = mant.split(".")
    else:
        intp, frac = mant, ""
    digits = intp + frac
    e = exp - len(frac)            # value == int(digits) * 10**e
    digits = digits.lstrip("0") or "0"      # leading zeros don't change the int value
    stripped = digits.rstrip("0") or "0"
    e += len(digits) - len(stripped)        # trailing zeros do
    digits = stripped
    k = len(digits)
    n = k + e                      # ECMAScript 'n': value == digits * 10**(n-k)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e_exp = n - 1
    mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{mantissa}e{'+' if e_exp >= 0 else '-'}{abs(e_exp)}"


def _jcs(obj: Any) -> str:
    """RFC 8785 (JCS) canonical serialization, scoped to the styxx artifact
    domain (ASCII keys, finite doubles, no NaN/Inf, no control chars in values).
    """
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
        return "[" + ",".join(_jcs(x) for x in obj) + "]"
    if isinstance(obj, dict):
        parts = (
            json.dumps(k, ensure_ascii=False) + ":" + _jcs(v)
            for k, v in sorted(obj.items(), key=lambda kv: kv[0])
        )
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"not JCS-serializable: {type(obj).__name__}")


def _portable_canonical_payload(artifact: dict[str, Any]) -> str:
    core = {k: v for k, v in artifact.items() if k not in ("generated_at", "digest")}
    return _jcs(core)


def _compute_portable_digest(artifact: dict[str, Any]) -> str:
    payload = _portable_canonical_payload(artifact).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


# ---------------------------------------------------------------------------
# Cognometric vitals. The text-heuristic instruments are a pure function of
# (prompt, response) — no network, no randomness — so the scores are content-
# addressable and reproducible. They are RELATIONAL: score_all(response=...)
# alone returns {} (sycophancy/overconfidence/deception/refusal are undefined
# without a prompt). We therefore record the prompt the report responds to as
# part of the attested substrate, and verify re-derives the scores from it.
# ---------------------------------------------------------------------------

# Stored scores are rounded to this many decimals so the content address is
# stable across platforms; tamper differences dwarf this precision.
_VITALS_PRECISION = 12

# Per-axis honest scope. These are register (textual surface) measurements, NOT
# ground-truth honesty — styxx's validated construct ceiling. Reference-less
# deception in particular saturates on benign text and is register-only.
_VITALS_CAVEATS = {
    "sycophancy": "register: agreement/flattery surface features vs the prompt; not a measure of whether the response is true.",
    "overconfidence": "register: assertive/hedge-free surface features; not a measure of calibration against ground truth (validated construct ceiling).",
    "deception": "register only, REFERENCE-LESS: saturates on benign text; not a deception finding without a grounding reference.",
    "refusal": "register: textual refusal-pattern features.",
}


def _round_scores(scores: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), _VITALS_PRECISION) for k, v in sorted(scores.items())}


def _compute_vitals(prompt: str, response: str) -> dict[str, Any]:
    """Deterministic text-heuristic cognometric vitals for (prompt, response).

    Pure function of the two texts (no network, no randomness). The prompt and
    response are recorded so a third party re-derives the scores rather than
    trusting them.
    """
    from .attack import score_all

    scores = _round_scores(score_all(prompt=prompt, response=response))
    return {
        "tier": "text-heuristic",
        "instrument": "styxx.attack.score_all",
        "styxx_version": _STYXX_VERSION,
        "prompt": prompt,
        "response": response,
        "scores": scores,
        "measures": (
            "register (textual surface features), NOT ground-truth honesty or "
            "correctness. These are re-derivable from the recorded (prompt, "
            "response); they do not prove the report is true."
        ),
        "caveats": {k: _VITALS_CAVEATS[k] for k in scores if k in _VITALS_CAVEATS},
        "axes_undefined_without_prompt": True,
    }


def attest(
    report_text: str,
    repo_path: str | Path,
    *,
    id_prefix: str = "A",
    ref: str | None = None,
    prompt: str | None = None,
    vitals: bool = False,
    redactable: bool = False,
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
        prompt: the task/instruction the report responds to. Required when
            ``vitals=True`` (the cognometric instruments are relational —
            undefined without a prompt).
        vitals: when True, embed the deterministic text-heuristic cognometric
            vitals (``styxx.attack.score_all``) of the (prompt, report) pair,
            re-derivable by :func:`verify_attestation`. Measures REGISTER, not
            ground-truth honesty (see the embedded ``vitals.measures``).

    Returns:
        An :class:`Attestation` whose ``.artifact`` is a JSON-serializable,
        content-addressed dict. The verdicts are produced by re-running each
        extracted claim's checker against the substrate — the same mechanical
        audit any third party reproduces via :func:`verify_attestation`.
    """
    if vitals and prompt is None:
        raise ValueError(
            "vitals=True requires a prompt: the cognometric instruments are "
            "relational and undefined for a referent-free monologue."
        )
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
    if vitals:
        artifact["vitals"] = _compute_vitals(prompt, report_text)

    artifact["digest"] = {
        "alg": "sha256",
        "value": _compute_digest(artifact),
        # Additive cross-language address (RFC 8785 numbers); legacy `value`
        # above is unchanged. Verifiable in any language — see web/styxx_verify.js.
        "portable": {"alg": "sha256-jcs", "value": _compute_portable_digest(artifact)},
    }

    redaction = None
    if redactable:
        # Additive salted-Merkle commitment over the attested fields, enabling
        # later SELECTIVE DISCLOSURE. The public artifact carries only the root;
        # the secret salts ride on the returned object. By design this makes the
        # receipt non-deterministic (the salts ARE the confidentiality); the
        # legacy + portable digests above are computed before this and unchanged.
        from . import redact
        core = {k: v for k, v in artifact.items() if k not in ("generated_at", "digest")}
        redaction = redact.redactable_commit(core)
        artifact["digest"]["redactable"] = {
            "alg": redaction["alg"],
            "version": redaction["version"],
            "root": redaction["root"],
            "tree_size": redaction["tree_size"],
        }
    return Attestation(artifact=artifact, redaction=redaction)


def verify_attestation(
    artifact: dict[str, Any] | Attestation,
    repo_path: str | Path,
    *,
    trust_substrate: bool = False,
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

    Untrusted-substrate safety: checkers that must import substrate code to
    evaluate a claim (``python_attr_*``) are REFUSED by default, because
    importing executes the repo's top-level code and a third-party receipt's
    repo is attacker-controlled. They run only when ``trust_substrate=True``.
    Refused checkers are recorded in ``unsafe_checkers`` and force ``ok`` False.

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

    portable_node = (art.get("digest") or {}).get("portable")
    portable_present = isinstance(portable_node, dict) and "value" in portable_node
    portable_ok = (
        portable_node.get("value") == _compute_portable_digest(art)
        if portable_present
        else True
    )

    substrate = art.get("substrate", {})
    pinned_ref = substrate.get("pinned_ref")
    pinned_commit = substrate.get("commit")

    reproduced: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unknown: list[str] = []
    unsafe: list[str] = []

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
            portable_present=portable_present, portable_ok=portable_ok,
        )

    try:
        auditor = AgentClaimAuditor(audit_path)
        for entry in art.get("claims", []):
            name = entry.get("checker")
            checker = _CHECKER_ALLOWLIST.get(name)
            if checker is None:
                unknown.append(name)
                continue
            if name in _CODE_EXECUTION_CHECKERS and not trust_substrate:
                # Refuse to import attacker-controlled substrate code. Recorded
                # (not silently skipped) so ok is False and the caller sees why.
                unsafe.append(name)
                rec = {
                    "id": entry.get("id"),
                    "embedded_verdict": entry.get("verdict"),
                    "reproduced_verdict": "REFUSED",
                    "evidence": "",
                    "error": (
                        f"checker {name!r} requires executing substrate code; "
                        "refused (pass trust_substrate=True to allow)"
                    ),
                }
                reproduced.append(rec)
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

    # Cognometric vitals: re-derive the scores from the recorded (prompt,
    # response) and compare. A re-sealed-digest score tamper is caught here
    # because the score is recomputed from the substrate text, never trusted.
    vitals_present = "vitals" in art
    vitals_mismatches: list[dict[str, Any]] = []
    if vitals_present:
        v = art["vitals"]
        embedded = v.get("scores", {})
        try:
            rederived = _compute_vitals(v.get("prompt", ""), v.get("response", ""))["scores"]
        except Exception as e:  # pragma: no cover - scorer import/parse failure
            vitals_mismatches.append({"axis": "*", "error": f"{type(e).__name__}: {e}"})
            rederived = {}
        else:
            axes = set(embedded) | set(rederived)
            for axis in sorted(axes):
                emb = embedded.get(axis)
                red = rederived.get(axis)
                diverged = (
                    emb is None or red is None
                    or abs(float(emb) - float(red)) > 1e-9
                )
                if diverged:
                    vitals_mismatches.append(
                        {"axis": axis, "embedded": emb, "rederived": red}
                    )

    return VerificationResult(
        digest_ok=digest_ok,
        reproduced=reproduced,
        mismatches=mismatches,
        unknown_checkers=unknown,
        unsafe_checkers=unsafe,
        portable_present=portable_present,
        portable_ok=portable_ok,
        vitals_present=vitals_present,
        vitals_mismatches=vitals_mismatches,
    )


# ===========================================================================
# Attestation chain — a Merkle-linked, tamper-evident ledger of attestations.
# Order is bound into a rolling digest, so a third party can verify not just
# "this claim was true at commit X" but the agent's whole ordered claim
# history: complete, in sequence, and untampered. Trust the substrate, not the
# agent — across time.
# ===========================================================================


@dataclass
class AttestationChain:
    """An ordered, Merkle-linked sequence of attestations. ``artifact`` is the
    JSON-serializable dict; ``head`` is the rolling digest over the whole chain."""

    artifact: dict[str, Any]

    @property
    def head(self) -> str:
        return self.artifact["head_chain_digest"]

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.artifact, indent=indent, ensure_ascii=False, sort_keys=True)


@dataclass
class ChainVerificationResult:
    """Outcome of independently re-verifying an attestation chain."""

    links_ok: bool
    head_ok: bool
    per_link: list[dict[str, Any]] = field(default_factory=list)
    broken_at: int | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """True iff the Merkle links are intact, the head matches, AND every
        link's attestation independently reproduces against its pinned commit."""
        return (
            self.links_ok
            and self.head_ok
            and self.broken_at is None
            and all(p["attestation_ok"] for p in self.per_link)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "links_ok": self.links_ok,
            "head_ok": self.head_ok,
            "broken_at": self.broken_at,
            "reason": self.reason,
            "n_links": len(self.per_link),
        }


def _chain_digest(prev: str, attestation_digest: str) -> str:
    return hashlib.sha256(f"{prev}|{attestation_digest}".encode("utf-8")).hexdigest()


def _link_chain(att_digests: list[str]) -> list[str]:
    """Roll the per-attestation digests into a Merkle-style chain."""
    chain: list[str] = []
    prev = _CHAIN_GENESIS
    for d in att_digests:
        prev = _chain_digest(prev, d)
        chain.append(prev)
    return chain


def attest_chain(
    items: list[tuple[str, str | None]] | list[tuple[str, str | None, str | None]],
    repo_path: str | Path,
    *,
    id_prefix: str = "A",
    vitals: bool = False,
) -> AttestationChain:
    """Produce a Merkle-linked chain of attestations over an ordered sequence.

    Args:
        items: ordered links. Each is ``(report_text, ref)`` or, to carry
            cognometric vitals, ``(report_text, ref, prompt)``. ``ref`` pins the
            link's substrate to a commit (or ``None`` for the live working tree).
        repo_path: the substrate git repository.
        id_prefix: claim-id prefix; each link gets ``f"{id_prefix}{seq}"``.
        vitals: when True, embed the re-derivable cognometric vitals on each
            link. Every item must then supply a 3rd element ``prompt`` (the
            instruments are relational); a missing prompt raises ``ValueError``.

    Returns:
        An :class:`AttestationChain`. Each link stores its full attestation
        artifact, that artifact's digest, the previous chain digest, and the
        rolling chain digest — so order is content-addressed and any
        insertion / deletion / reorder is detectable by :func:`verify_chain`.
    """
    repo = Path(repo_path).resolve()

    def _unpack(item: tuple) -> tuple[str, str | None, str | None]:
        if len(item) == 3:
            return item[0], item[1], item[2]
        report, ref = item
        return report, ref, None

    attestations = []
    for i, item in enumerate(items):
        report, ref, prompt = _unpack(item)
        attestations.append(
            attest(report, repo, id_prefix=f"{id_prefix}{i}", ref=ref,
                   prompt=prompt, vitals=vitals)
        )
    att_digests = [a.digest for a in attestations]
    chain = _link_chain(att_digests)
    # Additive portable chain: roll the cross-language attestation addresses with
    # the SAME hex-only Merkle rule (the rule was already language-agnostic).
    att_portable = [a.artifact["digest"]["portable"]["value"] for a in attestations]
    chain_portable = _link_chain(att_portable)

    links = [
        {
            "seq": i,
            "attestation": a.artifact,
            "attestation_digest": att_digests[i],
            "attestation_portable_digest": att_portable[i],
            "prev_chain_digest": (_CHAIN_GENESIS if i == 0 else chain[i - 1]),
            "chain_digest": chain[i],
        }
        for i, a in enumerate(attestations)
    ]

    artifact: dict[str, Any] = {
        "styxx_chain_version": CHAIN_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": {"styxx_version": _STYXX_VERSION},
        "n_links": len(links),
        "links": links,
        "head_chain_digest": (chain[-1] if chain else _CHAIN_GENESIS),
        "head_chain_portable_digest": (chain_portable[-1] if chain_portable else _CHAIN_GENESIS),
    }
    return AttestationChain(artifact=artifact)


def verify_chain(
    chain: dict[str, Any] | AttestationChain,
    repo_path: str | Path,
    *,
    expected_head: str | None = None,
) -> ChainVerificationResult:
    """Independently re-verify an attestation chain against the substrate.

    Recomputes the rolling Merkle digest from each link's per-attestation
    digest and confirms (a) each link's stored ``chain_digest`` matches the
    recomputation, (b) the head matches, and (c) every link's attestation
    independently reproduces via :func:`verify_attestation` against its pinned
    commit. The embedded values are never taken as truth.

    Honest tamper model: the chain binds ORDER into a single head digest. A
    naive reorder / insert / delete (without re-sealing the chain digests) is
    caught outright as a broken link. A sophisticated attacker who re-seals
    every chain digest produces an internally-consistent chain — that is only
    detectable against a head that was anchored externally BEFORE the tamper
    (committed to git, timestamped, published). Pass that anchored value as
    ``expected_head`` to get that guarantee; otherwise ``head_ok`` only checks
    the chain's own stored head (internal consistency).
    """
    art = chain.artifact if isinstance(chain, AttestationChain) else chain
    repo = Path(repo_path).resolve()
    links = art.get("links", [])

    per_link: list[dict[str, Any]] = []
    broken_at: int | None = None
    reason = ""
    prev = _CHAIN_GENESIS

    for i, link in enumerate(links):
        att = link.get("attestation", {})
        # the per-attestation digest must match the attestation's own content
        recomputed_att_digest = _compute_digest(att)
        stored_att_digest = link.get("attestation_digest")
        recomputed_chain = _chain_digest(prev, recomputed_att_digest)
        stored_chain = link.get("chain_digest")

        att_res = verify_attestation(att, repo)

        link_intact = (
            recomputed_att_digest == stored_att_digest
            and link.get("prev_chain_digest") == prev
            and recomputed_chain == stored_chain
        )
        per_link.append(
            {
                "seq": link.get("seq", i),
                "attestation_ok": att_res.ok,
                "attestation_digest_ok": recomputed_att_digest == stored_att_digest,
                "chain_link_ok": link_intact,
            }
        )
        if broken_at is None and not link_intact:
            broken_at = i
            reason = f"chain broken at link {i}: recomputed digest != stored"
        prev = recomputed_chain  # walk the RECOMPUTED chain, not the stored one

    links_ok = broken_at is None
    anchor = expected_head if expected_head is not None else art.get("head_chain_digest")
    head_ok = (prev == anchor) if links else False
    if links_ok and not head_ok:
        reason = (
            "recomputed chain head does not match the anchored expected_head"
            if expected_head is not None
            else "head_chain_digest does not match the recomputed chain head"
        )

    return ChainVerificationResult(
        links_ok=links_ok,
        head_ok=head_ok,
        per_link=per_link,
        broken_at=broken_at,
        reason=reason,
    )
