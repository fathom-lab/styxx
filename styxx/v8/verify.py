"""styxx.v8.verify -- ``verify --diff``, ``verify --ref``, result certs, challenge bodies (spec v0.2 section 6, section 9).

Contract: ``styxx/v8/INTERFACES_layer2.md`` section 8.

Two entry points and the exit-code contract of section 6:

* ``diff(a, b, resolver)`` compares two existing certs and executes nothing.  It uses the LEFT
  cert's floor (``floor_owner: "A"``) and prints B's floor beside it.  A ``--diff`` has no
  confirmation step, so it can never say ``drift``: an exceedance is ``exceeds_floor`` (GATED
  S5-02, recommendation A, implemented in ``floor.decide``).
* ``ref(cert, runner, resolver)`` re-runs the cert's own recipe now, compares the new run to the
  cert's body, and -- when any channel exceeds the floor -- runs the mandatory confirmation run
  before saying ``drift``.

| code | meaning |
|---|---|
| 0 | same |
| 1 | drift, or identity |
| 2 | inconclusive, skew, beyond-floor-coverage, sensitivity unmeasured, exceeds_floor, or a --diff without a usable floor |
| 3 | recipe mismatch, or a schema version this verifier does not implement |
| 4 | invalid cert: id does not recompute, signature fails, a ref does not resolve |
| 5 | unavailable: the subject or the environment could not be obtained |

Every verdict string comes from ``styxx.v8.floor`` (``decide`` / ``overall``); nothing here
invents vocabulary except the two compounds documented under "Decisions" below.

Decisions (where the contract or the spec leaves room; each is pinned by a test in
``tests/test_v8_verify.py``):

* **Schema version is exit 3, not exit 4.**  Section 2.5 says a verifier refuses a higher
  ``major.minor`` with exit 3, but ``cert.check`` folds the version into a plain reason string,
  which would read as exit 4.  ``verify`` tests ``version_ok`` FIRST and returns
  ``mismatch (schema version)`` with exit 3 and ``mismatched = ["schema_version:<value>"]``.  A
  ``styxx`` field that does not parse as ``major.minor`` at all is not a version the spec covers;
  it falls through to ``check`` and exits 4.
* **``identity`` is a ``--ref`` verdict, never a ``--diff`` verdict.**  ``weights_sha256`` and its
  three siblings are inside ``S_identity`` (section 2.2), so two certs that differ there are not
  comparable and a ``--diff`` exits 3 (section 2.3).  Section 5.2's ``identity`` is about "the
  reference and the new run": it fires when the subject a verifier actually OBTAINED differs from
  the subject the cert names (Appendix D step 3).
* **How ``ref`` learns the obtained subject, and what it does when it cannot.**  Every runner
  implements ``subject(requested)`` (``styxx.v8.runner``), reporting the ``kind`` and every
  ``S_identity`` field it will ACTUALLY run.  ``ref`` compares that against the cert:

  - the four A.2 hashes (``floor.IDENTITY_FIELDS``) differ -> ``identity (<fields>)``, exit 1,
    the section 5.2 subject verdict;
  - ``precision``, ``revision``, ``hf_repo`` or ``kind`` differ -> ``mismatch``, exit 3.  These
    are ``S_identity`` too (section 2.2) but section 5.2's ``identity`` names only the four
    hashes, so a ``--ref`` that did not re-run the cert's subject refuses to compare rather
    than inventing a verdict word.  ``cross-subject`` is a ``--diff`` affordance (section 6):
    a ``--ref`` must re-run the SAME subject;
  - the runner has no ``subject`` member, returns something that is not a subject mapping, or
    leaves an ``S_identity`` field unreported -> ``unavailable``, exit 5.  It NEVER falls back
    to the cert's own subject.

  The fallback is what this repair removed.  ``ref`` used to read an OPTIONAL ``subject``
  member and, absent it, compare the cert against itself: with the real runner the guard could
  not fail, so ``--ref --dtype float16`` against a bf16 cert ran other weights, reported
  ``identity_diff: []``, and signed a ``drift`` accusation against another party's cert
  (``papers/v8/challenge_and_attack_2026_09_09``, C2).  Advisory fields (``model_family``,
  ``environment``, the alias ``observed_*``) still come from the cert -- they are not identity
  and the runner is not asked for them; identity fields come from the runner alone.
* **A synthetic runner cannot verify a measured cert, and the result says which it was.**  A
  runner that computes from a hash rather than from a model declares ``synthetic``
  (``runner.is_synthetic``), and ``ref`` puts that into the object it compares against the cert,
  so ``comparable`` returns ``body.synthetic`` and the outcome is ``mismatch``, exit 3, with no
  distances and nothing a challenge can be built from.  The reverse holds too: a real runner
  against a synthetic cert is the same mismatch, because ``_observed_subject`` never inherits
  the cert's own marker.  ``--runner mock`` is the CLI's default, and it signed a section 9
  challenge against a published canonical carrying distances from a model that was never loaded
  (``papers/v8/challenge_and_attack_2026_09_09``, C-MOCK); the argument for a marker rather than
  a refusal to sign is in ``styxx/v8/cert.py`` under "Decisions".
* **The result body records what the verifier ran, and the challenge body carries it.**
  ``observed_subject`` (the runner's own report) and ``observed_recipe_core`` go into every
  ``ref`` result body; ``challenge_body`` copies the S_identity half and the recipe core into
  the section 9 body, which is C3's repair -- see that function.  A ``diff`` outcome carries
  neither, and ``challenge_body`` refuses it: a diff compares two certs and runs nothing, so it
  has no observation to file.
* **How ``ref`` learns the environment it ran in: it asks the runner, and it does not take an
  answer from a file.**  ``observed_env`` is ``runner.environment()`` through
  ``runner.observed_environment``; the ``environment=`` argument (the CLI's ``--environment
  <file>``) may only ANNOTATE it with leaves the runner did not report, and a leaf that
  contradicts an observed one is ``EnvironmentUnavailable`` -- exit 5, no result cert, no
  challenge.  The exception is a **synthetic** runner, which observes no hardware at all and
  whose certs are already fenced (``cert.comparable`` returns ``body.synthetic``).

  This is S-DEVICE's repair.  ``TransformersRunner.environment()`` reported the box's card
  rather than the runner's own device and ``--environment`` overwrote the block wholesale, so
  ``verify --ref --device cpu --environment <a file naming an RTX 4070> --challenge`` ran one CPU
  forward pass and signed a result and a challenge naming that card, with ``mismatched: []`` and
  ``identity_diff: []``.  Nothing here makes ``device`` identity -- section 2.2 forbids that and
  section 5.4 is where an environment difference belongs -- the field is simply observed now, and
  a floor whose ``not_covered`` names ``hardware.gpu`` turns the CPU run into
  ``beyond-floor-coverage``, which is section 5.4 working rather than a new gate.
* **How ``ref`` learns the verifier's own harness (skew).**  The annotation may carry ``harness``
  and/or ``env_lock_sha256`` beside ``runtime``/``hardware``; they are compared against the cert's
  recipe by ``cert.skew_fields``.  Absent, there is no skew -- the cert's own recipe is what was
  re-run.  They are exactly the fields a runner cannot observe, which is why an annotation exists
  at all.
* **``environment_diff``** lists every ``runtime``/``hardware`` leaf where the observed
  environment differs from the cert's ``subject.environment``.  It is recorded and printed and
  gates nothing (``_environment_diff``); ``coverage_diff`` is what acts, over the floor's own
  ``not_covered``.
* **Coverage** (section 5.4) is computed over ``noise_floor.not_covered`` read as dotted paths into
  ``subject.environment``: a ``not_covered`` field whose observed value differs from the cert's is
  outside the floor's coverage, and every channel is ``beyond-floor-coverage``.  ``covers`` is
  printed, never used as a gate -- it names nuisance factors ("order", "batch_size"), not
  environment fields.
* **A cross-subject ``--diff`` can never say ``same``** (section 6).  When the only comparability
  entries are ``cross-subject:`` and the overall verdict would have been ``same`` or
  ``same (sensitivity unmeasured)``, the verdict is ``inconclusive (cross-subject)`` and the exit
  code is 2.  The compound follows the shape ``floor.overall`` already uses
  (``same (sensitivity unmeasured)``); no new bare verdict word is introduced.
* **A channel whose distance cannot be computed is ``inconclusive``**, not an exception: Appendix B
  makes ``lens`` inconclusive when the two ``n_layers`` differ, and a verifier that a stranger runs
  must report rather than crash.  The reason is recorded in the channel block under ``note``.
* **``ratio``** is ``distance / floor`` rounded to ``ROUND_PLACES`` when ``floor > 0``, else
  ``null`` (JCS forbids NaN and Infinity).
* **The alias sentence of section 2.2 is printed with every alias verdict** and stored in the
  result body under ``alias_note``.
* ``make_result_cert`` refuses to build a cert whose body embeds a cert id that the caller's
  ``refs`` do not carry (section 2.1, A-09) -- ``cert.check`` would refuse it anyway, later and
  with a less useful message.

Pure CPU; the only execution is whatever ``runner`` does.
"""
from __future__ import annotations

import copy
import datetime
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from styxx.v8 import cert as certmod
from styxx.v8 import distances as D
from styxx.v8 import fingerprint as FP
from styxx.v8 import floor as floormod
from styxx.v8 import runner as runnermod
from styxx.v8.consts import CHANNELS, EXIT, ROUND_PLACES, SCHEMA_VERSION

__all__ = [
    "ALIAS_NOTE",
    "VerifyOutcome",
    "challenge_body",
    "diff",
    "make_result_cert",
    "ref",
]

# ``floor`` owns the Appendix B dispatch for two run BODIES (the resid/lens per-item re-keying
# lives there).  Re-implementing it here would be a second implementation of the same table, and
# the spec says `verify` never picks a distance function.  See the deviations note.
_body_distance = floormod._distance
_channel_block = floormod._channel_block

# Section 2.2: printed with every alias verdict, verbatim.
ALIAS_NOTE = (
    "for an alias subject, `same` is evidence only against a provider that does not condition on "
    "the request; a provider that recognises published prompts and serves them from a pinned model "
    "or a cache defeats every black-box channel, and the log cannot tell."
)

_VERSION_RE = re.compile(r"^([0-9]+)\.([0-9]+)$")
_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEFAULT_ISSUER_NAME = "styxx verify"

# Sentinel: no resolver was supplied, so ref resolution is not part of this verification.
_NO_RESOLVER = object()


@dataclass
class VerifyOutcome:
    """The whole outcome of one verification: the body of the result cert, the verdict, the
    exit code, the fields that made it exit 3 (empty otherwise), and the human report."""

    result_body: dict
    verdict: str
    exit_code: int
    mismatched: list[str] = field(default_factory=list)
    printed: str = ""


# --------------------------------------------------------------------------- small helpers


def _mapping(what: str, value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a dict, got {type(value).__name__}")
    return dict(value)


def _dotted(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _cert_id(cert: Mapping) -> Optional[str]:
    """The cert's id: the stored one when it is well formed, else the recomputed one."""
    stored = cert.get("id")
    if isinstance(stored, str) and _ID_RE.match(stored):
        return stored
    try:
        return certmod.compute_id(dict(cert))
    except Exception:
        return None


def _parsed_version(cert: Mapping) -> Optional[tuple[int, int]]:
    raw = cert.get("styxx")
    if not isinstance(raw, str):
        return None
    m = _VERSION_RE.match(raw)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _own_version() -> tuple[int, int]:
    m = _VERSION_RE.match(SCHEMA_VERSION)
    assert m is not None, f"SCHEMA_VERSION {SCHEMA_VERSION!r} is not major.minor"
    return int(m.group(1)), int(m.group(2))


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    return f"{float(x):.{ROUND_PLACES}f}"


def _roles_of(body: Mapping) -> dict[str, str]:
    """``{item_id: role}`` straight off a fingerprint body's items (section 3.1)."""
    out: dict[str, str] = {}
    items = body.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return out
    for item in items:
        if isinstance(item, Mapping) and isinstance(item.get("item_id"), str):
            role = item.get("role", "item")
            out[item["item_id"]] = role if isinstance(role, str) and role else "item"
    return out


def _present_channels(body: Mapping) -> set[str]:
    out = set()
    for channel in CHANNELS:
        try:
            block = _channel_block(body, channel)
        except ValueError:
            continue
        if block is not None:
            out.add(channel)
    return out


def _floor_map(body: Mapping) -> dict[str, Optional[float]]:
    """``{channel: floor}`` from ``body.noise_floor.per_channel``; a channel with no block has
    no floor and maps to ``None``."""
    out: dict[str, Optional[float]] = {channel: None for channel in CHANNELS}
    nf = body.get("noise_floor")
    if not isinstance(nf, Mapping):
        return out
    per = nf.get("per_channel")
    if not isinstance(per, Mapping):
        return out
    for channel, block in per.items():
        if channel not in CHANNELS or not isinstance(block, Mapping):
            continue
        value = block.get("floor")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[channel] = float(value)
    return out


def _not_covered(body: Mapping) -> list[str]:
    nf = body.get("noise_floor")
    if not isinstance(nf, Mapping):
        return []
    names = nf.get("not_covered")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        return []
    return [n for n in names if isinstance(n, str) and n]


def _covers(body: Mapping) -> list[str]:
    nf = body.get("noise_floor")
    if not isinstance(nf, Mapping):
        return []
    names = nf.get("covers")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        return []
    return [n for n in names if isinstance(n, str) and n]


def _coverage_diff(not_covered: Sequence[str], cert_env: Any, observed_env: Any) -> list[str]:
    """The ``not_covered`` environment fields whose observed value differs (section 5.4)."""
    out: list[str] = []
    for name in not_covered:
        if _dotted(cert_env, name) != _dotted(observed_env, name):
            out.append(name)
    return out


def _environment_diff(cert_env: Any, observed_env: Any) -> list[str]:
    """Every ``runtime``/``hardware`` leaf where what was OBSERVED differs from what the cert says.

    Only the two blocks a runner observes (``runner.ENVIRONMENT_BLOCKS``): ``harness`` and
    ``env_lock_sha256`` describe the verifier, are compared as skew (section 2.3), and would
    otherwise show up here as a difference from every cert that never carried them.

    This is a RECORD, not a gate.  Section 2.2 puts the environment outside identity and section
    5.4 spends it on coverage, so what acts is ``coverage_diff`` -- the fields the floor's own
    ``not_covered`` names.  A floor whose ``not_covered`` does not name ``hardware.gpu`` therefore
    still reaches a verdict on a run from another device, and this list is what a reader has to
    see that.  Whether such a difference should force ``beyond-floor-coverage`` on its own is
    OPERATOR-GATED S5-10 in the spec; nothing here decides it.
    """
    cert_env = cert_env if isinstance(cert_env, Mapping) else {}
    observed_env = observed_env if isinstance(observed_env, Mapping) else {}
    paths: set[str] = set()
    for block in runnermod.ENVIRONMENT_BLOCKS:
        for source in (cert_env, observed_env):
            value = source.get(block)
            if isinstance(value, Mapping):
                paths.update(f"{block}.{p}" for p, _ in runnermod.environment_leaves(value))
            elif value is not None:
                paths.add(block)
    return sorted(p for p in paths if _dotted(cert_env, p) != _dotted(observed_env, p))


# --------------------------------------------------------------------------- ref resolution


def _resolve(resolver: Any, cert_id: str) -> Any:
    """One id -> a cert dict, ``None`` (does not resolve), or ``_NO_RESOLVER``.

    ``resolver`` may be ``None`` (no resolution performed), a callable, a mapping of id -> cert,
    or a ``Log`` (anything exposing ``find`` and ``cert``).
    """
    if resolver is None:
        return _NO_RESOLVER
    if isinstance(resolver, Mapping):
        return resolver.get(cert_id)
    find = getattr(resolver, "find", None)
    get = getattr(resolver, "cert", None)
    if callable(find) and callable(get):
        index = find(cert_id)
        if index is None:
            return None
        return get(index)
    if callable(resolver):
        return resolver(cert_id)
    raise TypeError(
        "resolver must be None, a callable, a mapping of id -> cert, or a Log; got "
        f"{type(resolver).__name__}"
    )


def _baseline_gap(cert: Mapping, cert_body: Mapping, resolver: Any) -> Optional[dict]:
    """How far this fingerprint sits from the baseline it replaces, or None (section 5.5).

    **Disclosure, not prevention, and it does not always reach.**  Section 5.5 lets a second
    canonical fingerprint for one subject append when it carries a ``previous`` ref, or when it
    is another run under the same logged noise plan.  Choosing a new baseline is legitimate;
    choosing one silently is the BASELINE-CHOICE attack, and the number that ends the silence is
    the distance between the two baselines.  ``styxx.v8.log.Log.append`` computes it from two
    certs the log holds and writes it into the entry's metadata; this function is how it reaches
    a reader who is running ``verify``.

    Two sources, in this order:

    1. **The log**, through a resolver that offers ``baseline_gap(cert) -> dict | None`` -- a
       ``Log`` does, and so does the resolver ``styxx v8 verify --log`` builds.  This is the only
       source that sees the *unannounced* case, because the previous comparable fingerprint is a
       fact about the log and not about the cert.
    2. **The cert's own ``previous`` ref**, resolved like any other ref.  This reaches only the
       baselines an issuer chose to name, which is the case that was never the attack.

    A ``--diff`` or ``--ref`` run with no log and against a cert carrying no ``previous`` ref
    therefore prints no gap, and the absence is not evidence that no baseline was replaced.  That
    limit is structural: with the cert alone there is nothing to be apart from.

    Whatever the source, the gap is arithmetic over bytes the issuer wrote.  It says the baseline
    moved and by how much; it cannot say which of the two baselines measured the subject
    (``papers/v8/THE_BOUNDARY_2026_09_09.md``, class two -- BASELINE-CHOICE stays there where the
    relabel itself is concerned).
    """
    from_log = getattr(resolver, "baseline_gap", None)
    if callable(from_log):
        try:
            found = from_log(dict(cert))
        except Exception:
            found = None
        if isinstance(found, Mapping):
            out = dict(found)
            out["source"] = "log"
            return out

    previous_ids = [rid for role, rid in certmod.refs(dict(cert)) if role == "previous"]
    for previous_id in previous_ids:
        found = _resolve(resolver, previous_id)
        if found is _NO_RESOLVER or not isinstance(found, Mapping):
            continue
        if found.get("type") != "fingerprint":
            continue
        prev_body = found.get("body") if isinstance(found.get("body"), Mapping) else {}
        try:
            out = dict(floormod.baseline_gap(prev_body, cert_body))
        except (ValueError, TypeError, KeyError) as exc:
            out = {"note": f"{type(exc).__name__}: {exc}"}
        out["previous_id"] = previous_id
        out["declared_previous"] = list(previous_ids)
        out["announced"] = True
        out["source"] = "previous_ref"
        return out
    return None


def _unresolved_refs(cert: Mapping, resolver: Any) -> list[str]:
    """Refs that do not resolve, or resolve to the wrong type (section 2.1 last bullet).

    Reason strings are ``"<role>:<id>: does not resolve"`` / ``": resolves to <type>, not <want>"``.
    With no resolver the list is empty: resolution is the log's job and a caller that did not
    supply one did not ask for it.
    """
    if resolver is None:
        return []
    own_type = cert.get("type")
    reasons: list[str] = []
    for role, cert_id in certmod.refs(dict(cert)):
        found = _resolve(resolver, cert_id)
        if found is _NO_RESOLVER:
            continue
        if not isinstance(found, Mapping):
            reasons.append(f"{role}:{cert_id}: does not resolve")
            continue
        want = certmod.ROLE_TYPES.get(role)
        if want is None:
            want = own_type
        if want == "*":
            continue
        got = found.get("type")
        if got != want:
            reasons.append(f"{role}:{cert_id}: resolves to {got!r}, not {want!r}")
    return reasons


# --------------------------------------------------------------------------- the comparison


def _one_distance(
    a_body: Mapping,
    b_body: Mapping,
    channel: str,
    roles: Mapping[str, str],
) -> tuple[Optional[float], int, Optional[str]]:
    """(distance, anchor_flips, note) for one channel between two run bodies.

    ``distance`` is ``None`` when the channel cannot be compared (Appendix B: ``lens`` with
    differing ``n_layers``, a run missing an item); the note says why and the channel becomes
    ``inconclusive`` rather than raising out of a stranger's verifier.
    """
    try:
        if channel == "exact":
            raw, flips = D.exact(a_body.get("items"), b_body.get("items"), roles)
            return D.rounded(float(raw)), int(flips), None
        return D.rounded(float(_body_distance(a_body, b_body, channel, None))), 0, None
    except (ValueError, TypeError, KeyError) as exc:
        return None, 0, f"{type(exc).__name__}: {exc}"


def _per_channel(
    a_body: Mapping,
    b_body: Mapping,
    floors: Mapping[str, Optional[float]],
    *,
    roles: Mapping[str, str],
    covered: bool,
    skew: bool,
    confirmation_body: Optional[Mapping] = None,
) -> tuple[dict, list[str], int]:
    """``({channel: {distance, floor, ratio, verdict}}, skipped_channels, anchor_flips)``."""
    present_a = _present_channels(a_body)
    present_b = _present_channels(b_body)
    per: dict[str, dict] = {}
    skipped: list[str] = []
    anchor_flips = 0
    for channel in CHANNELS:
        in_a, in_b = channel in present_a, channel in present_b
        if in_a != in_b:
            # Section 5.2: present on one side only -> listed, never an exit code.
            skipped.append(channel)
            continue
        if not in_a:
            continue
        distance, flips, note = _one_distance(a_body, b_body, channel, roles)
        if channel == "exact":
            anchor_flips = flips
        f = floors.get(channel)
        if f is not None:
            f = D.rounded(float(f))
        confirmation: Optional[float] = None
        if confirmation_body is not None and distance is not None:
            confirmation, _flips, cnote = _one_distance(a_body, confirmation_body, channel, roles)
            if confirmation is None:
                note = note or cnote
        verdict = floormod.decide(distance, f, confirmation, covered=covered, skew=skew)
        ratio = None
        if distance is not None and f is not None and f > 0.0:
            ratio = D.rounded(distance / f)
        block: dict[str, Any] = {
            "distance": distance,
            "floor": f,
            "ratio": ratio,
            "verdict": verdict,
        }
        if confirmation_body is not None:
            block["confirmation_distance"] = confirmation
        if note is not None:
            block["note"] = note
        per[channel] = block
    return per, skipped, anchor_flips


# --------------------------------------------------------------------------- reporting


def _report(header: str, body: Mapping) -> str:
    """The one human report a verify prints; every number in it is in ``body``."""
    lines = [header]
    if body.get("mismatched"):
        lines.append("  not comparable: " + ", ".join(body["mismatched"]))
    if body.get("invalid_reasons"):
        for reason in body["invalid_reasons"]:
            lines.append(f"  invalid: {reason}")
    if body.get("unavailable_reason"):
        # Exit 5 printed "overall: unavailable" and nothing else, so the one sentence that says
        # WHICH obligation was not met -- the runner would not name its subject, would not name
        # its environment, or named one an `--environment` file contradicted -- reached stdout
        # only in the result body.  A refusal a reader cannot act on is half a refusal.
        lines.append(f"  unavailable: {body['unavailable_reason']}")
    if body.get("attempted"):
        for key in sorted(body["attempted"]):
            lines.append(f"  attempted {key}: {body['attempted'][key]}")
    if body.get("ref") is not None:
        lines.append(f"  ref: {body['ref']}")
    owner = body.get("floor_owner")
    if owner is not None:
        lines.append(f"  floor_owner: {owner}")
    if body.get("cross_subject"):
        lines.append("  cross-subject: " + ", ".join(body["cross_subject"]))
    if body.get("identity_diff"):
        lines.append("  identity: " + ", ".join(body["identity_diff"]))
    if body.get("skew_fields"):
        lines.append("  skew: " + ", ".join(body["skew_fields"]))
    coverage = body.get("coverage")
    if coverage is not None:
        line = f"  coverage: {coverage}"
        if body.get("covers"):
            line += "  covers=" + ",".join(body["covers"])
        if body.get("not_covered"):
            line += "  not_covered=" + ",".join(body["not_covered"])
        lines.append(line)
        if body.get("coverage_diff"):
            lines.append("    outside coverage on: " + ", ".join(body["coverage_diff"]))
    if body.get("environment_diff"):
        # Printed beside the coverage line, never as a verdict: section 2.2 keeps the environment
        # out of identity and section 5.4 acts through `not_covered` alone.
        lines.append("  observed environment differs from the cert on: "
                     + ", ".join(body["environment_diff"]))
    per = body.get("per_channel") or {}
    for channel in CHANNELS:
        block = per.get(channel)
        if not isinstance(block, Mapping):
            continue
        line = (
            f"  {channel:<6} distance={_fmt(block.get('distance'))}"
            f"  floor={_fmt(block.get('floor'))}"
            f"  ratio={_fmt(block.get('ratio'))}"
            f"  {block.get('verdict')}"
        )
        if block.get("confirmation_distance") is not None:
            line += f"  confirmation={_fmt(block['confirmation_distance'])}"
        lines.append(line)
        if block.get("note"):
            lines.append(f"         note: {block['note']}")
    zero = [
        channel
        for channel in CHANNELS
        if isinstance(per.get(channel), Mapping) and per[channel].get("floor") == 0
    ]
    if zero:
        # Section 5.1: a floor of 0.0 may be honest -- a configuration that repeats itself across
        # genuinely different computations produces one -- or it may be a floor whose runs were
        # one computation wearing R labels, which `styxx.v8.log` now refuses at append and
        # counts in the entry's `floor` metadata either way.  Against 0.0 every difference
        # exceeds and no difference is `same`, so the verdict above rests on the floor's
        # provenance and the reader is told to go and look at it.
        lines.append(
            "  zero floor on: " + ", ".join(zero)
            + "  -- every difference exceeds a floor of 0.0 and none is 'same'; the log entry's"
        )
        lines.append(
            "     floor census says how many distinct computations that floor rests on"
        )
    if body.get("floor_b"):
        pairs = ", ".join(f"{c}={_fmt(v)}" for c, v in sorted(body["floor_b"].items()))
        lines.append(f"  floor(B): {pairs}")
    gap = body.get("baseline_gap")
    if isinstance(gap, Mapping):
        # Section 5.5: a second canonical fingerprint replaced a baseline that was already on
        # record.  That is allowed and this is not a verdict -- it is the announcement, printed
        # beside the verdict rather than in a limits section, because a reader comparing against
        # the new baseline is otherwise never told the old one existed (BASELINE-CHOICE).  It
        # says how far the two are apart; it cannot say which of them measured the subject.
        head = f"  baseline replaced: {gap.get('previous_id')}"
        if gap.get("previous_index") is not None:
            head += f" (log index {gap['previous_index']})"
        lines.append(head)
        how = "named by this cert's previous ref" if gap.get("announced") else (
            "NOT named by this cert: another run under the same noise plan"
            if gap.get("same_noise_plan") else "NOT named by this cert"
        )
        lines.append(f"    announced: {how}  [source: {gap.get('source')}]")
        gap_per = gap.get("per_channel") or {}
        for channel in CHANNELS:
            block = gap_per.get(channel)
            if not isinstance(block, Mapping):
                continue
            line = (
                f"    gap {channel:<6} distance={_fmt(block.get('distance'))}"
                f"  floor(previous)={_fmt(block.get('floor'))}"
                f"  ratio={_fmt(block.get('ratio'))}"
            )
            if block.get("exceeds_floor"):
                line += "  exceeds the previous floor"
            lines.append(line)
            if block.get("note"):
                lines.append(f"           note: {block['note']}")
        if gap.get("skipped_channels"):
            lines.append("    gap skipped channels: " + ", ".join(gap["skipped_channels"]))
        if gap.get("note"):
            lines.append(f"    gap note: {gap['note']}")
        lines.append(
            "    the gap says the baseline moved and by how much; it does not say which "
            "baseline measured the subject"
        )
    if body.get("skipped_channels"):
        lines.append("  skipped channels: " + ", ".join(body["skipped_channels"]))
    if body.get("anchor_flips") is not None:
        lines.append(f"  anchor_flips: {body['anchor_flips']}")
    sensitivity = body.get("sensitivity")
    lines.append(f"  sensitivity: {sensitivity if sensitivity else 'none on record'}")
    lines.append(f"  overall: {body.get('overall')}  (exit {body.get('exit_code')})")
    if body.get("alias_note"):
        lines.append(f"  {body['alias_note']}")
    return "\n".join(lines)


def _finish(header: str, body: dict, verdict: str, code: int, mismatched: list[str]) -> VerifyOutcome:
    body["overall"] = verdict
    body["exit_code"] = code
    if mismatched:
        body["mismatched"] = list(mismatched)
    printed = _report(header, body)
    # the exit code is the process's, not the cert's: it is not stored in the result body
    body.pop("exit_code", None)
    return VerifyOutcome(
        result_body=body,
        verdict=verdict,
        exit_code=code,
        mismatched=list(mismatched),
        printed=printed,
    )


def _shell(mode: str, ref_id: Optional[str]) -> dict:
    """The section 6.1 ``kind: verify`` body, with every required key already present."""
    return {
        "kind": "verify",
        "mode": mode,
        "ref": ref_id,
        "new_run": None,
        "confirmation_run": None,
        "per_channel": {},
        "overall": "",
        "floor_owner": None,
        "coverage": None,
        "skipped_channels": [],
        "rounding": ROUND_PLACES,
    }


def _gate_certs(
    mode: str,
    certs: Sequence[tuple[str, Mapping]],
    resolver: Any,
    ref_id: Optional[str],
    header: str,
) -> Optional[VerifyOutcome]:
    """The shared section 6 gates: schema version (3), then validity and refs (4)."""
    own = _own_version()
    for _label, cert in certs:
        version = _parsed_version(cert)
        if version is not None and version > own:
            body = _shell(mode, ref_id)
            body["floor_owner"] = None
            mismatched = [f"schema_version:{cert.get('styxx')}"]
            body["mismatched"] = mismatched
            return _finish(
                header, body, "mismatch (schema version)", EXIT["mismatch"], mismatched
            )
    reasons: list[str] = []
    for label, cert in certs:
        outcome = certmod.check(dict(cert))
        reasons.extend(f"{label}: {r}" for r in outcome.reasons)
    for label, cert in certs:
        reasons.extend(f"{label}: refs: {r}" for r in _unresolved_refs(cert, resolver))
    if reasons:
        body = _shell(mode, ref_id)
        body["invalid_reasons"] = reasons
        return _finish(header, body, "invalid", EXIT["invalid"], [])
    return None


# --------------------------------------------------------------------------- verify --diff


def diff(cert_a: dict, cert_b: dict, resolver: Any) -> VerifyOutcome:
    """Compare two existing certs.  Executes nothing; uses A's floor (section 6).

    ``resolver(id) -> cert | None`` (a ``Log``, a mapping, or a callable; ``None`` skips ref
    resolution).  A ``--diff`` has no confirmation run and therefore never says ``drift``.
    """
    a = _mapping("cert_a", cert_a)
    b = _mapping("cert_b", cert_b)
    a_id, b_id = _cert_id(a), _cert_id(b)
    header = "styxx verify --diff"

    gated = _gate_certs("diff", (("A", a), ("B", b)), resolver, a_id, header)
    if gated is not None:
        gated.result_body["new_run"] = b_id
        return gated

    body = _shell("diff", a_id)
    body["new_run"] = b_id
    body["floor_owner"] = "A"
    body["floor_owner_id"] = a_id

    if a.get("type") != "fingerprint" or b.get("type") != "fingerprint":
        mismatched = [f"cert.type:{a.get('type')!r} vs {b.get('type')!r}"]
        body["mismatched"] = mismatched
        return _finish(header, body, "mismatch", EXIT["mismatch"], mismatched)

    entries = certmod.comparable(a, b)
    cross = [e.split(":", 1)[1] for e in entries if e.startswith("cross-subject:")]
    hard = [e for e in entries if not e.startswith("cross-subject:")]
    body["cross_subject"] = cross
    if hard:
        body["mismatched"] = hard
        # Section 6 exit 3: the mismatched fields are printed and nothing else -- no distances.
        return _finish(header, body, "mismatch", EXIT["mismatch"], hard)

    a_body = a.get("body") if isinstance(a.get("body"), Mapping) else {}
    b_body = b.get("body") if isinstance(b.get("body"), Mapping) else {}
    a_subject = a.get("subject") if isinstance(a.get("subject"), Mapping) else {}
    b_subject = b.get("subject") if isinstance(b.get("subject"), Mapping) else {}

    not_covered = _not_covered(a_body)
    coverage_diff = _coverage_diff(
        not_covered, a_subject.get("environment"), b_subject.get("environment")
    )
    covered = not coverage_diff
    skew = certmod.skew_fields(a, b)

    floors = _floor_map(a_body)
    per, skipped, anchor_flips = _per_channel(
        a_body,
        b_body,
        floors,
        roles=_roles_of(a_body),
        covered=covered,
        skew=bool(skew),
        confirmation_body=None,
    )
    sensitivity = a_body.get("sensitivity")
    sensitivity = sensitivity if isinstance(sensitivity, str) else None

    body["per_channel"] = per
    body["skipped_channels"] = skipped
    body["anchor_flips"] = anchor_flips
    body["coverage"] = "within" if covered else "beyond-floor-coverage"
    body["coverage_diff"] = coverage_diff
    body["covers"] = _covers(a_body)
    body["not_covered"] = not_covered
    body["skew_fields"] = skew
    body["identity_diff"] = []
    body["sensitivity"] = sensitivity
    body["floor_b"] = {c: v for c, v in _floor_map(b_body).items() if v is not None}
    # A --diff uses A's floor, so A is the cert whose baseline choice governs this comparison;
    # the key is set only when there IS a replaced baseline, so a result body that carries no
    # gap is byte-identical to one from before this disclosure existed.
    gap = _baseline_gap(a, a_body, resolver)
    if gap is not None:
        body["baseline_gap"] = gap
    if a_subject.get("kind") == "alias" or b_subject.get("kind") == "alias":
        body["alias_note"] = ALIAS_NOTE

    verdict, code = floormod.overall(
        {c: block["verdict"] for c, block in per.items()},
        identity_diff=[],
        skipped=skipped,
        sensitivity_present=sensitivity is not None,
    )
    # A --diff has no confirmation run, so `drift` is unreachable by construction (GATED S5-02).
    assert verdict != "drift" and all(
        block["verdict"] != "drift" for block in per.values()
    ), "a --diff must never say drift"
    if cross and verdict.startswith("same"):
        # Section 6: a cross-subject --diff can never say `same`.
        verdict, code = "inconclusive (cross-subject)", EXIT["inconclusive"]
    body["refs_suggested"] = _refs_suggested(a_id, b_id, sensitivity, body.get("baseline_gap"))
    return _finish(header, body, verdict, code, [])


def _refs_suggested(
    target_id: Optional[str],
    own_id: Optional[str],
    sensitivity: Optional[str],
    baseline_gap: Optional[Mapping] = None,
) -> list[dict]:
    """The refs a result cert over this outcome must carry (section 2.1).

    Section 2.1 makes every cert id embedded in a cert's bytes a ref that cert carries, and the
    section 5.5 disclosure embeds one: the id of the baseline this cert's subject replaced. It is
    a cert the result depends on -- a reader who cannot resolve it cannot check the gap -- so it
    rides as a `previous` ref rather than as a bare string, and `make_result_cert` refuses the
    result if a caller drops it. Ids are not repeated: a previous baseline that is already the
    `target` or `own` of this outcome is carried once, under the role it already had.
    """
    out: list[dict] = []
    if target_id:
        out.append({"role": "target", "id": target_id})
    if own_id:
        out.append({"role": "own", "id": own_id})
    if sensitivity:
        out.append({"role": "sensitivity", "id": sensitivity})
    if isinstance(baseline_gap, Mapping):
        seen = {entry["id"] for entry in out}
        ids: list[str] = []
        previous_id = baseline_gap.get("previous_id")
        if isinstance(previous_id, str):
            ids.append(previous_id)
        for declared in baseline_gap.get("declared_previous") or ():
            if isinstance(declared, str):
                ids.append(declared)
        for cert_id in ids:
            if cert_id not in seen:
                seen.add(cert_id)
                out.append({"role": "previous", "id": cert_id})
    return out


# --------------------------------------------------------------------------- verify --ref


def _observed_subject(runner: Any, cert_subject: Mapping) -> dict:
    """The subject the runner reports it will actually run, over the cert's advisory fields.

    Identity comes from the runner and only from the runner: every ``S_identity`` field of the
    cert's kind is dropped before the report is laid on top, so no identity field can survive
    from the cert into what is compared against the cert.  Raises ``SubjectUnavailable`` when
    the runner does not report -- there is no fallback (see the module docstring).
    """
    reported = runnermod.reported_identity(runner, cert_subject)
    dropped = set(runnermod.SUBJECT_IDENTITY.get(cert_subject.get("kind"), ())) | {"kind"}
    merged = {k: copy.deepcopy(v) for k, v in cert_subject.items() if k not in dropped}
    merged.update(reported)
    return merged


def _observed_recipe(cert_recipe: Mapping, environment: Optional[Mapping]) -> dict:
    """The verifier's own harness fields for the skew comparison (section 2.3)."""
    out = {
        "harness": copy.deepcopy(cert_recipe.get("harness")),
        "env_lock_sha256": cert_recipe.get("env_lock_sha256"),
    }
    if isinstance(environment, Mapping):
        if isinstance(environment.get("harness"), Mapping):
            out["harness"] = copy.deepcopy(environment["harness"])
        if isinstance(environment.get("env_lock_sha256"), str):
            out["env_lock_sha256"] = environment["env_lock_sha256"]
    return out


def ref(
    cert: dict,
    runner: Any,
    resolver: Any,
    *,
    confirm: bool = True,
    environment: Optional[dict] = None,
) -> VerifyOutcome:
    """Re-run the cert's recipe now and compare (section 6, Appendix D step 4).

    Any channel above the floor triggers the mandatory confirmation run when ``confirm`` is
    true; without it an exceedance stays ``exceeds_floor`` (exit 2) and never reaches ``drift``.
    A runner that raises is ``unavailable`` (exit 5) with what was attempted recorded.
    """
    c = _mapping("cert", cert)
    cert_id = _cert_id(c)
    header = "styxx verify --ref"

    gated = _gate_certs("ref", (("cert", c),), resolver, cert_id, header)
    if gated is not None:
        return gated

    body = _shell("ref", cert_id)
    body["floor_owner"] = "ref"
    body["floor_owner_id"] = cert_id

    if c.get("type") != "fingerprint":
        mismatched = [f"cert.type:{c.get('type')!r}"]
        body["mismatched"] = mismatched
        return _finish(header, body, "mismatch", EXIT["mismatch"], mismatched)

    cert_body = c.get("body") if isinstance(c.get("body"), Mapping) else {}
    cert_subject = c.get("subject") if isinstance(c.get("subject"), Mapping) else {}
    cert_recipe = c.get("recipe") if isinstance(c.get("recipe"), Mapping) else {}

    battery_id = cert_recipe.get("battery")
    battery = _resolve(resolver, battery_id) if isinstance(battery_id, str) else None
    if battery is _NO_RESOLVER or not isinstance(battery, Mapping):
        body["invalid_reasons"] = [f"battery:{battery_id}: does not resolve"]
        return _finish(header, body, "invalid", EXIT["invalid"], [])

    attempted = {
        "battery": battery_id,
        "recipe.harness": cert_recipe.get("harness"),
        "subject.kind": cert_subject.get("kind"),
    }

    try:
        # Section 6 exit 5, and the one place this must not be lenient: a runner that will not
        # say what it runs leaves nothing to check, so nothing is compared and no verdict is
        # signed.  The reason is printed verbatim rather than wrapped in a class name.
        observed_subject = _observed_subject(runner, cert_subject)
    except runnermod.SubjectUnavailable as exc:
        body["attempted"] = attempted
        body["coverage"] = "unavailable"
        body["unavailable_reason"] = str(exc)
        return _finish(header, body, "unavailable", EXIT["unavailable"], [])

    try:
        # Section 2.2 environment, OBSERVED: it comes from the runner, the way the subject does.
        # `environment` (the CLI's `--environment <file>`) may only ANNOTATE what was reported --
        # a `harness` block, an `env_lock_sha256` -- and a leaf of it that contradicts an observed
        # one is `EnvironmentUnavailable`, exit 5, no cert.  A synthetic runner observes nothing
        # and its supplied environment stands; its certs carry `synthetic` and can be a baseline
        # for nothing (`runner.observed_environment`, `cert.comparable`).
        observed_env = runnermod.observed_environment(runner, environment)
    except runnermod.SubjectUnavailable as exc:
        # Verbatim, like the subject's: the message already names the member and the cause.
        body["attempted"] = attempted
        body["coverage"] = "unavailable"
        body["unavailable_reason"] = str(exc)
        return _finish(header, body, "unavailable", EXIT["unavailable"], [])
    except Exception as exc:  # a stranger's verifier reports; it does not crash
        body["attempted"] = attempted
        body["coverage"] = "unavailable"
        body["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        return _finish(header, body, "unavailable", EXIT["unavailable"], [])

    # What this verifier is about to produce, as a cert-shaped object `comparable` can read: the
    # subject the RUNNER reported (never the cert's own), the cert's recipe (a --ref re-runs it),
    # and the synthetic marker of the runner that is about to run. The marker is why a mock
    # cannot verify a measured cert: `comparable` returns `body.synthetic` and the outcome is a
    # mismatch with no distances, no verdict and nothing a challenge can be built from
    # (`styxx/v8/cert.py`, "Decisions", C-MOCK).
    observed_body: dict[str, Any] = {}
    if runnermod.is_synthetic(runner):
        observed_body[certmod.SYNTHETIC] = True
    entries = certmod.comparable(
        c, {"subject": observed_subject, "recipe": cert_recipe, "body": observed_body}
    )
    identity_names = set(floormod.IDENTITY_FIELDS)
    identity_diff = [f for f in floormod.IDENTITY_FIELDS if f"subject.{f}" in entries]
    hard = [
        e
        for e in entries
        if not (e.startswith("subject.") and e.split(".", 1)[1] in identity_names)
    ]
    if hard:
        # A --ref must re-run the SAME subject; `cross-subject` is a --diff affordance (section 6).
        body["mismatched"] = hard
        body["identity_diff"] = identity_diff
        return _finish(header, body, "mismatch", EXIT["mismatch"], hard)

    try:
        new_body = FP.run_fingerprint(
            runner,
            observed_subject,
            cert_recipe,
            battery,
            run_index=0,
            nuisance={},
            order=None,
            redacted=bool(cert_body.get("redacted", False)),
        )
    except Exception as exc:
        body["attempted"] = attempted
        body["coverage"] = "unavailable"
        body["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        return _finish(header, body, "unavailable", EXIT["unavailable"], [])

    not_covered = _not_covered(cert_body)
    coverage_diff = _coverage_diff(
        not_covered, cert_subject.get("environment"), observed_env
    )
    covered = not coverage_diff
    skew = certmod.skew_fields(
        {"recipe": cert_recipe}, {"recipe": _observed_recipe(cert_recipe, observed_env)}
    )
    floors = _floor_map(cert_body)
    roles = _roles_of(cert_body)

    per, skipped, anchor_flips = _per_channel(
        cert_body, new_body, floors, roles=roles, covered=covered, skew=bool(skew)
    )
    exceeded = any(
        block["distance"] is not None
        and block["floor"] is not None
        and block["distance"] > block["floor"]
        for block in per.values()
    )
    confirmation_body: Optional[dict] = None
    if exceeded and confirm and covered and not skew:
        try:
            confirmation_body = FP.run_fingerprint(
                runner,
                observed_subject,
                cert_recipe,
                battery,
                run_index=0,
                nuisance={},
                order=None,
                redacted=bool(cert_body.get("redacted", False)),
            )
        except Exception as exc:
            body["attempted"] = attempted
            body["coverage"] = "unavailable"
            body["unavailable_reason"] = f"confirmation run: {type(exc).__name__}: {exc}"
            return _finish(header, body, "unavailable", EXIT["unavailable"], [])
        per, skipped, anchor_flips = _per_channel(
            cert_body,
            new_body,
            floors,
            roles=roles,
            covered=covered,
            skew=bool(skew),
            confirmation_body=confirmation_body,
        )

    sensitivity = cert_body.get("sensitivity")
    sensitivity = sensitivity if isinstance(sensitivity, str) else None

    body["new_run"] = new_body
    body["confirmation_run"] = confirmation_body
    body["per_channel"] = per
    body["skipped_channels"] = skipped
    body["anchor_flips"] = anchor_flips
    body["coverage"] = "within" if covered else "beyond-floor-coverage"
    body["coverage_diff"] = coverage_diff
    body["covers"] = _covers(cert_body)
    body["not_covered"] = not_covered
    body["skew_fields"] = skew
    body["identity_diff"] = identity_diff
    body["sensitivity"] = sensitivity
    body["environment"] = observed_env
    body["environment_diff"] = _environment_diff(cert_subject.get("environment"), observed_env)
    body["confirm"] = bool(confirm)
    # What this verifier ran, from its own side: the runner's reported identity and the recipe
    # core it executed. `challenge_body` copies both into the section 9 body so a reader holding
    # only a challenge can see what produced the distances (C3), and the marker rides with them.
    body["observed_subject"] = copy.deepcopy(observed_subject)
    body["observed_recipe_core"] = certmod.recipe_core(dict(cert_recipe))
    if observed_body.get(certmod.SYNTHETIC):
        body[certmod.SYNTHETIC] = True
    # Section 5.5: a --ref against the CURRENT baseline is exactly the reading BASELINE-CHOICE
    # attacks, so the announcement belongs here. Set only when a baseline was replaced.
    gap = _baseline_gap(c, cert_body, resolver)
    if gap is not None:
        body["baseline_gap"] = gap
    if cert_subject.get("kind") == "alias":
        body["alias_note"] = ALIAS_NOTE

    verdict, code = floormod.overall(
        {c2: block["verdict"] for c2, block in per.items()},
        identity_diff=identity_diff,
        skipped=skipped,
        sensitivity_present=sensitivity is not None,
    )
    body["refs_suggested"] = _refs_suggested(cert_id, None, sensitivity, body.get("baseline_gap"))
    return _finish(header, body, verdict, code, [])


# --------------------------------------------------------------------------- section 9


def challenge_body(
    outcome: VerifyOutcome, environment: dict, *, note: Optional[str] = None
) -> dict:
    """The section 9 challenge body built from a verification the challenger just ran.

    ``{per_channel, coverage, environment, subject, recipe_core[, note]}``.  The coverage word
    is the outcome's own; an outcome that compared nothing (exit 3, 4 or 5) cannot become a
    challenge and raises.

    ``subject`` and ``recipe_core`` are the C3 repair
    ------------------------------------------------
    A challenge used to carry ``{per_channel, coverage, environment}`` and an envelope with
    ``subject: {}`` and ``recipe: {}``, so a reader holding the cert could not tell from any
    signed byte what produced the distances -- the only pointer was the ``own`` ref, and
    nothing validated it (``papers/v8/challenge_and_attack_2026_09_09``, C1 and C3).  Section 9
    puts the identity computation in the client's hands ON PURPOSE ("nothing in the body asserts
    it"), and that sentence is about VALIDITY: rule 1 is still computed from the two certs and
    nothing here changes it.  What the body now carries is the challenger's own report of what
    it ran, which is a different thing from an assertion about the target -- and section 2.7
    keeps the ENVELOPE's ``subject``/``recipe`` out of a challenge, so the body is the only
    place a self-description can go.

    It is not decoration: ``Log._check_challenge_subject`` refuses a challenge whose body
    disagrees with the ``own`` cert it names, so the two can never drift, and a reader who holds
    only the challenge gets a statement the log has already checked against the cert it points
    at.  ``subject`` is the S_identity of section 2.2 (plus ``synthetic`` when a synthetic
    runner produced the distances); ``recipe_core`` is section 2.3's four fields.
    """
    if not isinstance(outcome, VerifyOutcome):
        raise TypeError(f"outcome must be a VerifyOutcome, got {type(outcome).__name__}")
    env = _mapping("environment", environment)
    src = outcome.result_body
    coverage = src.get("coverage")
    if coverage not in ("within", "beyond-floor-coverage"):
        raise ValueError(
            f"a challenge needs a coverage word of 'within' or 'beyond-floor-coverage', "
            f"got {coverage!r} (verdict {outcome.verdict!r})"
        )
    per = src.get("per_channel")
    if not isinstance(per, Mapping) or not per:
        raise ValueError("a challenge needs at least one compared channel; this outcome has none")
    observed_subject = src.get("observed_subject")
    observed_core = src.get("observed_recipe_core")
    if not isinstance(observed_subject, Mapping) or not isinstance(observed_core, Mapping):
        raise ValueError(
            "a challenge needs the challenger's own report of what it ran; this outcome carries "
            "no 'observed_subject'/'observed_recipe_core', which is what a result of "
            "`verify --diff` looks like -- a diff compares two certs and runs nothing, so it "
            "has no observation to file (section 9 challenges a cert by REPRODUCING it)"
        )
    body: dict[str, Any] = {
        "per_channel": {
            channel: {
                "distance": block.get("distance"),
                "target_floor": block.get("floor"),
            }
            for channel, block in sorted(per.items())
        },
        "coverage": coverage,
        "environment": env,
        # S_identity only (section 2.2), which is what the reader's rule 1 and the log's
        # `_check_challenge_self_report` compare; the advisory fields the runner echoed
        # (`model_family`, `environment`) would say nothing checkable, and `environment` is
        # already the body's own member.
        "subject": certmod.identity_fields(dict(observed_subject)),
        "recipe_core": certmod.recipe_core(dict(observed_core)),
    }
    if src.get(certmod.SYNTHETIC):
        body[certmod.SYNTHETIC] = True
    if note is not None:
        if not isinstance(note, str):
            raise TypeError(f"note must be a str, got {type(note).__name__}")
        body["note"] = note
    return body


# --------------------------------------------------------------------------- the result cert


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_refs(refs: Any) -> list[dict]:
    if refs is None:
        return []
    if isinstance(refs, Mapping) or isinstance(refs, (str, bytes)):
        raise TypeError("refs must be a list of (role, id) pairs or {'role','id'} dicts")
    out: list[dict] = []
    for entry in refs:
        if isinstance(entry, Mapping):
            role, cert_id = entry.get("role"), entry.get("id")
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
            role, cert_id = entry[0], entry[1]
        else:
            raise TypeError(f"refs entry must be (role, id) or {{'role','id'}}, got {entry!r}")
        if not isinstance(role, str) or not isinstance(cert_id, str):
            raise TypeError(f"refs entry must carry str role and str id, got {entry!r}")
        out.append({"role": role, "id": cert_id})
    return out


def make_result_cert(
    outcome: VerifyOutcome,
    issuer_key: Any,
    private_seed: bytes,
    refs: Any,
    subject: dict,
    recipe: dict,
    *,
    created: Optional[str] = None,
) -> dict:
    """A signed ``result`` cert of kind ``verify`` carrying ``outcome.result_body``.

    ``issuer_key`` is either the ``ed25519:...`` wire string or a whole ``{name, key}`` issuer
    block.  ``refs`` are ``(role, id)`` pairs or ``{'role','id'}`` dicts -- usually
    ``outcome.result_body['refs_suggested']``.  Raises ValueError when the body embeds a cert id
    that ``refs`` does not carry (section 2.1, A-09).
    """
    if not isinstance(outcome, VerifyOutcome):
        raise TypeError(f"outcome must be a VerifyOutcome, got {type(outcome).__name__}")
    if isinstance(issuer_key, Mapping):
        issuer = copy.deepcopy(dict(issuer_key))
    elif isinstance(issuer_key, str):
        issuer = {"name": _DEFAULT_ISSUER_NAME, "key": issuer_key}
    else:
        raise TypeError(
            f"issuer_key must be an 'ed25519:...' string or a {{name, key}} dict, got "
            f"{type(issuer_key).__name__}"
        )
    core = {
        "styxx": SCHEMA_VERSION,
        "type": "result",
        "created": created if created is not None else _now(),
        "issuer": issuer,
        "subject": copy.deepcopy(_mapping("subject", subject)),
        "recipe": copy.deepcopy(_mapping("recipe", recipe)),
        "body": copy.deepcopy(outcome.result_body),
        "refs": _normalize_refs(refs),
    }
    carried = {entry["id"] for entry in core["refs"]}
    missing = sorted(certmod.embedded_ids(core) - carried)
    if missing:
        raise ValueError(
            "refs does not carry every cert id the result body embeds (section 2.1): "
            + ", ".join(missing)
        )
    return certmod.sign(core, private_seed)
