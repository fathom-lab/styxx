"""Model access for styxx.v8 (spec section 3): the ``Runner`` protocol and a deterministic mock.

A runner turns battery items into ``ItemResult`` records.  The real one lives in
``styxx.v8.runner_hf`` (torch/transformers, imported lazily there); this module is pure
CPU and standard library plus ``styxx.v8.jcs``.

The mock is a hash-derived function of what the spec says a fingerprint depends on: the
subject's identity fields (section 2.2), the item id, and ``recipe_core`` (section 2.3).
Two keys are deliberately taken OUT of the hashed base so that the perturbation families
of section 4 can be modelled at all:

* ``decoding.batch_size`` and ``decoding.padding_side`` -- they are nuisance factors; if the
  base output depended on them every item would flip under a batch-size change and the
  delta-2 sweep would measure nothing.  Only ``nuisance_items`` respond to batch size.
* ``precision`` -- the delta-1 family varies it; only ``precision_items[precision]`` respond.

Perturbation rules (each perturbed output differs from the base output):

* ``nuisance_items``: when ``decoding.batch_size != 1`` the item's output is replaced by a
  variant keyed on the parity of the item's position in the order the caller gave, so a
  permutation at batch > 1 moves it again.  At batch 1 the item is never touched, in any
  order -- the probe on this box (``papers/v8/probe_batch_invariance_2026_09_08``) found
  that a batch-1 rerun moved nothing and a batch-size change alone moved outputs, and the
  fingerprint contract wants ``nuisance_items`` to flip only under batch > 1.
* ``drift_items``: always the drift variant.
* ``precision_items[p]``: the precision variant when ``subject.precision == p``.

Log-probability fields (``seq_logprob``, ``topk``, ``margin_by_position``, ``stay``) are derived
from the same seed as the token ids, so they change whenever the output changes; with
``logprobs=False`` they are ``None`` (an alias provider that returns no log-probs).

``subject`` is part of the contract, not an option
--------------------------------------------------
Every runner MUST implement ``subject(requested)``: the ``kind`` and every ``S_identity`` field
(section 2.2, ``SUBJECT_IDENTITY``) of the subject it will ACTUALLY run.  ``verify --ref`` and
``run_fingerprint`` compare that report against the subject the cert names and refuse when the
two disagree; a runner that does not report leaves them nothing to check, so they refuse it too
(``SubjectUnavailable`` -> exit 5) rather than believing the cert about itself.

This is the repair of the defect recorded as C2 in
``papers/v8/challenge_and_attack_2026_09_09``: ``verify --ref`` read an OPTIONAL ``subject``
member, ``TransformersRunner`` had none, so the guard compared the cert against itself, could
not fail, and ``--dtype float16`` against a bf16 cert produced a signed ``drift`` accusation
from weights the cert never named.  An optional guard is not a guard.

A runner is free to serve whatever it is asked for -- ``MockRunner`` is a pure function of the
identity it is handed and so reports ``requested`` back -- but a runner holding fixed weights
MUST report what it holds and MUST NOT echo an identity field it did not observe.
``MockRunner(loaded_subject=...)`` is the test double for exactly that: it reports the subject
it "loaded" and runs it, whatever the caller asks for.

``environment`` is part of the contract too
-------------------------------------------
Every runner MUST implement ``environment()``: the ``runtime`` and ``hardware`` blocks of
section 2.2 for the run it is about to make, reported by the thing that makes it.  ``verify --ref``
takes the environment it records from there and NOWHERE else, the way it takes the subject from
``subject(requested)``; ``--environment <file>`` may only ANNOTATE what was observed (add a
``harness`` block, an ``env_lock_sha256``) and a field of it that contradicts an observed one is
refused (``EnvironmentUnavailable`` -> exit 5), never written over the observation.

This is the repair of S-DEVICE (``papers/v8/challenge_and_attack_2026_09_09`` and the third
adversarial pass): ``TransformersRunner.environment()`` reported the card the BOX holds rather
than the device the RUN uses, ``device`` is not an S_identity field and cannot become one
(section 2.2 says the environment is never identity), and ``--environment <file>`` replaced the
observed block wholesale in both the result body and the challenge body.  One CPU forward pass
therefore signed a record naming an RTX 4070 it never touched, with ``mismatched: []`` and
``identity_diff: []``.  An environment nobody observed is not an environment; it is a sentence
the issuer wrote.

A **synthetic** runner is the one exception, and it is not a hole: it computes from a hash, so
it observes no environment at all, and every cert built from it carries ``synthetic`` in the
signed bytes and cannot be compared against, diffed against or used to challenge a measured cert
(``styxx.v8.cert.comparable``).  ``observed_environment`` therefore lets a caller supply the
environment for a synthetic runner -- that is how the coverage and skew paths are exercised
without a second box -- and refuses to for a runner that claims to be a model.

``synthetic`` is part of the contract too
-----------------------------------------
A runner that computes its outputs from a hash instead of from a model sets ``synthetic = True``
as a class member, and every body built from it carries ``synthetic: true`` in the SIGNED bytes
(``styxx.v8.fingerprint.run_fingerprint``, ``styxx.v8.verify``).

Echoing ``requested`` back is what made the section 6 guard vacuous for this mock: a runner that
will serve ANY identity satisfies every comparison against every cert, so ``--runner mock`` --
which is also the CLI's default when ``--runner`` is omitted -- signed a section 9 challenge
against a published canonical carrying distances from a model that was never loaded
(``papers/v8/challenge_and_attack_2026_09_09``, C-MOCK). The echo is KEPT: it is what makes the
mock a deterministic function of the identity it is handed, which is what the fingerprint tests
measure. What changes is that the numbers it produces are labelled, and
``styxx.v8.cert.comparable`` reads that label as a hard difference -- so a synthetic run cannot
be a baseline for, a diff against, or a challenge to a measured one. See ``styxx/v8/cert.py``
under "Decisions" for why the marker sits in the body rather than in the subject.

A real runner does not set the flag, and MUST NOT: ``synthetic`` absent means "a model produced
this", and that is the claim the flag exists to keep honest.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Iterable, Mapping, Optional, Protocol, TypedDict, runtime_checkable

from styxx.v8.jcs import canonical_bytes

__all__ = [
    "ENVIRONMENT_BLOCKS",
    "EnvironmentUnavailable",
    "ItemResult",
    "Runner",
    "MockRunner",
    "SUBJECT_IDENTITY",
    "SubjectUnavailable",
    "SYNTHETIC",
    "annotate_environment",
    "environment_leaves",
    "identity_key",
    "core_key",
    "is_synthetic",
    "mock_derived_items",
    "mock_token_id_digests",
    "observed_environment",
    "reported_environment",
    "reported_identity",
    "subject_identity",
]

# The body member a synthetic runner's numbers carry (``styxx.v8.cert.SYNTHETIC``; restated here
# so this module keeps its "standard library plus jcs" dependency, and pinned against ``cert`` by
# ``tests/test_v8_subject_guard.py``).
SYNTHETIC = "synthetic"

# Section 2.2: S_identity per kind.  ``precision`` is an identity field for comparability
# but is held out of the mock's base key (see the module docstring).
_WEIGHTS_IDENTITY = (
    "hf_repo",
    "revision",
    "weights_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "generation_config_sha256",
)
_ALIAS_IDENTITY = ("provider", "alias", "region")

# Section 2.2 S_identity, whole, per kind -- ``precision`` included.  This is the same table as
# ``cert.IDENTITY_FIELDS``; it is restated here so that this module keeps its "standard library
# plus jcs" dependency, and ``tests/test_v8_subject_guard.py`` pins the two against each other.
SUBJECT_IDENTITY: dict[str, tuple[str, ...]] = {
    "weights": _WEIGHTS_IDENTITY + ("precision",),
    "alias": _ALIAS_IDENTITY,
}

# Section 2.2: the two blocks a ``subject.environment`` carries, and the two a runner OBSERVES.
# Anything else in an environment mapping (``harness``, ``env_lock_sha256``) is the verifier's
# own description of itself, which the runner has no way to observe and a caller may supply.
ENVIRONMENT_BLOCKS = ("runtime", "hardware")

_MOCK_VOCAB = 32000
_MOCK_MAX_TOKENS = 12
_TOPK_K = 5
_TOPK_POSITIONS = 8  # positions 0..min(7, n_generated-1), section 3.2
_STAY_TEMPERATURE = 0.2


class ItemResult(TypedDict):
    item_id: str
    token_ids: list[int]
    output_text: str
    n_generated: int
    seq_logprob: float | None
    topk: list[dict] | None
    margin_by_position: list[float] | None
    stay: float | None


class SubjectUnavailable(Exception):
    """A runner did not report the subject it runs, or reported an incomplete one.

    Section 6 exit 5: "the subject or the environment could not be obtained".  It is never a
    verdict about the model -- it is the verifier saying it cannot check what it ran.
    """


class EnvironmentUnavailable(SubjectUnavailable):
    """The environment of a run could not be obtained, or was obtained and then contradicted.

    The second half is the one worth naming.  A runner reported ``hardware.gpu: "cpu"`` and the
    caller handed ``--environment`` a file saying ``"NVIDIA GeForce RTX 4070 Laptop GPU"``: the
    environment WAS obtained, and what cannot be obtained is one the verifier can stand behind.
    Section 6 exit 5 is the honest code for that -- it produces no cert, no challenge and no
    verdict, which is the only outcome that does not put a card in the signed bytes.

    A subclass of ``SubjectUnavailable`` so that every caller already catching the exit-5 case
    catches this one too.
    """


@runtime_checkable
class Runner(Protocol):
    def run(self, items: list[dict], recipe: dict, subject: dict) -> list[ItemResult]:
        """Run ``items`` (``[{item_id, prompt_text}]``) in the order given; results come back
        in that same order.  ``batch_size``/``padding_side`` come from ``recipe.decoding``."""
        ...

    def environment(self) -> dict:
        """REQUIRED.  ``{"runtime": {...}, "hardware": {...}}`` for ``subject.environment``
        (section 2.2), describing the run this runner is about to make.

        ``hardware`` names the device the run COMPUTES ON, not the devices the box holds: a run
        on the CPU reports a CPU while a card sits idle beside it.  A runner is the only thing
        in the system that can observe this, which is why it is an obligation and not an
        option; ``verify --ref`` records what comes back here and never what it was told.
        """
        ...

    def subject(self, requested: Mapping[str, Any]) -> Mapping[str, Any]:
        """REQUIRED.  The subject this runner will ACTUALLY run, as ``kind`` plus every
        ``S_identity`` field of that kind (``SUBJECT_IDENTITY``); advisory fields
        (``model_family``, ``environment``, ...) may be omitted and are not read.

        ``requested`` is the subject the caller wants -- the one a cert names.  A runner that
        can serve any identity (``MockRunner``) may report it back; a runner holding fixed
        weights MUST report what it holds and MUST NOT copy a field it did not observe.  A
        field the runner cannot observe is omitted, and every caller treats an omission as
        ``SubjectUnavailable`` -- fail closed, because a missing field is the one an attacker
        would choose.
        """
        ...


def subject_identity(subject: Mapping[str, Any]) -> dict:
    """``kind`` plus every S_identity field of that kind; missing fields read as ``None``."""
    src = subject if isinstance(subject, Mapping) else {}
    kind = src.get("kind")
    out: dict[str, Any] = {"kind": kind}
    for name in SUBJECT_IDENTITY.get(kind if isinstance(kind, str) else "", ()):
        out[name] = copy.deepcopy(src.get(name))
    return out


def is_synthetic(runner: Any) -> bool:
    """True when ``runner`` declares it is not a model (``runner.synthetic is True``).

    The default is False, which is the only honest default for a class that never heard of the
    member: a runner that does not declare itself synthetic is treated as a model, and a model's
    body carries no marker. Every path that builds a body from a runner asks this and stamps
    ``cert.SYNTHETIC`` into what it produces.
    """
    return getattr(runner, "synthetic", False) is True


def reported_identity(runner: Any, requested: Mapping[str, Any]) -> dict:
    """What ``runner`` says it will actually run, as ``kind`` + the S_identity fields of that kind.

    Raises ``SubjectUnavailable`` when the runner has no ``subject`` member, when the member
    does not return a mapping, when the reported ``kind`` is not one section 2.2 defines, or
    when any S_identity field of that kind is missing or empty.  A caller may compare what
    comes back against a cert; it may never fall back to the cert's own subject, which is the
    comparison that cannot fail.
    """
    name = type(runner).__name__
    member = getattr(runner, "subject", None)
    if member is None:
        raise SubjectUnavailable(
            f"runner {name} does not report the subject it runs: a Runner must implement "
            "subject(requested) naming the identity it actually loaded (styxx/v8/runner.py); "
            "without it nothing can check that the cert's subject is what was run"
        )
    try:
        reported = member(dict(requested) if isinstance(requested, Mapping) else {}) if callable(member) else member
    except SubjectUnavailable:
        raise
    except Exception as exc:
        raise SubjectUnavailable(f"runner {name}.subject(): {type(exc).__name__}: {exc}") from None
    if not isinstance(reported, Mapping):
        raise SubjectUnavailable(
            f"runner {name}.subject() returned {type(reported).__name__}, not a subject mapping"
        )
    kind = reported.get("kind")
    if kind not in SUBJECT_IDENTITY:
        raise SubjectUnavailable(
            f"runner {name}.subject() reported kind {kind!r}; section 2.2 defines "
            f"{sorted(SUBJECT_IDENTITY)}"
        )
    out: dict[str, Any] = {"kind": kind}
    missing: list[str] = []
    for field in SUBJECT_IDENTITY[kind]:
        value = reported.get(field)
        if value is None or value == "":
            missing.append(field)
        else:
            out[field] = copy.deepcopy(value)
    if missing:
        raise SubjectUnavailable(
            f"runner {name}.subject() left {', '.join(missing)} unreported; a subject identity "
            "with a hole in it cannot be compared against a cert"
        )
    return out


_MISSING = object()
_BLOCKED = object()


def environment_leaves(environment: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """``[(dotted path, value)]`` for every non-mapping leaf, sorted -- the section 5.4 vocabulary.

    ``{"hardware": {"gpu": "cpu"}}`` -> ``[("hardware.gpu", "cpu")]``.  An empty mapping is a
    leaf of its own (``("harness", {})``) so that "the field is present and empty" and "the field
    is absent" stay different facts.
    """
    out: list[tuple[str, Any]] = []
    for key in sorted(k for k in environment if isinstance(k, str)):
        value = environment[key]
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.extend(environment_leaves(value, path + "."))
        else:
            out.append((path, value))
    return out


def _at(obj: Any, path: str) -> Any:
    """The value at a dotted path: ``_MISSING`` when absent, ``_BLOCKED`` when a parent is a leaf."""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return _BLOCKED
        if part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _put(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def reported_environment(runner: Any) -> dict:
    """What ``runner`` says about the environment it will actually run in (section 2.2).

    Raises ``EnvironmentUnavailable`` when the runner has no ``environment`` member, when it
    raises, when it returns something that is not a mapping, or when either of the two blocks a
    subject's environment is made of (``ENVIRONMENT_BLOCKS``) is missing or is not a mapping.
    Fail closed, for the same reason ``reported_identity`` does: a missing block is the one an
    attacker would choose, and there is nothing else in the process that saw the hardware.
    """
    name = type(runner).__name__
    member = getattr(runner, "environment", None)
    if member is None:
        raise EnvironmentUnavailable(
            f"runner {name} does not report the environment it runs in: a Runner must implement "
            "environment() naming the runtime and the device it actually uses "
            "(styxx/v8/runner.py); without it nothing can check what the cert records"
        )
    try:
        reported = member() if callable(member) else member
    except EnvironmentUnavailable:
        raise
    except Exception as exc:
        raise EnvironmentUnavailable(f"runner {name}.environment(): {type(exc).__name__}: {exc}") from None
    if not isinstance(reported, Mapping):
        raise EnvironmentUnavailable(
            f"runner {name}.environment() returned {type(reported).__name__}, not a mapping"
        )
    missing = [b for b in ENVIRONMENT_BLOCKS if not isinstance(reported.get(b), Mapping)]
    if missing:
        raise EnvironmentUnavailable(
            f"runner {name}.environment() left {', '.join(missing)} unreported; section 2.2's "
            "environment is a runtime block and a hardware block, and a hole in either is a "
            "field the cert would have to take from somewhere nobody observed"
        )
    return {k: copy.deepcopy(v) for k, v in reported.items()}


def annotate_environment(observed: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict:
    """``observed`` with ``annotation``'s NEW leaves added; a contradiction raises.

    An annotation may say things the runner cannot observe -- the verifier's own ``harness``
    block, an ``env_lock_sha256`` -- and those are what the skew comparison of section 2.3 reads.
    It may not say a different value for a leaf the runner reported: the runner is the only
    witness to the hardware, so a disagreement is not a merge, it is one of the two being wrong,
    and the one that came from a file is the one nobody measured.  Every conflicting path is
    named in the refusal, not just the first.
    """
    out = {k: copy.deepcopy(v) for k, v in dict(observed).items()}
    conflicts: list[str] = []
    for path, value in environment_leaves(dict(annotation)):
        current = _at(out, path)
        if current is _MISSING:
            _put(out, path, copy.deepcopy(value))
        elif current is _BLOCKED or current != value:
            shown = "a value under an observed leaf" if current is _BLOCKED else repr(current)
            conflicts.append(f"{path}: observed {shown}, asserted {value!r}")
    if conflicts:
        raise EnvironmentUnavailable(
            "the environment supplied contradicts the environment observed: "
            + "; ".join(conflicts)
            + " -- an environment is recorded because it was observed (section 2.2); a file may "
            "annotate what the runner reported and may not overwrite it"
        )
    return out


def observed_environment(runner: Any, annotation: Optional[Mapping[str, Any]] = None) -> dict:
    """The environment a run records: OBSERVED from ``runner``, annotated, never replaced.

    ``annotation`` is what ``--environment <file>`` carries.  For a runner that claims to be a
    model it may only add leaves the runner did not report (``annotate_environment``).  For a
    **synthetic** runner it stands on its own: such a runner computes from a hash and observes no
    hardware, and every cert built from it already carries ``synthetic`` in the signed bytes and
    is refused as a baseline for, a diff against or a challenge to a measured cert -- so the
    environment it names is fenced off from every claim that could reach a stranger.
    """
    if annotation is not None and is_synthetic(runner):
        return {k: copy.deepcopy(v) for k, v in dict(annotation).items()}
    observed = reported_environment(runner)
    if annotation is None:
        return observed
    return annotate_environment(observed, annotation)


def identity_key(subject: Mapping[str, Any]) -> dict:
    """The subject fields the mock's base output depends on (section 2.2 minus precision)."""
    kind = subject.get("kind")
    if kind == "weights":
        fields = _WEIGHTS_IDENTITY
    elif kind == "alias":
        fields = _ALIAS_IDENTITY
    else:
        raise ValueError(f"subject.kind must be 'weights' or 'alias', got {kind!r}")
    return {"kind": kind, **{f: subject.get(f) for f in fields}}


def core_key(recipe: Mapping[str, Any]) -> dict:
    """``recipe_core`` (section 2.3) with the two nuisance decoding fields removed."""
    decoding = dict(recipe.get("decoding") or {})
    decoding.pop("batch_size", None)
    decoding.pop("padding_side", None)
    return {
        "battery": recipe.get("battery"),
        "decoding": decoding,
        "chat_template_sha256": recipe.get("chat_template_sha256"),
        "system_prompt_sha256": recipe.get("system_prompt_sha256"),
    }


def _check_items(items: Iterable[Mapping[str, Any]]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, Mapping):
            raise ValueError(f"item {i} is not a mapping")
        item_id = it.get("item_id")
        prompt = it.get("prompt_text")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"item {i}: item_id must be a non-empty string")
        if not isinstance(prompt, str):
            raise ValueError(f"item {item_id!r}: prompt_text must be a string")
        if item_id in seen:
            raise ValueError(f"duplicate item_id {item_id!r}")
        seen.add(item_id)
        out.append(dict(it))
    return out


def _batch_size(recipe: Mapping[str, Any]) -> int:
    decoding = recipe.get("decoding") or {}
    bs = decoding.get("batch_size", 1)
    if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
        raise ValueError(f"decoding.batch_size must be an int >= 1, got {bs!r}")
    return bs


def _max_new_tokens(recipe: Mapping[str, Any]) -> int:
    decoding = recipe.get("decoding") or {}
    m = decoding.get("max_new_tokens", 64)
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError(f"decoding.max_new_tokens must be an int >= 1, got {m!r}")
    return m


class _Stream:
    """Deterministic byte stream: sha256(seed || counter)."""

    def __init__(self, seed: bytes) -> None:
        self._seed = seed
        self._counter = 0
        self._buf = b""

    def u32(self) -> int:
        if len(self._buf) < 4:
            self._buf += hashlib.sha256(self._seed + self._counter.to_bytes(4, "big")).digest()
            self._counter += 1
        v = int.from_bytes(self._buf[:4], "big")
        self._buf = self._buf[4:]
        return v

    def unit(self) -> float:
        return self.u32() / 4294967296.0


def _log_softmax(values: list[float]) -> list[float]:
    m = max(values)
    lse = m + math.log(math.fsum(math.exp(v - m) for v in values))
    return [v - lse for v in values]


def _derive(seed: bytes, max_new_tokens: int) -> tuple[list[int], list[list[int]], list[list[float]]]:
    """Token ids plus, per generated position, the top-5 ids and log-probs (top-1 = greedy)."""
    s = _Stream(seed)
    n = 1 + s.u32() % min(max_new_tokens, _MOCK_MAX_TOKENS)
    ids: list[int] = []
    top_ids: list[list[int]] = []
    top_lps: list[list[float]] = []
    for _ in range(n):
        cand: list[int] = []
        while len(cand) < _TOPK_K:
            t = s.u32() % _MOCK_VOCAB
            if t not in cand:
                cand.append(t)
        lps = [-0.5 * s.unit()]
        for _k in range(1, _TOPK_K):
            lps.append(lps[-1] - (0.2 + 3.0 * s.unit()))
        ids.append(cand[0])
        top_ids.append(cand)
        top_lps.append(lps)
    return ids, top_ids, top_lps


class MockRunner:
    """Deterministic stand-in for a model (see the module docstring for the rules)."""

    #: This runner computes from a hash, not from weights. Every cert minted from it carries
    #: ``subject.synthetic`` (``styxx.v8.cert.SYNTHETIC``) in the signed bytes.
    synthetic = True

    def __init__(
        self,
        *,
        nuisance_items: set[str] = frozenset(),
        drift_items: set[str] = frozenset(),
        precision_items: Mapping[str, set[str]] = {},
        logprobs: bool = True,
        loaded_subject: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.nuisance_items = frozenset(nuisance_items)
        self.drift_items = frozenset(drift_items)
        self.precision_items = {k: frozenset(v) for k, v in dict(precision_items).items()}
        self.logprobs = bool(logprobs)
        self.loaded_subject = dict(loaded_subject) if loaded_subject is not None else None

    def subject(self, requested: Mapping[str, Any]) -> dict:
        """The identity this mock will run.

        With no ``loaded_subject`` the mock is a pure function of the identity it is handed --
        it really does run whatever it is asked for, so it reports ``requested`` back and the
        guard is true and vacuous.  With one, the mock holds fixed "weights" like a real
        runner: it reports them and runs them whatever the caller asks, which is how a swapped
        model is modelled without a GPU.

        The echo is why ``synthetic`` exists: a runner that satisfies every request satisfies
        every guard, so the guard has to be told that this one is not a model at all. That is
        ``MockRunner.synthetic``, and it lands in every body this runner produces rather than
        in what this method returns -- see the module docstring.
        """
        return subject_identity(self.loaded_subject if self.loaded_subject is not None else requested)

    def environment(self) -> dict:
        return {
            "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
            "hardware": {"gpu": "none", "driver": "none", "count": 0},
        }

    def _variants(self, item_id: str, position: int, batch_size: int, precision: Any) -> list:
        v: list = []
        if item_id in self.nuisance_items and batch_size != 1:
            v.append(["nuisance", position % 2])
        if item_id in self.drift_items:
            v.append(["drift"])
        if precision is not None and item_id in self.precision_items.get(precision, frozenset()):
            v.append(["precision", precision])
        return v

    def run(self, items: list[dict], recipe: dict, subject: dict) -> list[ItemResult]:
        checked = _check_items(items)
        batch_size = _batch_size(recipe)
        max_new = _max_new_tokens(recipe)
        # A mock with fixed weights runs THOSE, not the ones it was asked for -- the caller is
        # supposed to have compared ``subject()`` against the cert before getting here.
        effective = self.loaded_subject if self.loaded_subject is not None else subject
        ident = identity_key(effective)
        core = core_key(recipe)
        precision = effective.get("precision")
        results: list[ItemResult] = []
        for position, it in enumerate(checked):
            item_id = it["item_id"]
            base_key = {"identity": ident, "item_id": item_id, "recipe_core": core}
            base_seed = hashlib.sha256(canonical_bytes(base_key)).digest()
            ids, top_ids, top_lps = _derive(base_seed, max_new)
            variants = self._variants(item_id, position, batch_size, precision)
            if variants:
                var_seed = hashlib.sha256(
                    canonical_bytes({"base": base_seed.hex(), "variants": variants})
                ).digest()
                v_ids, v_top_ids, v_top_lps = _derive(var_seed, max_new)
                if v_ids == ids:  # not expected; keeps "differs from base" a guarantee
                    v_ids = v_ids[:-1] + [(v_ids[-1] + 1) % _MOCK_VOCAB]
                    v_top_ids[-1][0] = v_ids[-1]
                ids, top_ids, top_lps = v_ids, v_top_ids, v_top_lps
            results.append(self._result(item_id, ids, top_ids, top_lps))
        return results

    def _result(self, item_id: str, ids: list[int], top_ids: list[list[int]], top_lps: list[list[float]]) -> ItemResult:
        text = " ".join(f"w{t}" for t in ids)
        n = len(ids)
        if not self.logprobs:
            return ItemResult(
                item_id=item_id, token_ids=ids, output_text=text, n_generated=n,
                seq_logprob=None, topk=None, margin_by_position=None, stay=None,
            )
        seq_lp = math.fsum(lps[0] for lps in top_lps)
        margins = [lps[0] - lps[1] for lps in top_lps]
        stay = math.fsum(_log_softmax([v / _STAY_TEMPERATURE for v in lps])[0] for lps in top_lps)
        topk = [
            {"pos": p, "ids": list(top_ids[p]), "lps": list(top_lps[p])}
            for p in range(min(_TOPK_POSITIONS, n))
        ]
        return ItemResult(
            item_id=item_id, token_ids=ids, output_text=text, n_generated=n,
            seq_logprob=seq_lp, topk=topk, margin_by_position=margins, stay=stay,
        )


# --------------------------------------------------------------------------- the mock oracle

# Every ``variants`` list ``MockRunner._variants`` can build, in its order (nuisance, then drift,
# then precision). Which one an item took depends on ``nuisance_items`` / ``drift_items`` /
# ``precision_items``, which are runner configuration and appear in no cert, so the oracle tries
# all of them. ``[]`` (the base, no variant) is the first entry.
def _mock_variant_lists(precision: Any) -> list[list]:
    out: list[list] = [[]]
    for nuisance in ([["nuisance", 0]], [["nuisance", 1]]):
        out.append(list(nuisance))
    out = [v for base in out for v in (base, base + [["drift"]])]
    if precision is not None:
        out = [v for base in out for v in (base, base + [["precision", precision]])]
    return out


def mock_token_id_digests(
    subject: Mapping[str, Any], recipe: Mapping[str, Any], item_id: str
) -> set[str]:
    """Every ``token_ids_sha256`` ``MockRunner`` can produce for ``item_id`` here.

    The mock's output is a pure function of ``identity_key(subject)``, ``item_id``,
    ``core_key(recipe)``, ``max_new_tokens`` and which variant list the item took -- all of which
    a fingerprint cert carries or this function enumerates. So a body's item digests can be
    checked against the mock's, by anyone, from the cert's own bytes.

    Raises ValueError for a subject or recipe the mock could not have run.
    """
    ident = identity_key(subject)
    core = core_key(recipe)
    max_new = _max_new_tokens(recipe)
    base_seed = hashlib.sha256(
        canonical_bytes({"identity": ident, "item_id": item_id, "recipe_core": core})
    ).digest()
    base_ids, _, _ = _derive(base_seed, max_new)
    out = {hashlib.sha256(canonical_bytes(base_ids)).hexdigest()}
    for variants in _mock_variant_lists(subject.get("precision")):
        if not variants:
            continue
        seed = hashlib.sha256(
            canonical_bytes({"base": base_seed.hex(), "variants": variants})
        ).digest()
        ids, _, _ = _derive(seed, max_new)
        if ids == base_ids:  # MockRunner.run's "differs from base" nudge
            ids = ids[:-1] + [(ids[-1] + 1) % _MOCK_VOCAB]
        out.add(hashlib.sha256(canonical_bytes(ids)).hexdigest())
    return out


def mock_derived_items(
    body: Mapping[str, Any], subject: Mapping[str, Any], recipe: Mapping[str, Any]
) -> list[str]:
    """The ``item_id``s in ``body.items`` whose ``token_ids_sha256`` ``MockRunner`` produces.

    THE MOCK ORACLE (C-MOCK-2). ``synthetic`` was a member the shipped tool always wrote and
    nothing ever re-derived, so deleting it from a mock's body and re-signing produced an
    unmarked fabrication that ``cert.comparable`` read as a measurement. A key nothing recomputes
    is a key the issuer sets; the repair is to stop asking the issuer. ``MockRunner`` computes
    from a hash, the hash's inputs are in the cert, so the numbers say for themselves whose they
    are: a body carrying the mock's digests IS the mock's output, marker or no marker.

    ``[]`` for a body no mock produced, and for a subject or recipe the mock could not have run
    (an unreadable one is not an accusation). The first item is tried alone before the rest, so
    the ordinary answer costs one item's derivations and not the battery's.
    """
    items = body.get("items")
    if not isinstance(items, list) or not items:
        return []
    heads = [it for it in items if isinstance(it, Mapping) and isinstance(it.get("item_id"), str)]
    if not heads:
        return []
    try:
        first = heads[0]
        if first.get("token_ids_sha256") not in mock_token_id_digests(
            subject, recipe, first["item_id"]
        ):
            return []
        return [
            it["item_id"]
            for it in heads
            if it.get("token_ids_sha256")
            in mock_token_id_digests(subject, recipe, it["item_id"])
        ]
    except (TypeError, ValueError):
        return []
