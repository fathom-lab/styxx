"""styxx.v8.cli -- the ``python -m styxx.v8`` command surface (contract section 9, spec section 11).

Two rules hold for every verb here, and the tests pin both:

1. **stdout is exactly one JSON object.**  The result on success, ``{"error": ...}`` on failure.
   Nothing else is ever written to stdout, and nothing at all is written to stderr; a caller can
   pipe the output straight into ``jq`` without filtering a banner out of it.
2. **the exit code is a value of ``consts.EXIT``.**  A ``verify`` passes through
   ``VerifyOutcome.exit_code`` verbatim -- never re-derived from the verdict string, which carries
   compounds (``same (transient)``, ``mismatch (schema version)``) that are not keys of ``EXIT``.
   Every other verb: ``0`` when it did what it was asked, ``4`` (``invalid``) when the request or
   its inputs are malformed or a claim does not check out, ``5`` (``unavailable``) when a named
   file or model cannot be reached.  There is no separate usage code -- ``EXIT`` has none, and
   inventing one would break rule 2.

The v8 surface lives ONLY under ``python -m styxx.v8`` (GATED S11-01: recommendation (a) --
namespace v8 and leave the 7.x top-level verbs alone until 8.1; the operator may reverse).
``styxx/cli.py`` is untouched by this module.

Verbs::

    key         generate | show
    prereg      noise-plan --runs R --nuisance <factor=values,...> --subject <spec> --battery <cert>
    fingerprint --subject <spec.json> --battery <cert> --key <pem> [--runs R] [--plan <id|file>] ...
    verify      --ref <cert> [--challenge] | --diff <a> <b>
    battery     select --sweep <record> --pool <cert> --n N --k K --perm-seed S
                fixed  --source <items.json> [--label <str>]
                pool   --source <items.json>
    log         init | append <cert> | prove <index> | sth | mirror --to <dir>
                verify-cert <cert> | verify-inclusion <proof> <sth>
                verify-consistency <sth1> <sth2> | verify-sth <sth>

``run(argv) -> (exit_code, payload)`` is the whole CLI as a pure function of its arguments; it
touches the filesystem but never ``sys.exit`` and never prints, so a test can drive it in-process.
``main(argv=None)`` prints the payload and returns the code.

Decisions
---------

**Nothing is signed while section 9 rule 1 is uncomputed.** ``--challenge`` with an ``--own``
this command cannot resolve used to sign anyway and print ``challenge_validity: null``, so the
gap existed in stdout and in no signed byte (C-MINT-UNCOMPUTED of
``papers/v8/challenge_and_attack_2026_09_09``). It refuses now; the reasoning is on
``_challenge``.

**``--runner mock`` is still allowed to mint, and everything it mints says so.** ``mock`` is
also the DEFAULT when ``--runner`` is omitted, which is how a fabrication reached a signed
challenge against a published canonical. The marker is ``styxx.v8.cert.SYNTHETIC``, it is
stamped in ``fingerprint.run_fingerprint`` and in ``verify``, and the argument for marking
rather than refusing is in ``styxx/v8/cert.py`` under "Decisions". Nothing in this module
special-cases the mock: it asks the runner (``runner.is_synthetic``) and the answer travels in
the bytes.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import battery as batterymod
from . import cert as certmod
from . import fingerprint as fpmod
from . import floor as floormod
from . import keys
from . import log as logmod
from . import runner as runnermod
from . import sweep as sweepmod
from . import verify as verifymod
from .consts import EXIT, ID_RE, SCHEMA_VERSION
from .jcs import canonical_bytes
from .runner import MockRunner

__all__ = [
    "CliError",
    "USAGE",
    "main",
    "run",
    "cmd_key",
    "cmd_prereg",
    "cmd_fingerprint",
    "cmd_verify",
    "cmd_battery",
    "cmd_log",
]

DEFAULT_KEY_PATH = Path.home() / ".styxx" / "keys" / "v8.pem"
DEFAULT_RUNS = 5
# There is no DEFAULT_COVERS / DEFAULT_NOT_COVERED any more. Section 5.4's two lists are derived
# from the plan (`floor.plan_coverage`) and the log re-derives them; a constant standing behind
# that derivation is a coverage claim this file makes on the issuer's behalf (A-COVER).
ISSUER_NAME = "styxx v8"
# Section 5.1 step 2: "R = 5 minimum (10 pairwise distances)".
MIN_FLOOR_RUNS = 5
# Section 5.1 step 1: what a noise plan fixes before the runs.
PLAN_FIELDS = ("runs", "nuisance", "environment")

USAGE = (
    "python -m styxx.v8 <verb> ...  (one JSON object on stdout; exit codes are consts.EXIT)",
    "",
    "key generate [--out <pem>] [--force]",
    "key show [--key <pem>] [--pub <file>]",
    "",
    "prereg noise-plan --runs R --nuisance <factor=v1|v2>[,<factor=v>]... --subject <spec.json>",
    "                  --battery <cert.json> --key <pem> [--recipe <recipe.json>]",
    "                  [--runner mock|hf] [--snapshot <dir>] [--dtype <t>] [--device <d>]",
    "                  [--window <start>,<end>] [--out <file>] [--created <RFC3339 Z>]",
    "                        (the plan OBSERVES the environment it names through the runner, the",
    "                         same rule `fingerprint` applies (section 2.2); the runs are then",
    "                         refused unless they match it outside the declared factors.",
    "                         `covers` and `not_covered` are DERIVED from what the plan fixed",
    "                         (section 5.4) and the log re-derives them -- there is no",
    "                         --not-covered flag, because a list the anchor recomputes is not a",
    "                         list the command line may state.)",
    "",
    "fingerprint --subject <spec.json> --battery <cert.json> --key <pem>",
    "            [--recipe <recipe.json>] [--runner mock|hf] [--runs R] [--plan <cert file>]",
    "                        (--plan takes a bare cert id only at --runs 1: R runs need the",
    "                         plan's declared factors, which only the cert carries)",
    "            [--sensitivity <cert id>] [--covers a,b] [--not-covered a,b] [--redact]",
    "            [--out <dir>] [--nuisance ids] [--drift ids] [--no-logprobs]",
    "            [--snapshot <dir>] [--dtype <t>] [--device <d>] [--created <RFC3339 Z>]",
    "",
    "verify --ref <cert.json> [--runner mock|hf] [--no-confirm] [--challenge] [--own <id|file>]",
    "       [--note <text>] [--environment <json>] [--key <pem>] [--result-out <file>]",
    "                        (--environment ANNOTATES what the runner observed -- a harness block,",
    "                         an env_lock_sha256.  A field of it that contradicts an observed one",
    "                         is exit 5: the runtime and the hardware come from the runner.)",
    "verify --diff <a.json> <b.json>",
    "       (both: [--log <dir>] [--resolve <file|dir>]... [--battery <cert.json>])",
    "",
    "battery select --sweep <record.json> --pool <cert.json> --n N --k K --perm-seed S",
    "               [--tau T] [--max-family-share F] [--out <file>]",
    "               [--key <pem> --subject <spec.json> --selected-against <cert id>]",
    "battery fixed --source <items.json> [--label <str>] [--out <file>]",
    "battery pool  --source <items.json> [--out <file>]",
    "",
    "log init --log <dir> (--key <pem> | --pub <file>) [--issuer [name=]<key|file>]...",
    "log append <cert.json> --log <dir> [--blob <file>]...",
    "log prove <index> --log <dir> [--tree-size N] [--out <file>]",
    "log sth --log <dir> --key <pem> [--timestamp <RFC3339 Z>]",
    "log mirror --log <dir> --to <dir> [--pin <log.pub>] [--pinned-sth <file>]",
    "log verify-cert <cert.json>",
    "log verify-inclusion <proof.json> <sth.json> [--pin <log.pub>] [--log <dir>]",
    "log verify-consistency <sth1.json> <sth2.json> [--proof <file>] [--pin ...] [--log <dir>]",
    "log verify-sth <sth.json> [--pin <log.pub>] [--log <dir>]",
)


# --------------------------------------------------------------------------- errors


class CliError(Exception):
    """A refusal with the exit code it carries. ``extra`` is merged into the error payload."""

    def __init__(self, message: str, code: int = EXIT["invalid"], **extra: Any):
        super().__init__(message)
        self.message = str(message)
        self.code = int(code)
        self.extra = dict(extra)


def _unavailable(message: str, **extra: Any) -> CliError:
    return CliError(message, EXIT["unavailable"], **extra)


# --------------------------------------------------------------------------- argument parsing


def _parse_options(
    argv: Sequence[str],
    *,
    flags: Iterable[str] = (),
    values: Iterable[str] = (),
    multi: Iterable[str] = (),
    positional_max: Optional[int] = None,
) -> tuple[dict, list[str]]:
    """A small, total option parser: ``--name value``, ``--name=value``, flags, repeats.

    Option names are normalized ``-`` -> ``_``, so ``--perm-seed`` reads back as ``perm_seed``.
    An unknown option is a refusal rather than a positional, so a typo never runs silently.
    """
    flag_set = set(flags)
    value_set = set(values)
    multi_set = set(multi)
    opts: dict[str, Any] = {}
    opts.update({name: False for name in flag_set})
    opts.update({name: None for name in value_set})
    opts.update({name: [] for name in multi_set})
    positional: list[str] = []

    i = 0
    argv = list(argv)
    while i < len(argv):
        token = argv[i]
        if token == "--":
            positional.extend(argv[i + 1:])
            break
        if token.startswith("--"):
            name = token[2:]
            inline: Optional[str] = None
            if "=" in name:
                name, inline = name.split("=", 1)
            key = name.replace("-", "_")
            if key in flag_set:
                if inline is not None:
                    raise CliError(f"--{name} takes no value")
                opts[key] = True
            elif key in value_set or key in multi_set:
                if inline is None:
                    i += 1
                    if i >= len(argv):
                        raise CliError(f"--{name} needs a value")
                    inline = argv[i]
                if key in multi_set:
                    opts[key].append(inline)
                else:
                    if opts[key] is not None:
                        raise CliError(f"--{name} is given more than once")
                    opts[key] = inline
            else:
                raise CliError(f"unknown option --{name}")
        elif token.startswith("-") and token != "-":
            raise CliError(f"unknown option {token}")
        else:
            positional.append(token)
        i += 1

    if positional_max is not None and len(positional) > positional_max:
        raise CliError(f"unexpected argument {positional[positional_max]!r}")
    return opts, positional


def _require(opts: dict, name: str, what: str = "") -> str:
    value = opts.get(name)
    if value is None or value == "":
        raise CliError(f"--{name.replace('_', '-')} is required{(' (' + what + ')') if what else ''}")
    return str(value)


def _as_int(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    try:
        out = int(str(value), 10)
    except (TypeError, ValueError):
        raise CliError(f"--{name.replace('_', '-')} must be an integer, got {value!r}") from None
    if minimum is not None and out < minimum:
        raise CliError(f"--{name.replace('_', '-')} must be >= {minimum}, got {out}")
    return out


def _as_float(value: Any, name: str) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        raise CliError(f"--{name.replace('_', '-')} must be a number, got {value!r}") from None
    if out != out or out in (float("inf"), float("-inf")):
        raise CliError(f"--{name.replace('_', '-')} must be finite, got {value!r}")
    return out


def _csv(value: Any) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


# --------------------------------------------------------------------------- files


def _read_bytes(path: Any, what: str) -> bytes:
    p = Path(path)
    if not p.is_file():
        raise _unavailable(f"{what}: no such file: {p}")
    try:
        with open(os.fspath(p), "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise _unavailable(f"{what}: cannot read {p}: {exc}") from None


def _load_json(path: Any, what: str) -> Any:
    raw = _read_bytes(path, what)
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise CliError(f"{what}: {path} is not UTF-8: {exc}") from None
    except ValueError as exc:
        raise CliError(f"{what}: {path} is not JSON: {exc}") from None


def _load_object(path: Any, what: str) -> dict:
    obj = _load_json(path, what)
    if not isinstance(obj, dict):
        raise CliError(f"{what}: {path} is a {type(obj).__name__}, not a JSON object")
    return obj


def _write_json(path: Any, obj: Any) -> str:
    """UTF-8, LF, no BOM, one trailing newline. Returns the path as a string."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with open(os.fspath(p), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return str(p)


def _write_cert(directory: Any, name: str, cert: dict) -> str:
    """Write a cert as its own canonical bytes plus one LF, so the file is checkable by eye."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    payload = canonical_bytes(cert) + b"\n"
    with open(os.fspath(path), "wb") as fh:
        fh.write(payload)
    return str(path)


def _short(cert_id: Any) -> str:
    if isinstance(cert_id, str) and cert_id.startswith("sha256:"):
        return cert_id[7:19]
    return "unknown"


# --------------------------------------------------------------------------- keys and signing


def _load_seed(path: Any, what: str = "--key") -> bytes:
    _read_bytes(path, what)  # a missing key file is unavailable, not invalid
    try:
        return keys.load_private_pem(path)
    except ValueError as exc:
        raise CliError(f"{what}: {exc}") from None


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sign_cert(
    cert_type: str,
    *,
    subject: dict,
    recipe: dict,
    body: dict,
    refs: list[dict],
    seed: bytes,
    issuer_name: str = ISSUER_NAME,
    created: Optional[str] = None,
) -> dict:
    core = {
        "styxx": SCHEMA_VERSION,
        "type": cert_type,
        "created": created if created else _now(),
        "issuer": {"name": issuer_name, "key": keys.encode_public(keys.public_from_private(seed))},
        "subject": copy.deepcopy(subject),
        "recipe": copy.deepcopy(recipe),
        "body": copy.deepcopy(body),
        "refs": [dict(r) for r in refs],
    }
    try:
        signed = certmod.sign(core, seed)
    except (ValueError, TypeError) as exc:
        raise CliError(f"sign {cert_type}: {exc}") from None
    outcome = certmod.check(signed)
    if not outcome.ok:
        raise CliError(
            f"the {cert_type} cert this build produced does not check out",
            reasons=list(outcome.reasons),
        )
    return signed


# --------------------------------------------------------------------------- runners


def _make_runner(opts: dict) -> Any:
    which = opts.get("runner") or "mock"
    if which == "mock":
        return MockRunner(
            nuisance_items=set(_csv(opts.get("nuisance"))),
            drift_items=set(_csv(opts.get("drift"))),
            logprobs=not bool(opts.get("no_logprobs")),
        )
    if which == "hf":
        snapshot = opts.get("snapshot")
        if not snapshot:
            raise CliError("--runner hf needs --snapshot <dir> (a local snapshot, never a hub id)")
        try:
            from .runner_hf import TransformersRunner
        except Exception as exc:  # torch/transformers absent is unavailability, not invalidity
            raise _unavailable(f"runner hf: {type(exc).__name__}: {exc}") from None
        try:
            return TransformersRunner(
                str(snapshot),
                dtype=str(opts.get("dtype") or "bfloat16"),
                device=str(opts.get("device") or "cuda"),
            )
        except Exception as exc:
            raise _unavailable(f"runner hf: {type(exc).__name__}: {exc}") from None
    raise CliError(f"--runner must be 'mock' or 'hf', got {which!r}")


_RUNNER_FLAGS = ("no_logprobs",)
_RUNNER_VALUES = ("runner", "nuisance", "drift", "snapshot", "dtype", "device")


def _observe_environment(runner: Any, subject: dict, flag: str) -> Any:
    """Set ``subject['environment']`` to what ``runner`` OBSERVED, the spec's block annotating it.

    In place and returning it, because the thing this closes is an ABSENT key: a caller that
    assigns the return value cannot express "leave it off", which is the state ENV-ABSENT is
    about, and a test that restores the old behaviour has to be able to produce it.

    The mint side of the rule ``verify --ref`` applies (section 2.2, S-DEVICE): the environment a
    cert records is observed by the runner and what the ``--subject`` spec carries may only
    annotate it.  Without it the spec file was the whole of the claim -- a subject block naming an
    RTX 4070 minted a canonical fingerprint naming one, whatever the run touched, and every floor
    built on it inherited the sentence.  A synthetic runner observes nothing and its spec stands;
    its certs carry ``synthetic`` in the signed bytes and are refused as a baseline for anything
    measured (``cert.comparable``).

    **This runs whether or not the spec carries an ``environment`` key** (ENV-ABSENT). It used to
    sit behind ``if "environment" in subject:``, so an issuer reached the un-observed path by
    deleting one key: a weights subject without it is refused later by ``schema/subject.json``
    with a schema message rather than this one, but an ALIAS subject without it minted a
    fingerprint carrying no environment at all -- and a floor whose subject has no environment
    derives an empty ``not_covered``, which section 5.4 reads as covering every environment there
    is.  A guard that an omission turns off is not a guard, so there is no omission branch: with
    no block in the spec the annotation is ``None`` and the runner's report stands alone.
    """
    try:
        subject["environment"] = runnermod.observed_environment(
            runner, subject.get("environment")
        )
    except runnermod.SubjectUnavailable as exc:
        raise _unavailable(f"{flag} environment: {exc}") from None
    return subject["environment"]


def _runs_match_the_plans_environment(plan_body: dict, plan: Any, observed: dict) -> None:
    """Refuse runs that did not happen in the environment their plan fixed (PLAN-ENV, mint side).

    The plan names the environment before the runs (section 5.1 step 1) and the runs vary only the
    factors it declared (step 2), so every environment leaf the plan fixed must read the same in
    what the runner just observed.  ``Log._check_floor_environment_matches_the_plan`` is the same
    predicate over bytes on disk; this one is here so the refusal arrives before the runs cost
    anything and names the flag that would have to change.

    One-directional, like the log's: a run may carry leaves the plan never mentioned.
    """
    environment = plan_body.get("environment")
    if not isinstance(environment, dict) or not environment:
        return
    held = {
        block.get("factor")
        for block in (plan_body.get("nuisance") or [])
        if isinstance(block, dict)
    }
    for path in floormod.environment_paths(environment):
        if path in held or path.split(".")[0] in held:
            continue
        want = _at_path(environment, path)
        got = _at_path(observed, path)
        if got == want:
            continue
        shown = "nothing" if got is _ABSENT_PATH else repr(got)
        raise CliError(
            f"--plan {plan}: the plan fixed environment {path} at {want!r} and this runner "
            f"observes {shown}; the plan names the environment BEFORE the runs and the runs vary "
            "only the nuisance factors it declared (section 5.1 steps 1 and 2), so these runs "
            "would measure a floor of a different quantity than the plan describes"
        )


_ABSENT_PATH = object()


def _at_path(obj: Any, path: str) -> Any:
    """The value at a dotted path, or ``_ABSENT_PATH``; a parent that is not a dict is absent."""
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT_PATH
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------- resolvers


def _collect_certs(target: Any, mapping: dict) -> int:
    """Add every cert found at ``target`` (a file, a list of certs, or a directory) by id."""
    p = Path(target)
    found = 0
    if p.is_dir():
        for path in sorted(p.rglob("*.json")):
            if path.name.endswith(".meta.json"):
                continue
            try:
                obj = json.loads(path.read_bytes().decode("utf-8"))
            except Exception:
                continue
            found += _add_cert(obj, mapping)
        return found
    obj = _load_json(p, "--resolve")
    return _add_cert(obj, mapping)


def _add_cert(obj: Any, mapping: dict) -> int:
    if isinstance(obj, dict):
        cert_id = obj.get("id")
        if isinstance(cert_id, str):
            mapping[cert_id] = obj
            return 1
        return 0
    if isinstance(obj, list):
        return sum(_add_cert(entry, mapping) for entry in obj)
    return 0


def _open_log(directory: Any, *, create: bool = False) -> "logmod.Log":
    p = Path(directory)
    if not create and not p.is_dir():
        raise _unavailable(f"--log: no such directory: {p}")
    try:
        return logmod.Log(p)
    except OSError as exc:
        raise _unavailable(f"--log: {p}: {exc}") from None


def _build_resolver(opts: dict) -> Any:
    mapping: dict[str, dict] = {}
    for target in list(opts.get("resolve") or []):
        _collect_certs(target, mapping)
    if opts.get("battery"):
        _collect_certs(opts["battery"], mapping)
    log_obj = _open_log(opts["log"]) if opts.get("log") else None
    if not mapping and log_obj is None:
        return None

    def resolve(cert_id: str) -> Optional[dict]:
        hit = mapping.get(cert_id)
        if hit is not None:
            return hit
        if log_obj is not None:
            at = log_obj.find(cert_id)
            if at is not None:
                try:
                    return log_obj.cert(at)
                except Exception:
                    return None
        return None

    def baseline_gap(cert: dict) -> Optional[dict]:
        """Section 5.5's disclosure, from the log this resolver was built over.

        ``verify`` looks for this member (``verify._baseline_gap``) and prints what it returns:
        how far the fingerprint being verified sits from the baseline already on record. Without
        a ``--log`` there is no such member and no gap is printed -- which is a limit of the
        reading, not evidence that no baseline was replaced. The entry's metadata carries the
        same numbers (``Log.append`` wrote them); this recomputes rather than reads them, so the
        two can be compared and a reader is not taking the metadata's word for it.
        """
        if log_obj is None:
            return None
        try:
            return log_obj.baseline_gap(cert)
        except Exception:
            return None

    if log_obj is not None:
        resolve.baseline_gap = baseline_gap  # type: ignore[attr-defined]
    return resolve


# --------------------------------------------------------------------------- key


def cmd_key(argv: Sequence[str]) -> tuple[int, dict]:
    sub = argv[0] if argv else None
    if sub == "generate":
        opts, _ = _parse_options(argv[1:], values=("out",), flags=("force",), positional_max=0)
        out = Path(opts["out"]) if opts["out"] else DEFAULT_KEY_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        seed, public = keys.generate()
        try:
            keys.save_private_pem(seed, out, overwrite=bool(opts["force"]))
        except FileExistsError as exc:
            raise CliError(f"key generate: {exc}") from None
        except OSError as exc:
            raise _unavailable(f"key generate: cannot write {out}: {exc}") from None
        pub_path = Path(str(out) + ".pub")
        keys.save_public(public, pub_path)
        return 0, {
            "command": "key generate",
            "private_key_path": str(out),
            "public_key_path": str(pub_path),
            "public": keys.encode_public(public),
        }

    if sub == "show":
        opts, _ = _parse_options(argv[1:], values=("key", "pub"), positional_max=0)
        if opts["key"] and opts["pub"]:
            raise CliError("key show takes --key or --pub, not both")
        if opts["pub"]:
            _read_bytes(opts["pub"], "--pub")
            try:
                public = keys.load_public(opts["pub"])
            except ValueError as exc:
                raise CliError(f"--pub: {exc}") from None
            source = str(opts["pub"])
        else:
            path = opts["key"] or DEFAULT_KEY_PATH
            public = keys.public_from_private(_load_seed(path))
            source = str(path)
        return 0, {"command": "key show", "path": source, "public": keys.encode_public(public)}

    raise CliError(f"key: expected 'generate' or 'show', got {sub!r}")


# --------------------------------------------------------------------------- fingerprint


def _spec_subject_recipe(opts: dict) -> tuple[dict, dict]:
    """``--subject`` is a subject object or a ``{subject, recipe}`` spec; ``--recipe`` overrides."""
    spec = _load_object(_require(opts, "subject"), "--subject")
    if "kind" in spec and "subject" not in spec:
        subject, recipe = spec, None
    else:
        subject = spec.get("subject")
        recipe = spec.get("recipe")
        if not isinstance(subject, dict):
            raise CliError("--subject: the spec carries no 'subject' object")
    if opts.get("recipe"):
        recipe = _load_object(opts["recipe"], "--recipe")
    if not isinstance(recipe, dict):
        raise CliError("no recipe: pass --recipe <file> or a spec file carrying a 'recipe' object")
    return copy.deepcopy(subject), copy.deepcopy(recipe)


def _plan_reference(value: Any) -> tuple[Optional[str], Optional[dict]]:
    """``--plan``: a cert id (nothing to read) or a path to the plan cert (read and checked).

    Returns ``(plan_id, plan_body_or_None)``.  A bare id is all a signature needs -- the log
    resolves it at append -- but it tells this process nothing about what the plan fixed, so the
    section 5.1 checks below run only on the file form.  A stranger reproducing a floor has the
    cert, so the file form is the one to use.
    """
    if value is None:
        return None, None
    text = str(value)
    if _is_cert_id(text):
        return text, None
    if not Path(text).exists():
        raise CliError(
            f"--plan must be a cert id matching {ID_RE} or a path to the plan cert, got {text!r}"
        )
    cert = _load_object(text, "--plan")
    if cert.get("type") != "prereg":
        raise CliError(f"--plan: {text} is a {cert.get('type')!r} cert, not a prereg")
    body = cert.get("body")
    if not isinstance(body, dict) or body.get("kind") != "noise-plan":
        kind = body.get("kind") if isinstance(body, dict) else None
        raise CliError(f"--plan: the prereg's kind is {kind!r}, not 'noise-plan' (section 5.1)")
    # Step 1's three fields come before ``check``, so that a plan fixing nothing gets the sentence
    # that says which of them it is missing rather than the schema's report of the same absence.
    # ``runs`` is now BOTH: required by schema/prereg.json's noise-plan branch (A-NORUNS) and named
    # here, because a caller who passed a plan that fixes no R is owed the section 5.1 reason.
    missing = [name for name in PLAN_FIELDS if name not in body]
    if missing:
        raise CliError(
            f"--plan: the plan does not name {', '.join(missing)}; section 5.1 step 1 has it fix "
            "the run count, the enumerated nuisance factors with their values, and the environment"
        )
    outcome = certmod.check(cert)
    if not outcome.ok:
        raise CliError(f"--plan: {text} does not check out", reasons=list(outcome.reasons))
    return str(cert["id"]), body


def _run_settings(
    plan_body: Optional[dict],
    plan: Optional[str],
    runs: int,
    recipe: dict,
    item_ids: Sequence[str],
) -> list[dict]:
    """One ``{assignment, recipe, order, nuisance}`` per run -- the plan, applied (section 5.1).

    A single run under no plan is the recipe as written in A.3 order.  Anything else takes its
    settings from the plan cert, so ``--plan <id>`` (which carries no factors this process can
    read) cannot drive a multi-run floor: the alternative is R runs at the default under a plan
    that declared factors, which is the defect of `papers/v8/vacuous_floor_2026_09_09/`.
    """
    if runs > 1 and plan_body is None:
        raise CliError(
            f"--plan {plan}: --runs {runs} needs the plan CERT (a path), not a bare id: the R "
            "runs take their nuisance assignments from the plan's declared factors, and an id "
            "names a plan this process cannot read. Pass the file."
        )
    if plan_body is None:
        return [{"assignment": {}, "recipe": recipe, "order": None, "nuisance": {}}]
    try:
        return fpmod.plan_run_settings(plan_body, runs, recipe, item_ids)
    except (ValueError, TypeError) as exc:
        raise CliError(f"--plan: {exc}") from None


def cmd_fingerprint(argv: Sequence[str]) -> tuple[int, dict]:
    opts, _ = _parse_options(
        argv,
        flags=("redact",) + _RUNNER_FLAGS,
        values=(
            "subject", "recipe", "battery", "key", "out", "runs", "plan", "sensitivity",
            "covers", "not_covered", "created", "issuer_name",
        ) + _RUNNER_VALUES,
        positional_max=0,
    )
    subject, recipe = _spec_subject_recipe(opts)
    battery_cert = _load_object(_require(opts, "battery"), "--battery")
    seed = _load_seed(_require(opts, "key", "certs are signed"))
    runs = _as_int(opts["runs"], "runs", minimum=1) if opts["runs"] else DEFAULT_RUNS
    plan, plan_body = _plan_reference(opts.get("plan"))
    if runs > 1 and plan is None:
        raise CliError(
            f"--runs {runs} without --plan is not a floor: section 5.1 step 1 logs the nuisance "
            "plan BEFORE the runs, and a floor whose runs are not all under a logged plan is not "
            "a floor. Mint one with `prereg noise-plan` and pass it as --plan."
        )
    if plan_body is not None:
        planned = plan_body.get("runs")
        if planned != runs:
            raise CliError(
                f"--runs {runs} is not the R the plan fixed ({planned!r}): the plan is on record "
                "before the runs so that R is not chosen after seeing them (section 5.1 step 1)"
            )
    sensitivity = opts.get("sensitivity")
    if sensitivity is not None and not _is_cert_id(sensitivity):
        raise CliError(f"--sensitivity must be a cert id matching {ID_RE}, got {sensitivity!r}")

    battery_id = battery_cert.get("id")
    if not _is_cert_id(battery_id):
        raise CliError("--battery: the battery cert carries no 'id'")
    declared = recipe.get("battery")
    if declared is None:
        recipe["battery"] = battery_id
    elif declared != battery_id:
        raise CliError(f"recipe.battery {declared} is not the battery cert's id {battery_id}")

    runner = _make_runner(opts)
    _observe_environment(runner, subject, "--subject")
    if plan_body is not None:
        _runs_match_the_plans_environment(plan_body, plan, subject["environment"])
    try:
        item_ids = sorted(fpmod.battery_item_map(battery_cert))
    except (ValueError, TypeError) as exc:
        raise CliError(f"--battery: {exc}") from None

    settings = _run_settings(plan_body, plan, runs, recipe, item_ids)

    bodies: list[dict] = []
    for k, setting in enumerate(settings):
        try:
            bodies.append(
                fpmod.run_fingerprint(
                    runner,
                    subject,
                    setting["recipe"],
                    battery_cert,
                    run_index=k,
                    nuisance=dict(setting["nuisance"]),
                    order=setting["order"],
                    redacted=bool(opts["redact"]),
                )
            )
        except (ValueError, TypeError) as exc:
            raise CliError(f"run {k}: {exc}") from None
        except Exception as exc:
            raise _unavailable(f"run {k}: {type(exc).__name__}: {exc}") from None

    issuer_name = opts.get("issuer_name") or ISSUER_NAME
    created = opts.get("created")
    base_refs = [{"role": "battery", "id": battery_id}]
    if plan:
        # Section 5.5: the runs of one plan share a `noise_plan` ref, which is what lets a log
        # accept them in sequence without a `previous` ref on each.
        base_refs.append({"role": "noise_plan", "id": plan})

    # Section 5.1 step 5: the R-1 NON-canonical runs are logged, and then the canonical
    # fingerprint (`run_index` 0) carrying the floor. Run 0 gets no second cert of its own --
    # that is what used to make the canonical byte-identical to run 0 and share its id.
    # Each run cert carries the recipe THAT run ran under: the plan's `batch_size` and
    # `padding_side` are written into `decoding` per run (section 2.3 nuisance), so a cert never
    # records an execution its run did not use.
    run_certs = [
        _sign_cert(
            "fingerprint",
            subject=subject,
            recipe=settings[k]["recipe"],
            body=body,
            refs=base_refs,
            seed=seed,
            issuer_name=issuer_name,
            created=created,
        )
        for k, body in enumerate(bodies[1:], start=1)
    ]
    run_ids = [c["id"] for c in run_certs]

    canonical_body = bodies[0]
    floor_block = None
    if plan and runs >= 2:
        # A plan read from its cert fixes the coverage, and it did so before the runs: letting
        # the command line restate it afterwards is the hole section 5.1 step 1 exists to close.
        if plan_body is not None and (opts["covers"] or opts["not_covered"]):
            raise CliError(
                "--covers / --not-covered cannot be given beside a plan cert: the plan fixed the "
                "coverage before the runs (sections 5.1 step 1, 5.4)"
            )
        # A-COVER: both lists are DERIVED from the plan's own `nuisance` and `environment` by
        # `floor.plan_coverage`, which is the function `Log._check_floor_covers_match_the_plan`
        # re-derives them with. They used to be read off the plan's own `covers`/`not_covered`
        # fields with a CLI default behind them, so a plan carrying neither minted a floor whose
        # coverage was two constants in this file -- and nothing downstream compared either list
        # against the plan at all. `--runs R > 1` requires the plan CERT (`_run_settings`), so
        # `plan_body` is never None on this branch.
        covers, not_covered = floormod.plan_coverage(plan_body)
        try:
            canonical_body = fpmod.attach_floor(
                bodies[0], bodies, plan, run_ids, covers, not_covered
            )
        except (ValueError, TypeError) as exc:
            raise CliError(f"attach_floor: {exc}") from None
        floor_block = canonical_body["noise_floor"]
    if sensitivity:
        canonical_body = dict(canonical_body)
        canonical_body["sensitivity"] = sensitivity

    refs = list(base_refs)
    if floor_block is not None:
        refs.extend({"role": "run", "id": rid} for rid in run_ids)
    if sensitivity:
        refs.append({"role": "sensitivity", "id": sensitivity})
    canonical = _sign_cert(
        "fingerprint",
        subject=subject,
        recipe=settings[0]["recipe"],
        body=canonical_body,
        refs=refs,
        seed=seed,
        issuer_name=issuer_name,
        created=created,
    )

    if canonical["id"] in run_ids:  # unreachable by construction; a duplicate id is not a floor
        raise CliError("the canonical fingerprint has the same id as one of the run certs")

    written: list[str] = []
    if opts.get("out"):
        for k, cert in enumerate(run_certs, start=1):
            written.append(_write_cert(opts["out"], f"fingerprint-run{k}-{_short(cert['id'])}", cert))
        written.append(
            _write_cert(opts["out"], f"fingerprint-canonical-{_short(canonical['id'])}", canonical)
        )

    channels = canonical_body.get("channels", {})
    payload = {
        "command": "fingerprint",
        "id": canonical["id"],
        "battery": battery_id,
        "runs": runs,
        "run_ids": run_ids,
        "append_order": [*run_ids, canonical["id"]],
        # What each run actually varied. Printed because a floor is only as good as the spread
        # its runs took, and a reader should not have to open R certs to see it.
        "assignments": [dict(s["assignment"]) for s in settings],
        "tier": canonical_body.get("tier"),
        "redacted": bool(canonical_body.get("redacted")),
        "channels": {
            name: (True if "hash" in block else bool(block.get("present")))
            for name, block in channels.items()
        },
        "floor": (
            {c: b["floor"] for c, b in sorted(floor_block["per_channel"].items())}
            if floor_block
            else None
        ),
        "alpha_overall": floor_block["alpha_overall"] if floor_block else None,
        "covers": floor_block["covers"] if floor_block else None,
        "not_covered": floor_block["not_covered"] if floor_block else None,
        "noise_plan": plan,
        "sensitivity": sensitivity,
        "written": written,
        "cert": canonical,
    }
    return 0, payload


def _is_cert_id(value: Any) -> bool:
    import re

    return isinstance(value, str) and re.match(ID_RE, value) is not None


# --------------------------------------------------------------------------- prereg


def _nuisance_factors(values: Sequence[str]) -> list[dict]:
    """``factor=v1|v2,factor2=v3`` (repeatable) -> the section 5.1 enumerated factor list.

    Values are text: what the plan fixes is the SET the runs may vary over, and comparing the
    plan against a run's recorded nuisance is a comparison of the names the two wrote down.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for chunk in values:
        for spec in _csv(chunk):
            if "=" not in spec:
                raise CliError(
                    f"--nuisance: {spec!r} is not 'factor=value[|value...]'; section 5.1 wants "
                    "each nuisance factor with the values the runs may take"
                )
            factor, raw = spec.split("=", 1)
            factor = factor.strip()
            if not factor:
                raise CliError(f"--nuisance: {spec!r} names no factor")
            if factor in seen:
                raise CliError(f"--nuisance: factor {factor!r} is given more than once")
            seen.add(factor)
            vals = [part.strip() for part in raw.split("|") if part.strip()]
            if not vals:
                raise CliError(f"--nuisance: factor {factor!r} has no values")
            if len(set(vals)) != len(vals):
                raise CliError(f"--nuisance: factor {factor!r} repeats a value")
            out.append({"factor": factor, "values": vals})
    if not out:
        raise CliError(
            "--nuisance is required: a plan that enumerates no factor fixes nothing, and "
            "section 5.1 step 1 is what stops the nuisance set being chosen after the runs"
        )
    return sorted(out, key=lambda block: block["factor"])


def cmd_prereg(argv: Sequence[str]) -> tuple[int, dict]:
    """``prereg noise-plan``: the cert section 5.1 step 1 puts on record before the floor runs.

    Nothing else in this surface can mint one, and without one no floor can be constructed: the
    runs have no ``noise_plan`` ref to share, so section 5.5's baseline rule refuses the second
    comparable fingerprint and section 5.1's own sentence ("a floor whose runs are not all under
    a logged plan is not a floor") is unsatisfiable.

    **PLAN-ENV: the plan OBSERVES the environment it names, and the runs must then match it.**
    This command built no runner at all and copied ``environment`` straight out of the ``--subject``
    spec, so the block the whole of section 5.4's coverage vocabulary resolves against was a value
    the issuer typed -- while section 2.2's mint rule had already been applied to the fingerprint.
    A plan is written before its runs, so it cannot observe THEM; what it can do, and now does, is
    observe the box it is minted on and name that as the environment it intends to fix. Two
    consequences, and the second is what makes the first bind:

    * the same ``--runner``/``--device``/``--snapshot`` flags ``fingerprint`` takes are taken here,
      and ``runner.observed_environment`` reports through them, so a plan minted on a CPU cannot
      name a card. The spec's block may still ANNOTATE (a ``harness``, an ``env_lock_sha256``) and
      a leaf of it contradicting an observed one is exit 5, exactly as at the fingerprint;
    * ``fingerprint --plan`` refuses runs whose observed environment differs from the plan's on any
      leaf the plan did not declare as a nuisance factor, and ``Log`` refuses the same floor from
      the other side (``_check_floor_environment_matches_the_plan``). Without that half a plan's
      environment is a sentence about a different process.

    A plan minted on one box for runs on another is therefore refused today. That is a real design
    choice and not an oversight, and it is marked ``[OPERATOR-GATED S5-10]`` in the spec: a docket
    or a multi-party replication may want a coordinator to write the plan for environments it does
    not hold, and the price of allowing it is that the plan's environment becomes an assertion
    again, with the runs bound to it only by the two rules above.
    """
    sub = argv[0] if argv else None
    if sub != "noise-plan":
        raise CliError(f"prereg: expected 'noise-plan', got {sub!r}")

    opts, _ = _parse_options(
        argv[1:],
        flags=_RUNNER_FLAGS,
        values=(
            "runs", "subject", "recipe", "battery", "key", "out", "window",
            "created", "issuer_name",
        ) + _RUNNER_VALUES,
        multi=("nuisance",),
        positional_max=0,
    )
    runs = _as_int(_require(opts, "runs"), "runs", minimum=1)
    if runs < MIN_FLOOR_RUNS:
        raise CliError(
            f"--runs {runs} is below the minimum: section 5.1 step 2 sets R = {MIN_FLOOR_RUNS} "
            f"minimum ({MIN_FLOOR_RUNS * (MIN_FLOOR_RUNS - 1) // 2} pairwise distances)"
        )
    factors = _nuisance_factors(opts["nuisance"])
    subject, recipe = _spec_subject_recipe(opts)
    seed = _load_seed(_require(opts, "key", "certs are signed"))

    battery_cert = _load_object(_require(opts, "battery"), "--battery")
    battery_id = battery_cert.get("id")
    if not _is_cert_id(battery_id):
        raise CliError("--battery: the battery cert carries no 'id'")
    declared = recipe.get("battery")
    if declared is None:
        recipe["battery"] = battery_id
    elif declared != battery_id:
        raise CliError(f"recipe.battery {declared} is not the battery cert's id {battery_id}")

    # PLAN-ENV. The plan's environment is what the runner reports, annotated by the spec's block;
    # see this command's docstring for why a plan observes at all when it runs nothing itself.
    _observe_environment(_make_runner(opts), subject, "--subject")
    environment = subject.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise CliError(
            "--subject: the plan names the environment (section 5.1 step 1) and this subject "
            "carries no 'environment' block to name, and the runner reported none either"
        )

    body: dict[str, Any] = {
        "kind": "noise-plan",
        "runs": runs,
        "nuisance": factors,
        "environment": copy.deepcopy(environment),
    }
    # Section 5.4's two lists are DERIVED from the two fields above, by the same function the log
    # re-derives them with (A-COVER). There is no `--not-covered` here any more: a list the anchor
    # recomputes is not a list the command line may state, and while it could be stated it was the
    # one place an issuer chose, before the runs but by hand, which environment fields would read
    # as held fixed -- and an empty `not_covered` is section 5.4's claim to cover everything.
    covers, not_covered = floormod.plan_coverage(body)
    body["covers"] = covers
    body["not_covered"] = not_covered
    window = _csv(opts["window"])
    if subject.get("kind") == "alias" and len(window) != 2:
        raise CliError(
            "--window <start>,<end> is required for an alias subject: an alias floor carries a "
            "mandatory window (sections 2.2, 5.1 step 2)"
        )
    if window:
        if len(window) != 2:
            raise CliError(f"--window takes exactly 'start,end', got {opts['window']!r}")
        body["window"] = {"start": window[0], "end": window[1]}

    cert = _sign_cert(
        "prereg",
        subject=subject,
        recipe=recipe,
        body=body,
        refs=[{"role": "battery", "id": battery_id}],
        seed=seed,
        issuer_name=opts.get("issuer_name") or ISSUER_NAME,
        created=opts.get("created"),
    )
    payload = {
        "command": "prereg noise-plan",
        "id": cert["id"],
        "kind": "noise-plan",
        "runs": runs,
        "battery": battery_id,
        "covers": covers,
        "not_covered": body["not_covered"],
        "nuisance": factors,
        "written": None,
        "cert": cert,
    }
    if opts["out"]:
        payload["written"] = _write_json(opts["out"], cert)
    return 0, payload


# --------------------------------------------------------------------------- verify


def cmd_verify(argv: Sequence[str]) -> tuple[int, dict]:
    opts, positional = _parse_options(
        argv,
        flags=("challenge", "no_confirm") + _RUNNER_FLAGS,
        values=(
            "ref", "log", "key", "own", "note", "environment", "result_out", "challenge_out",
            "battery", "issuer_name", "created",
        ) + _RUNNER_VALUES,
        multi=("diff", "resolve"),
    )
    diff_paths = list(opts["diff"]) + list(positional)
    if opts["ref"] and diff_paths:
        raise CliError("verify takes --ref or --diff, not both")
    resolver = _build_resolver(opts)

    if diff_paths:
        if len(diff_paths) != 2:
            raise CliError(f"verify --diff needs exactly two cert files, got {len(diff_paths)}")
        cert_a = _load_object(diff_paths[0], "--diff <a>")
        cert_b = _load_object(diff_paths[1], "--diff <b>")
        outcome = verifymod.diff(cert_a, cert_b, resolver)
        target_cert = cert_a
        mode = "diff"
    elif opts["ref"]:
        cert = _load_object(opts["ref"], "--ref")
        target_cert = cert
        runner = _make_runner(opts)
        environment = _load_object(opts["environment"], "--environment") if opts["environment"] else None
        outcome = verifymod.ref(
            cert,
            runner,
            resolver,
            confirm=not bool(opts["no_confirm"]),
            environment=environment,
        )
        mode = "ref"
    else:
        raise CliError("verify needs --ref <cert> or --diff <a> <b>")

    payload: dict[str, Any] = {
        "command": "verify",
        "mode": mode,
        "verdict": outcome.verdict,
        "exit_code": outcome.exit_code,
        "mismatched": list(outcome.mismatched),
        "result": outcome.result_body,
        "report": outcome.printed,
    }

    if opts["challenge"]:
        payload.update(_challenge(opts, outcome, mode, target_cert, resolver))
    if opts["result_out"]:
        payload.update(_result_cert(opts, outcome))
    return outcome.exit_code, payload


def _challenge(
    opts: dict,
    outcome: "verifymod.VerifyOutcome",
    mode: str,
    target_cert: Optional[dict] = None,
    resolver: Any = None,
) -> dict:
    """The section 9 challenge body, and -- with ``--key`` and ``--own`` -- the signed cert.

    Section 9 rule 1 is checked HERE, at mint, before anything is signed: a challenge is valid
    only if the challenger's own fingerprint is comparable to the target AND carries the target's
    subject identity, and a tool does not sign a cert it can already show is not a challenge.
    The rule itself is ``cert.challenge_validity``; the reader computes the same function from
    the same two certs (section 9 gives the computation to clients, and ``styxx.v8.log`` runs it
    again at append).

    **Nothing is signed while rule 1 is uncomputed.**  When ``--own`` is a bare id that no
    ``--log`` or ``--resolve`` resolves, this command cannot run the rule -- and it used to sign
    anyway, recording ``challenge_validity: null`` in stdout and NOTHING in the signed bytes, so
    the cert that reached a reader was indistinguishable from one whose rule 1 had passed
    (``papers/v8/challenge_and_attack_2026_09_09``, C-MINT-UNCOMPUTED).  Two repairs were
    available: refuse, or write the uncomputed state into the signed body.  Refusing governs.
    The second one signs an accusation whose validity is unknown TO THE SIGNER, and a challenge
    is an accusation; a tool that cannot resolve ``--own`` also cannot know whether what it is
    about to sign is a challenge at all, and section 9's rule is computable from two certs the
    challenger holds by definition -- the challenger minted one of them.  So an unresolvable
    ``--own`` is a missing input, and the fix is to pass the cert (as a file, or through
    ``--log`` / ``--resolve``), not to publish the gap.  ``challenge_validity`` is therefore
    never ``null`` in a payload that carries a ``challenge_cert``.
    """
    # The challenge carries the environment the VERIFY OBSERVED, which already has whatever
    # `--environment` was allowed to annotate onto it (`runner.observed_environment`).  Re-loading
    # the file here overwrote the observation a second time, in the body that travels furthest:
    # a challenge is read by strangers who hold neither the runner nor the box.  S-DEVICE signed
    # its RTX 4070 through this line as well as through the result body.
    environment = outcome.result_body.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    try:
        body = verifymod.challenge_body(outcome, environment, note=opts["note"])
    except (ValueError, TypeError) as exc:
        # The refusal carries the VERIFY's own exit code when the verify had one to give: a
        # `--ref` that came back `unavailable` and was also asked for a challenge is still
        # unavailable (section 6 exit 5), and calling it an invalid cert names the wrong thing.
        # Only the three refusal codes are borrowed -- a `--diff` exits 0/1/2 and cannot lend a
        # number to an error.
        code = outcome.exit_code
        if code not in (EXIT["mismatch"], EXIT["invalid"], EXIT["unavailable"]):
            code = EXIT["invalid"]
        # The verify's own report rides along: "a challenge needs a coverage word" says what this
        # command refused, and the operator still has to be told WHY the verify came back with
        # nothing -- which, for the S-DEVICE refusal, is the sentence naming the contradicted
        # fields.  Without it the diagnostic stops one level above the cause.
        raise CliError(
            f"--challenge: {exc}", code, verify_verdict=outcome.verdict, report=outcome.printed
        ) from None
    out: dict[str, Any] = {"challenge": body}

    target = outcome.result_body.get("ref")
    own = opts["own"]
    own_cert: Optional[dict] = None
    if own and not _is_cert_id(own):
        own_cert = _load_object(own, "--own")
        own = own_cert.get("id")
    elif own:
        resolved = resolver(own) if callable(resolver) else None
        own_cert = resolved if isinstance(resolved, dict) else None
    if own:
        if isinstance(target_cert, dict) and isinstance(own_cert, dict):
            reasons = certmod.challenge_validity(target_cert, own_cert)
            if reasons:
                raise CliError(
                    "--challenge: the fingerprint given as --own differs from the target on "
                    f"{reasons}; section 9 rule 1 makes a challenge valid only when the two are "
                    "comparable AND every S_identity field is equal, and a floor measured on one "
                    "subject says nothing about another (section 5.3). No match, no challenge",
                    challenge_validity=reasons,
                )
            out["challenge_validity"] = []
        else:
            missing = "the target cert" if not isinstance(target_cert, dict) else "--own"
            raise CliError(
                f"--challenge: section 9 rule 1 cannot be computed here because {missing} did "
                "not resolve to a cert, and this command does not sign a challenge whose "
                "validity it could not check. Pass --own as a file, or add --log/--resolve so "
                "both certs resolve",
                challenge_validity=None,
            )
    if opts["key"] and own and _is_cert_id(target):
        # Section 2.7 forbids `subject` and `recipe` on a challenge cert, so both stay `{}` --
        # `schema/envelope.json` requires the two members on every cert, and reconciling that
        # with 2.7's "a forbidden member is absent" moves every committed conformance vector, so
        # it is NOT landed here. The challenger's observed subject and recipe core go into the
        # BODY instead, which is where C3's repair belongs anyway (`verify.challenge_body`).
        cert = _sign_cert(
            "challenge",
            subject={},
            recipe={},
            body=body,
            refs=[{"role": "target", "id": target}, {"role": "own", "id": own}],
            seed=_load_seed(opts["key"]),
            issuer_name=opts.get("issuer_name") or ISSUER_NAME,
            created=opts.get("created"),
        )
        out["challenge_cert"] = cert
        out["challenge_cert_id"] = cert["id"]
        if opts["challenge_out"]:
            out["challenge_written"] = _write_json(opts["challenge_out"], cert)
    elif opts["challenge_out"] or opts["own"]:
        raise CliError("a signed challenge cert needs both --key <pem> and --own <cert id|file>")
    return out


def _result_cert(opts: dict, outcome: "verifymod.VerifyOutcome") -> dict:
    if not opts["key"]:
        raise CliError("--result-out needs --key <pem> to sign the result cert")
    seed = _load_seed(opts["key"])
    try:
        cert = verifymod.make_result_cert(
            outcome,
            keys.encode_public(keys.public_from_private(seed)),
            seed,
            outcome.result_body.get("refs_suggested") or [],
            {},
            {},
            created=opts.get("created"),
        )
    except (ValueError, TypeError) as exc:
        raise CliError(f"--result-out: {exc}") from None
    return {"result_cert_id": cert["id"], "result_written": _write_json(opts["result_out"], cert)}


# --------------------------------------------------------------------------- battery


def _items_from(path: Any, what: str) -> list[dict]:
    obj = _load_json(path, what)
    if isinstance(obj, dict):
        obj = obj.get("body", obj)
        if isinstance(obj, dict):
            obj = obj.get("items")
    if not isinstance(obj, list):
        raise CliError(f"{what}: expected a list of items, or an object carrying 'items'")
    return obj


def cmd_battery(argv: Sequence[str]) -> tuple[int, dict]:
    sub = argv[0] if argv else None

    if sub == "select":
        opts, _ = _parse_options(
            argv[1:],
            values=(
                "sweep", "pool", "n", "k", "perm_seed", "tau", "max_family_share", "out",
                "key", "subject", "recipe", "selected_against", "created", "issuer_name",
            ),
            positional_max=0,
        )
        sweep_path = _require(opts, "sweep")
        _read_bytes(sweep_path, "--sweep")
        try:
            record = sweepmod.read_record(sweep_path)
        except (ValueError, TypeError) as exc:
            raise CliError(f"--sweep: {exc}") from None
        pool_items = _items_from(_require(opts, "pool"), "--pool")
        n = _as_int(_require(opts, "n"), "n", minimum=1)
        k = _as_int(_require(opts, "k"), "k", minimum=0)
        perm_seed = _as_int(_require(opts, "perm_seed"), "perm_seed")
        tau = _as_float(opts["tau"], "tau") if opts["tau"] else 1.0
        share = (
            _as_float(opts["max_family_share"], "max_family_share")
            if opts["max_family_share"]
            else 0.25
        )
        try:
            scores = batterymod.score(record, tau)
            body = batterymod.select(
                pool_items, scores, n=n, k=k, max_family_share=share, tau=tau, perm_seed=perm_seed
            )
        except (ValueError, TypeError) as exc:
            raise CliError(f"battery select: {exc}") from None
        reasons = batterymod.validate_body(body)
        if reasons:
            raise CliError("battery select: the selected body does not validate", reasons=reasons)
        payload = {
            "command": "battery select",
            "kind": body["kind"],
            "n": n,
            "k": k,
            "selected": len(body["items"]),
            "excluded": len(body.get("excluded") or []),
            "k_anchors_actual": body["params"].get("k_anchors_actual"),
            "sensitivity_after_exclusion": body["params"].get("sensitivity_after_exclusion"),
            "validate": reasons,
            "body": body,
        }
        _maybe_sign_battery(opts, body, payload)
        if opts["out"]:
            payload["written"] = _write_json(opts["out"], payload.get("cert") or body)
        return 0, payload

    if sub in ("fixed", "pool"):
        opts, _ = _parse_options(
            argv[1:],
            values=("source", "label", "out", "key", "subject", "recipe", "created", "issuer_name"),
            positional_max=0,
        )
        source = _require(opts, "source")
        items = _items_from(source, "--source")
        try:
            if sub == "fixed":
                body = batterymod.fixed_v1(items, source=str(opts["label"] or Path(source).name))
            else:
                body = batterymod.pool_v1(items)
        except (ValueError, TypeError) as exc:
            raise CliError(f"battery {sub}: {exc}") from None
        reasons = batterymod.validate_body(body)
        if reasons:
            raise CliError(f"battery {sub}: the body does not validate", reasons=reasons)
        payload = {
            "command": f"battery {sub}",
            "kind": body["kind"],
            "items": len(body["items"]),
            "families": body.get("families"),
            "validate": reasons,
            "body": body,
        }
        _maybe_sign_battery(opts, body, payload)
        if opts["out"]:
            payload["written"] = _write_json(opts["out"], payload.get("cert") or body)
        return 0, payload

    raise CliError(f"battery: expected 'select', 'fixed' or 'pool', got {sub!r}")


def _maybe_sign_battery(opts: dict, body: dict, payload: dict) -> None:
    """Sign the body into a battery cert when a key and a subject spec were given."""
    if not opts.get("key"):
        if opts.get("selected_against"):
            raise CliError("--selected-against only means anything with --key <pem>")
        return
    if not opts.get("subject"):
        raise CliError("signing a battery cert needs --subject <spec.json> (subject and recipe)")
    subject, recipe = _spec_subject_recipe(opts)
    refs: list[dict] = []
    if _is_cert_id(recipe.get("battery")):
        refs.append({"role": "battery", "id": recipe["battery"]})
    if body["kind"] == "canary-v1":
        selected = opts.get("selected_against")
        if not _is_cert_id(selected):
            raise CliError(
                "a canary-v1 battery cert needs --selected-against <fingerprint cert id> "
                "(schema/battery.json requires the ref)"
            )
        refs.append({"role": "selected_against", "id": selected})
    cert = _sign_cert(
        "battery",
        subject=subject,
        recipe=recipe,
        body=body,
        refs=refs,
        seed=_load_seed(opts["key"]),
        issuer_name=opts.get("issuer_name") or ISSUER_NAME,
        created=opts.get("created"),
    )
    payload["cert"] = cert
    payload["id"] = cert["id"]


# --------------------------------------------------------------------------- log


_LOG_VALUES = ("log", "key", "pub", "pin", "out", "to", "pinned_sth", "proof", "timestamp", "tree_size")
_LOG_MULTI = ("issuer", "blob")
# ``log init --open-issuers`` writes the marker that says this log admits any key. Without it, and
# without ``--issuer``, the log gets an empty roster and admits nobody -- which is a policy. What
# is no longer reachable is a log with no policy file at all, because that used to admit
# everything (L10c, ``log.Log.issuer_policy``).
_LOG_FLAGS = ("open_issuers",)


def _log_public(opts: dict) -> bytes:
    if opts.get("pin"):
        _read_bytes(opts["pin"], "--pin")
        try:
            return keys.load_public(opts["pin"])
        except ValueError as exc:
            raise CliError(f"--pin: {exc}") from None
    if opts.get("log"):
        public = _open_log(opts["log"]).log_public()
        if public is not None:
            return public
    raise _unavailable("no log public key: pass --pin <log.pub> or --log <dir> holding keys/log.pub")


def _issuer_entry(spec: str) -> dict:
    name = ""
    rest = spec
    if "=" in spec and not spec.startswith("ed25519:"):
        name, rest = spec.split("=", 1)
    if rest.startswith("ed25519:"):
        key = rest
    else:
        _read_bytes(rest, "--issuer")
        try:
            key = keys.encode_public(keys.load_public(rest))
        except ValueError as exc:
            raise CliError(f"--issuer {spec!r}: {exc}") from None
    return {"name": name, "key": key, "from_index": 0, "retired_at_index": None}


def cmd_log(argv: Sequence[str]) -> tuple[int, dict]:
    sub = argv[0] if argv else None
    if sub is None:
        raise CliError("log: expected a subcommand")
    opts, positional = _parse_options(
        argv[1:], flags=_LOG_FLAGS, values=_LOG_VALUES, multi=_LOG_MULTI, positional_max=2
    )

    if sub == "init":
        directory = _require(opts, "log")
        if opts["key"]:
            public = keys.public_from_private(_load_seed(opts["key"]))
        elif opts["pub"]:
            _read_bytes(opts["pub"], "--pub")
            try:
                public = keys.load_public(opts["pub"])
            except ValueError as exc:
                raise CliError(f"--pub: {exc}") from None
        else:
            raise CliError("log init needs --key <pem> or --pub <log.pub> for the log's own key")
        issuers = [_issuer_entry(spec) for spec in opts["issuer"]]
        if opts.get("open_issuers"):
            if issuers:
                raise CliError(
                    "log init: --open-issuers and --issuer are two different admission policies; "
                    "an open log admits any key and a roster admits the keys it lists"
                )
            policy: Any = {"policy": logmod.OPEN_POLICY}
        else:
            policy = issuers
        try:
            log_obj = logmod.Log.init(Path(directory), public, policy)
        except (ValueError, OSError) as exc:
            raise CliError(f"log init: {exc}") from None
        return 0, {
            "command": "log init",
            "path": str(log_obj.path),
            "log_id": log_obj.log_id(),
            "public": keys.encode_public(public),
            "issuer_policy": log_obj.issuer_policy()["policy"],
            "issuers": [entry["key"] for entry in issuers],
            "size": log_obj.size(),
        }

    if sub == "append":
        if len(positional) != 1:
            raise CliError("log append needs exactly one cert file")
        log_obj = _open_log(_require(opts, "log"))
        cert = _load_object(positional[0], "log append")
        blobs: dict[str, bytes] = {}
        for path in opts["blob"]:
            raw = _read_bytes(path, "--blob")
            blobs["sha256:" + hashlib.sha256(raw).hexdigest()] = raw
        try:
            index = log_obj.append(cert, blobs or None)
        except logmod.AppendRefused as exc:
            raise CliError(f"log append refused: {exc.reason}") from None
        return 0, {
            "command": "log append",
            "index": index,
            "id": cert.get("id"),
            "type": cert.get("type"),
            "size": log_obj.size(),
            "blobs": sorted(blobs),
        }

    if sub == "prove":
        if len(positional) != 1:
            raise CliError("log prove needs exactly one index")
        log_obj = _open_log(_require(opts, "log"))
        index = _as_int(positional[0], "index", minimum=0)
        tree_size = _as_int(opts["tree_size"], "tree_size", minimum=0) if opts["tree_size"] else None
        try:
            proof = log_obj.inclusion(index, tree_size)
        except (IndexError, ValueError) as exc:
            raise CliError(f"log prove: {exc}") from None
        payload = {
            "command": "log prove",
            "inclusion": proof,
            "sth": log_obj.latest_sth(),
            "size": log_obj.size(),
        }
        if opts["out"]:
            payload["written"] = _write_json(opts["out"], proof)
        return 0, payload

    if sub == "sth":
        log_obj = _open_log(_require(opts, "log"))
        seed = _load_seed(_require(opts, "key", "the log's own key signs an STH"))
        timestamp = opts["timestamp"] or _now()
        try:
            sth = log_obj.sth(seed, timestamp)
        except logmod.AppendRefused as exc:
            raise CliError(f"log sth refused: {exc.reason}") from None
        payload = {"command": "log sth", "sth": sth, "size": log_obj.size()}
        if opts["out"]:
            payload["written"] = _write_json(opts["out"], sth)
        return 0, payload

    if sub == "mirror":
        src = _require(opts, "log")
        _open_log(src)
        dst = _require(opts, "to")
        pinned_sth = _load_object(opts["pinned_sth"], "--pinned-sth") if opts["pinned_sth"] else None
        report = logmod.mirror(Path(src), Path(dst), _log_public(opts), pinned_sth)
        code = 0 if report.get("verified") else EXIT["invalid"]
        return code, {"command": "log mirror", "from": str(src), "to": str(dst), "report": report}

    if sub == "verify-cert":
        if len(positional) != 1:
            raise CliError("log verify-cert needs exactly one cert file")
        cert = _load_object(positional[0], "log verify-cert")
        outcome = certmod.check(cert)
        return (0 if outcome.ok else EXIT["invalid"]), {
            "command": "log verify-cert",
            "ok": outcome.ok,
            "id": outcome.id,
            "type": outcome.type,
            "reasons": list(outcome.reasons),
        }

    if sub == "verify-sth":
        if len(positional) != 1:
            raise CliError("log verify-sth needs exactly one sth file")
        sth = _load_object(positional[0], "log verify-sth")
        ok, reason = logmod.verify_sth(sth, _log_public(opts))
        return (0 if ok else EXIT["invalid"]), {
            "command": "log verify-sth",
            "ok": bool(ok),
            "reason": reason,
            "tree_size": sth.get("tree_size"),
        }

    if sub == "verify-inclusion":
        if len(positional) != 2:
            raise CliError("log verify-inclusion needs <proof.json> <sth.json>")
        proof = _load_object(positional[0], "log verify-inclusion <proof>")
        sth = _load_object(positional[1], "log verify-inclusion <sth>")
        ok, reason = logmod.verify_inclusion(proof, sth, _log_public(opts))
        return (0 if ok else EXIT["invalid"]), {
            "command": "log verify-inclusion",
            "ok": bool(ok),
            "reason": reason,
            "leaf_index": proof.get("leaf_index"),
            "tree_size": sth.get("tree_size"),
        }

    if sub == "verify-consistency":
        if len(positional) != 2:
            raise CliError("log verify-consistency needs <sth1.json> <sth2.json>")
        sth_m = _load_object(positional[0], "log verify-consistency <sth1>")
        sth_n = _load_object(positional[1], "log verify-consistency <sth2>")
        if opts["proof"]:
            proof = _load_object(opts["proof"], "--proof")
        elif opts["log"]:
            log_obj = _open_log(opts["log"])
            first, second = sth_m.get("tree_size"), sth_n.get("tree_size")
            if not isinstance(first, int) or not isinstance(second, int):
                raise CliError("verify-consistency: an STH without an integer tree_size")
            try:
                proof = log_obj.consistency(first, second)
            except (ValueError, IndexError) as exc:
                raise CliError(f"log verify-consistency: {exc}") from None
        else:
            raise CliError("log verify-consistency needs --proof <file> or --log <dir>")
        ok, reason = logmod.verify_consistency(sth_m, sth_n, proof, _log_public(opts))
        return (0 if ok else EXIT["invalid"]), {
            "command": "log verify-consistency",
            "ok": bool(ok),
            "reason": reason,
            "first": proof.get("first"),
            "second": proof.get("second"),
        }

    raise CliError(f"log: unknown subcommand {sub!r}")


# --------------------------------------------------------------------------- dispatch


# The whole v8 verb set. It is reachable only through `python -m styxx.v8`; the 7.x top-level
# `styxx` command surface is untouched and does not learn any of these names.
# GATED S11-01: recommendation implemented; operator may reverse.
_VERBS = {
    "key": cmd_key,
    "prereg": cmd_prereg,
    "fingerprint": cmd_fingerprint,
    "verify": cmd_verify,
    "battery": cmd_battery,
    "log": cmd_log,
}


def _dispatch(argv: Sequence[str]) -> tuple[int, dict]:
    if not argv:
        raise CliError("no verb: expected one of " + ", ".join(sorted(_VERBS)), usage=list(USAGE))
    verb = argv[0]
    if verb in ("help", "--help", "-h"):
        return 0, {"command": "help", "usage": list(USAGE), "verbs": sorted(_VERBS)}
    if verb == "--version":
        return 0, {"command": "version", "styxx": SCHEMA_VERSION}
    handler = _VERBS.get(verb)
    if handler is None:
        raise CliError(
            f"unknown verb {verb!r}: expected one of " + ", ".join(sorted(_VERBS)),
            usage=list(USAGE),
        )
    try:
        return handler(list(argv[1:]))
    except CliError as exc:
        exc.extra.setdefault("verb", verb)
        raise


def run(argv: Sequence[str]) -> tuple[int, dict]:
    """The whole CLI as ``(exit_code, payload)``. Never raises, never prints, never exits."""
    try:
        code, payload = _dispatch(list(argv))
    except CliError as exc:
        payload = {"error": exc.message}
        payload.update(exc.extra)
        return exc.code, payload
    except Exception as exc:  # a bug here is still a JSON object on stdout, never a traceback
        return EXIT["invalid"], {"error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return EXIT["invalid"], {"error": f"{argv[0] if argv else '?'}: payload is not an object"}
    return code, payload


def _dumps(payload: dict) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"result is not JSON-serializable: {exc}"}, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print one JSON object on stdout and return the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", newline="\n")
        except (ValueError, OSError):
            pass
    code, payload = run(args)
    stream.write(_dumps(payload) + "\n")
    stream.flush()
    return code
