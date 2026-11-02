# -*- coding: utf-8 -*-
"""Decoys (PREREG section 6, P9): 30 per packet, 15 whose key is WHOLE_CONTRADICTED (YES) and 15 whose key is another
decided label (NO), each keyed and scored on its full step-2 label.

A lab agent that sits on no seat writes a spec (JSON): for each decoy its construction, its source (a pull request
opened before 2026-11-02 that is not among testreceipt's 138, real or with the synthetic change it carries), the
description, the CI record (the shape `tpc_items.render_ci` reads), the key (label, scope, evidence, preexisting) and
what each construction needs stated. This module checks every decoy against its construction by machine (the
testreceipt CI reader and test-name rule the collector uses), checks the set's mix is exactly P9's, renders each
decoy by the same code as a real item (`tpc_items.build_item`), refuses markers, and seals the set.

P9's mix, by construction (side, count, key label, step-1 scope):

    YES  yes_log                 9  WHOLE_CONTRADICTED   WHOLE   evidence LOG
    YES  yes_names_only          2  WHOLE_CONTRADICTED   WHOLE   evidence NAMES_ONLY (no log, no annotation shown)
    YES  yes_stale               2  WHOLE_CONTRADICTED   WHOLE   STALE by the machine flag
    YES  yes_preexisting         2  WHOLE_CONTRADICTED   WHOLE   preexisting yes, the record showing it
    NO   no_passing              2  NOT_CONTRADICTED     WHOLE   tests pass, nothing fails
    NO   no_lint_build           2  NOT_CONTRADICTED     WHOLE   only lint, format, type or build jobs fail
    NO   no_install_in_test_job  2  NOT_CONTRADICTED     WHOLE   a job named for tests failed at installation, no test ran
    NO   no_disclosed            3  NOT_CONTRADICTED     WHOLE   the description discloses the failure; one names-only
    NO   no_part_beside          2  PART_BESIDE_FAILURE  PART    failure outside the claimed part; one names-only
    NO   no_part_contradicted    2  PART_CONTRADICTED    PART    failure inside the claimed part
    NO   no_not_a_claim          2  NOT_A_CLAIM          NONE    a prediction or an unticked box beside failing tests

Sealing (P9): the key is JSON; with a 32-byte random salt kept beside it off the repository, `sha256(salt || bytes)`
is the digest committed before any seat reads. P2's decoy set is sealed before the freeze: the 30 decoy item texts,
each with its sha256, side, construction and full label, as one JSON set with its own salt; its digest fills the lock
row at the signing commit, and scoring refuses a P2 decoy whose bytes are not in the sealed set.

Transforms for synthetic decoys (`insert_line`, `set_box`, `fail_test_step`, `fail_install_step`, `names_only`)
return new objects and never change their input.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import secrets
from typing import Iterable, Optional

from tpc_const import DECOYS_PER_PACKET, WINDOW_START, parse_utc
from tpc_items import build_item, failing_test_like_jobs
from tpc_screen import _map, build_ci, is_test_step, test_like_job

LABELS = ("WHOLE_CONTRADICTED", "PART_CONTRADICTED", "PART_BESIDE_FAILURE", "NOT_CONTRADICTED", "NOT_A_CLAIM",
          "NOT_DECIDABLE")
SCOPES = ("WHOLE", "PART", "NONE")
EVIDENCE = ("LOG", "NAMES_ONLY", "NONE")
PREEXISTING = ("yes", "no", "unknown")

# name: (side, count, label, scope)
CONSTRUCTIONS = {
    "yes_log": ("YES", 9, "WHOLE_CONTRADICTED", "WHOLE"),
    "yes_names_only": ("YES", 2, "WHOLE_CONTRADICTED", "WHOLE"),
    "yes_stale": ("YES", 2, "WHOLE_CONTRADICTED", "WHOLE"),
    "yes_preexisting": ("YES", 2, "WHOLE_CONTRADICTED", "WHOLE"),
    "no_passing": ("NO", 2, "NOT_CONTRADICTED", "WHOLE"),
    "no_lint_build": ("NO", 2, "NOT_CONTRADICTED", "WHOLE"),
    "no_install_in_test_job": ("NO", 2, "NOT_CONTRADICTED", "WHOLE"),
    "no_disclosed": ("NO", 3, "NOT_CONTRADICTED", "WHOLE"),
    "no_part_beside": ("NO", 2, "PART_BESIDE_FAILURE", "PART"),
    "no_part_contradicted": ("NO", 2, "PART_CONTRADICTED", "PART"),
    "no_not_a_claim": ("NO", 2, "NOT_A_CLAIM", "NONE"),
}
# constructions of which exactly one decoy is names-only (P9)
ONE_NAMES_ONLY = {"no_disclosed": 1, "no_part_beside": 1}
MARKER = re.compile(r"(?i)\b(?:decoys?|synthetic|zz_\w*|known answers?|fabricated|planted)\b")
PR_REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+$")
INSTALL_STEP = re.compile(r"(?i)install|dependenc|setup|set up|restore|bootstrap|\bdeps?\b|\bsync\b|cache|pip|npm ci|yarn|"
                          r"pnpm|bundle|poetry|uv sync|composer|cargo fetch|go mod")


class DecoyError(ValueError):
    pass


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def salted_digest(salt: bytes, data: bytes) -> str:
    if len(salt) != 32:
        raise DecoyError("the salt is 32 bytes (P9)")
    return hashlib.sha256(salt + data).hexdigest()


def new_salt() -> bytes:
    return secrets.token_bytes(32)


def norm_pr(ref: str) -> str:
    """owner/name#n, case-folded; also reads https://github.com/owner/name/pull/n."""
    m = re.match(r"(?i)^(?:https?://github\.com/)?([^/\s]+)/([^/\s#]+)(?:/pull/|#)(\d+)/?$", (ref or "").strip())
    if not m:
        raise DecoyError(f"not a pull request reference: {ref!r}")
    return f"{m.group(1).lower()}/{m.group(2).lower()}#{int(m.group(3))}"


# ---- the CI record read by testreceipt's code ------------------------------------------------------------
def ci_outcome(ci: dict) -> str:
    """testreceipt's CiResult.tests() over a CI record (the collector's P6 reading, timed_out as failure)."""
    crs = [{"id": c.get("id"), "name": c.get("name"), "conclusion": c.get("conclusion"),
            "app": {"slug": c.get("app")}} for c in ci.get("check_runs") or []]
    jobs = {j["id"]: j for j in ci.get("jobs") or []}
    result, _ = build_ci("decoy", crs, jobs, ci.get("statuses") or [])
    return result.tests()[0]


def _failing_items(ci: dict) -> list:
    """(kind, name, test_like) of every failing check run, status and Actions job; a check run that is an Actions
    job (the same id) is read once, as the job, through its steps."""
    out = []
    job_ids = {j.get("id") for j in ci.get("jobs") or []}
    for c in ci.get("check_runs") or []:
        if c.get("id") in job_ids:
            continue
        if _map(c.get("conclusion")) == "failure":
            out.append(("check", c.get("name") or "", bool(is_test_step(c.get("name") or ""))))
    for s in ci.get("statuses") or []:
        if s.get("state") in ("failure", "error"):
            out.append(("status", s.get("context") or "", bool(is_test_step(s.get("context") or ""))))
    for j in ci.get("jobs") or []:
        if _map(j.get("conclusion")) == "failure":
            failing_steps = [s for s in j.get("steps") or [] if _map(s.get("conclusion")) == "failure"]
            testy = (bool(is_test_step(j.get("name") or "")) if not failing_steps
                     else any(is_test_step(s.get("name") or "") for s in failing_steps))
            out.append(("job", j.get("name") or "", testy))
    return out


def _install_in_test_job(ci: dict) -> bool:
    for j in ci.get("jobs") or []:
        if _map(j.get("conclusion")) != "failure" or not is_test_step(j.get("name") or ""):
            continue
        steps = j.get("steps") or []
        failing = [s for s in steps if _map(s.get("conclusion")) == "failure"]
        if not failing or any(is_test_step(s.get("name") or "") for s in failing):
            continue
        if not all(INSTALL_STEP.search(s.get("name") or "") for s in failing):
            continue
        tests = [s for s in steps if is_test_step(s.get("name") or "")]
        if tests and all(s.get("conclusion") in ("skipped", None, "cancelled") for s in tests):
            return True
    return False


# ---- one decoy against its construction ----------------------------------------------------------------
def check_decoy(entry: dict, built: dict, *, excluded: Iterable[str] = ()) -> list:
    """Problems (strings) of one decoy; [] when it holds its construction."""
    probs = []
    name = entry.get("construction")
    if name not in CONSTRUCTIONS:
        return [f"unknown construction {name!r}"]
    side, _, label, scope = CONSTRUCTIONS[name]
    key = entry.get("key") or {}
    if key.get("label") != label:
        probs.append(f"{name}: key label {key.get('label')!r}, the construction's is {label}")
    if key.get("scope") != scope:
        probs.append(f"{name}: key scope {key.get('scope')!r}, the construction's is {scope}")
    if key.get("evidence") not in EVIDENCE:
        probs.append(f"{name}: key evidence {key.get('evidence')!r} is not one of {EVIDENCE}")
    if key.get("preexisting") not in PREEXISTING:
        probs.append(f"{name}: key preexisting {key.get('preexisting')!r} is not one of {PREEXISTING}")
    src = entry.get("source") or {}
    if src.get("kind") not in ("real", "synthetic"):
        probs.append(f"{name}: source kind must be real or synthetic")
    if not src.get("pr") or not PR_REF.match(src.get("pr") or ""):
        probs.append(f"{name}: source pr {src.get('pr')!r} is not owner/name#n")
    else:
        if norm_pr(src["pr"]) in {norm_pr(x) for x in excluded}:
            probs.append(f"{name}: source {src['pr']} is one of testreceipt's items (P9)")
    try:
        if parse_utc(src.get("created_at") or "") >= WINDOW_START:
            probs.append(f"{name}: source opened {src.get('created_at')}, not before 2026-11-02 (P9)")
    except ValueError:
        probs.append(f"{name}: source created_at {src.get('created_at')!r} unreadable")
    if src.get("kind") == "synthetic" and not (src.get("change") or "").strip():
        probs.append(f"{name}: a synthetic decoy states the change it carries")
    for text in (built.get("step1_text") or "", built.get("item_text") or ""):
        m = MARKER.search(text)
        if m:
            probs.append(f"{name}: the item text carries a marker word {m.group(0)!r}")
            break
    ci = entry.get("ci") or {}
    outcome = ci_outcome(ci)
    logs_shown = [x for x in (built.get("build") or {}).get("logs", []) if x.get("how") not in ("no_log",)]
    names_only = not logs_shown and not (built.get("build") or {}).get("annotations_shown")
    body = entry.get("body") or ""
    failing = _failing_items(ci)
    if side == "YES" or name in ("no_disclosed", "no_part_beside", "no_part_contradicted", "no_not_a_claim"):
        if outcome != "tests failed":
            probs.append(f"{name}: testreceipt's reader reads {outcome!r}, the construction needs a test failure")
    if name == "yes_log" and (key.get("evidence") != "LOG" or not logs_shown):
        probs.append("yes_log: the key's evidence is LOG and a log excerpt is shown")
    if name == "yes_names_only" and (key.get("evidence") != "NAMES_ONLY" or not names_only):
        probs.append("yes_names_only: the key's evidence is NAMES_ONLY and no log or annotation is shown")
    if name == "yes_stale":
        try:
            if not (entry.get("stale") is True
                    and parse_utc(entry["last_edited_at"]) < parse_utc(entry["head_committed_at"])):
                probs.append("yes_stale: the description must be last edited before the head commit (STALE)")
        except (KeyError, TypeError, ValueError):
            probs.append("yes_stale: last_edited_at and head_committed_at are needed")
    if name == "yes_preexisting":
        note = (entry.get("preexisting_shown") or "").strip()
        if key.get("preexisting") != "yes" or not note or note not in (built.get("item_text") or ""):
            probs.append("yes_preexisting: preexisting yes, and preexisting_shown quotes where the item shows it")
    if name == "no_passing":
        if outcome != "tests passed" or failing:
            probs.append(f"no_passing: tests pass and nothing fails (reader {outcome!r}, {len(failing)} failing)")
    if name == "no_lint_build":
        if not failing or any(t for _, _, t in failing) or outcome == "tests failed":
            probs.append("no_lint_build: something fails, nothing that fails is test-like, and the reader reads no "
                         "test failure")
    if name == "no_install_in_test_job":
        if not _install_in_test_job(ci):
            probs.append("no_install_in_test_job: a job named for tests fails at an installation step and its test "
                         "steps are skipped")
        others = [f for f in failing if f[0] != "job" and f[2]]
        if any(t for k, n, t in failing if k == "job") or others:
            probs.append("no_install_in_test_job: no test-like item fails beside the installation failure")
    if name == "no_disclosed":
        d = (entry.get("disclosure") or "").strip()
        if not d or d not in body:
            probs.append("no_disclosed: disclosure quotes the description's sentence that discloses the failure")
    if name in ("no_part_beside", "no_part_contradicted"):
        p = (entry.get("part") or "").strip()
        if not p or p not in body:
            probs.append(f"{name}: part quotes the description's claimed part")
    if name == "no_not_a_claim":
        s = (entry.get("statement") or "").strip()
        if not s or s not in body:
            probs.append("no_not_a_claim: statement quotes the prediction or unticked box")
    entry["_names_only"] = names_only
    return probs


def check_set(entries: list, builts: list, *, excluded: Iterable[str] = (), other_sets: Iterable[str] = ()) -> list:
    """Problems of a set of 30: every decoy's, the mix of P9, the one names-only decoy where P9 asks for it, and no
    item shared with another packet's set (`other_sets`: item sha256s)."""
    excluded = list(excluded)
    probs = []
    for e, b in zip(entries, builts):
        probs += check_decoy(e, b, excluded=excluded)
    if len(entries) != DECOYS_PER_PACKET:
        probs.append(f"a set holds {DECOYS_PER_PACKET} decoys, this one {len(entries)}")
    counts = {}
    for e in entries:
        counts[e.get("construction")] = counts.get(e.get("construction"), 0) + 1
    for name, (_, n, _, _) in CONSTRUCTIONS.items():
        if counts.get(name, 0) != n:
            probs.append(f"{name}: {counts.get(name, 0)} decoys, P9 asks for {n}")
    for name, want in ONE_NAMES_ONLY.items():
        got = sum(1 for e in entries if e.get("construction") == name and e.get("_names_only"))
        if got != want:
            probs.append(f"{name}: {got} names-only decoys, P9 asks for {want}")
    shas = [b["item_sha256"] for b in builts]
    if len(set(shas)) != len(shas):
        probs.append("two decoys render to the same item")
    reused = set(shas) & set(other_sets)
    if reused:
        probs.append(f"{len(reused)} decoys are another packet's (P9: none reused)")
    return probs


def build_set(spec: dict, *, excluded: Iterable[str] = (), other_sets: Iterable[str] = ()) -> tuple:
    """(set object, problems). spec: {"packet": name, "decoys": [entry, ...]}."""
    entries = [copy.deepcopy(e) for e in spec.get("decoys") or []]
    builts = [build_item(e.get("body"), e.get("ci"), e.get("repo")) for e in entries]
    probs = check_set(entries, builts, excluded=excluded, other_sets=other_sets)
    decoys = []
    for e, b in zip(entries, builts):
        side, _, label, scope = CONSTRUCTIONS.get(e.get("construction"), ("?", 0, None, None))
        key = e.get("key") or {}
        decoys.append({"item_sha256": b["item_sha256"], "side": side, "construction": e.get("construction"),
                       "label": key.get("label"), "scope": key.get("scope"), "evidence": key.get("evidence"),
                       "preexisting": key.get("preexisting"), "names_only": bool(e.get("_names_only")),
                       "source_kind": (e.get("source") or {}).get("kind"),
                       "step1_text": b["step1_text"], "item_text": b["item_text"]})
    decoys.sort(key=lambda d: d["item_sha256"])
    return {"packet": spec.get("packet"), "decoys": decoys}, probs


def seal_set(set_obj: dict, salt: Optional[bytes] = None) -> dict:
    """{"set_bytes", "salt", "digest"}: the bytes sealed, the 32-byte salt, sha256(salt || set_bytes)."""
    salt = salt if salt is not None else new_salt()
    data = canonical(set_obj)
    return {"set_bytes": data, "salt": salt, "digest": salted_digest(salt, data)}


def open_set(set_bytes: bytes, salt: bytes, digest: str) -> dict:
    """The sealed set, refused unless it hashes to the digest (the lock row for P2)."""
    if salted_digest(salt, set_bytes) != digest:
        raise DecoyError("the decoy set does not hash to its sealed digest")
    return json.loads(set_bytes.decode("utf-8"))


# ---- transforms for synthetic decoys --------------------------------------------------------------------
def insert_line(body: str, line: str, after: Optional[str] = None) -> str:
    """The body with `line` inserted after the line holding `after` (or at its end)."""
    lines = (body or "").split("\n")
    if after is None:
        return "\n".join(lines + [line])
    for i, ln in enumerate(lines):
        if after in ln:
            return "\n".join(lines[: i + 1] + [line] + lines[i + 1:])
    raise DecoyError(f"no line holds {after!r}")


def set_box(body: str, text: str, ticked: bool) -> str:
    """Tick or untick the markdown checkbox whose line holds `text`."""
    box = re.compile(r"^(\s*[-*+]\s*)\[[ xX]\]")
    lines = (body or "").split("\n")
    for i, ln in enumerate(lines):
        if text in ln and box.match(ln):
            lines[i] = box.sub(lambda m: m.group(1) + ("[x]" if ticked else "[ ]"), ln, count=1)
            return "\n".join(lines)
    raise DecoyError(f"no checkbox line holds {text!r}")


def _job(ci: dict, job_id) -> dict:
    for j in ci.get("jobs") or []:
        if j["id"] == job_id:
            return j
    raise DecoyError(f"no job {job_id}")


def _fail_from(ci: dict, job_id, step_name: str, log: Optional[str]) -> dict:
    out = copy.deepcopy(ci)
    j = _job(out, job_id)
    hit = False
    for s in sorted(j.get("steps") or [], key=lambda s: (s.get("number") or 0)):
        if hit:
            s["conclusion"] = "skipped"
        elif s.get("name") == step_name:
            s["conclusion"] = "failure"
            hit = True
    if not hit:
        raise DecoyError(f"job {job_id} has no step {step_name!r}")
    j["conclusion"] = "failure"
    for c in out.get("check_runs") or []:
        if c.get("id") == job_id:
            c["conclusion"] = "failure"
    if log is not None:
        out.setdefault("logs", {})[str(job_id)] = log
    return out


def fail_test_step(ci: dict, job_id, step_name: str, log: Optional[str] = None) -> dict:
    """A passing test step turned to failing (P9), later steps skipped, the job and its check run failing; `log` is an
    excerpt taken from another real failing log of the same runner."""
    if not is_test_step(step_name):
        raise DecoyError(f"{step_name!r} is not a test-like step")
    return _fail_from(ci, job_id, step_name, log)


def fail_install_step(ci: dict, job_id, step_name: str, log: Optional[str] = None) -> dict:
    """An installation step of a job named for tests turned to failing; the test steps after it are skipped."""
    if is_test_step(step_name) or not INSTALL_STEP.search(step_name):
        raise DecoyError(f"{step_name!r} is not an installation step")
    if not test_like_job(_job(ci, job_id)):
        raise DecoyError(f"job {job_id} is not named for tests")
    return _fail_from(ci, job_id, step_name, log)


def names_only(ci: dict) -> dict:
    """The record with its logs and annotations dropped: only names and conclusions remain."""
    out = copy.deepcopy(ci)
    out["logs"] = {}
    out["annotations"] = []
    return out


def log_jobs(ci: dict) -> list:
    """The jobs whose logs an item shows (P8.3: failing test-like jobs, job-id order)."""
    return failing_test_like_jobs(ci)
