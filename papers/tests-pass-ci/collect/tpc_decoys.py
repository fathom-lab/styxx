# -*- coding: utf-8 -*-
"""Decoys (PREREG section 6, P9): 30 per packet, 15 whose key is WHOLE_CONTRADICTED (YES) and 15 whose key is another
decided label (NO), each keyed and scored on its full step-2 label.

A lab agent that sits on no seat writes a spec (JSON): for each decoy its construction, its source (a pull request
opened before 2026-11-02 that is not among testreceipt's 138, real or with the synthetic change it carries), the
description, the CI record (the shape `tpc_items.render_ci` reads), the key (label, scope, evidence, preexisting) and
what each construction needs stated. This module checks every decoy against its construction by machine (the
testreceipt CI reader and test-name rule the collector uses), checks the set's mix is exactly P9's, renders each
decoy by the same code as a real item (`tpc_items.build_item`), refuses markers and shapes no real item shows, and
seals the set.

P9's mix, by construction (side, count in Q and P1, count in P2, key label, step-1 scope):

    YES  yes_log                 9  10  WHOLE_CONTRADICTED   WHOLE   evidence LOG
    YES  yes_names_only          2   1  WHOLE_CONTRADICTED   WHOLE   evidence NAMES_ONLY (no log, no annotation shown)
    YES  yes_stale               2   2  WHOLE_CONTRADICTED   WHOLE   STALE by the machine flag
    YES  yes_preexisting         2   2  WHOLE_CONTRADICTED   WHOLE   preexisting yes, shown by the CI record's own bytes
    NO   no_passing              2   0  NOT_CONTRADICTED     WHOLE   tests pass, nothing fails
    NO   no_lint_build           2   4  NOT_CONTRADICTED     WHOLE   only lint, format, type or build jobs fail
    NO   no_install_in_test_job  2   2  NOT_CONTRADICTED     WHOLE   a job named for tests failed at installation
    NO   no_disclosed            3   3  NOT_CONTRADICTED     WHOLE   the description discloses the failure; one names-only
    NO   no_part_beside          2   2  PART_BESIDE_FAILURE  PART    failure outside the part; one names-only (P2: none)
    NO   no_part_contradicted    2   2  PART_CONTRADICTED    PART    failure inside the claimed part
    NO   no_not_a_claim          2   2  NOT_A_CLAIM          NONE    a prediction or an unticked box beside failing tests

P2 differs (the review of 2026-11-02): no two-step real item of P2 shows nothing failing (P7: candidates and AU1 need
a test failure, AU2 a failed job or check, and AU3 and AU4 are step 1 only), so P2 has no `no_passing` decoy and its two
go to `no_lint_build`, AU2's shape; and P2 shows two names-only decoys, near the share of real candidate and AU1 rows
whose failures all lie outside Actions. Every P2 decoy must sit in a combination of claim reading and CI result that
P2's real two-step items show (`P2_COMBINATIONS`).

Agents (P9): P2's decoys hold M's agent mix, the largest-remainder allotment of 30 over P3.4's planned claimers with a
CI test result in M (`agent_mix`: 29 Claude Code, 1 Cursor), since the decisive count reads M's candidates only; each
P2 decoy states its source's agent (`source.agent`), and a description that lacks the text P2.1's frame requires of that
agent is refused, as no real item of P2 lacks it.

Markers (P9): a marker word is refused only in the text the construction itself writes (`written_text`: the lines it
inserts into the description and the words it changes, read against the source's description as read, `source.body`,
which every decoy states; a real decoy carries it unchanged). A source's own words stay as they are, marker words
included, so a marker word in an item says nothing of which items are decoys.

Shapes refused in every packet: a marker word the construction wrote; a date trace after P8.1's mask
(`tpc_items.date_traces`); a step listing
no real failing run shows; a failing Actions check run with no annotation (every failing Actions check run of the
development store carries one); an entry whose repository is not its source's; a synthetic CI change not declared as a
step flip, or a flip whose later steps are not those of a real failing run of the same job. In P2 also: a names-only
decoy with a failing Actions check run; a failing test-like Actions job shown without its log excerpt; an excerpt found
other than by the failing step's times. Across a set: one decoy per source pull request and two per repository; across
packets: no item, source pull request or near-duplicate description shared with another packet's decoys or real items.

Sealing (P9): the key is JSON; with a 32-byte random salt kept off the repository, `sha256(salt || bytes)` is the
digest committed before any seat reads. P2's decoy set is sealed before the freeze: the 30 decoy item texts, each with
its sha256, side, construction and full label, as one JSON set with its own salt; its digest fills the lock row at the
signing commit, and scoring refuses a P2 decoy whose bytes are not in the sealed set.

Transforms for synthetic decoys (`insert_line`, `set_box`, `fail_test_step`, `fail_install_step`, `names_only`)
return new objects and never change their input.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import secrets
from fractions import Fraction
from typing import Iterable, Optional

from tpc_const import AGENT_ORDER, DECOYS_PER_PACKET, M_CLAIMERS_CI_PLANNING, POSTFILTER, WINDOW_START, parse_utc
from tpc_items import CI_HEAD, build_item, date_traces, failing_test_like_jobs
from tpc_screen import _map, build_ci, is_test_step, screen, test_like_job

LABELS = ("WHOLE_CONTRADICTED", "PART_CONTRADICTED", "PART_BESIDE_FAILURE", "NOT_CONTRADICTED", "NOT_A_CLAIM",
          "NOT_DECIDABLE")
SCOPES = ("WHOLE", "PART", "NONE")
EVIDENCE = ("LOG", "NAMES_ONLY", "NONE")
PREEXISTING = ("yes", "no", "unknown")

# name: (side, count in Q and P1, label, scope)
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
# P2's counts where they differ from Q's and P1's (P9)
P2_COUNTS = {"yes_log": 10, "yes_names_only": 1, "no_passing": 0, "no_lint_build": 4}
# constructions of which exactly this many decoys are names-only (P9)
ONE_NAMES_ONLY = {"no_disclosed": 1, "no_part_beside": 1}
P2_NAMES_ONLY = {"no_disclosed": 1, "no_part_beside": 0}
# P2's two-step real items (P7): (screen reading, testreceipt's CI reading, whether any job or check failed)
P2_COMBINATIONS = (("claimer", "tests failed", None), ("none", "tests failed", None), ("claimer", "tests passed", True))
MAX_PER_REPOSITORY = 2  # decoys of one set from one repository
NEAR_DUPLICATE = 0.8  # share of the shorter description's word 3-shingles found in the other
MARKER = re.compile(r"(?i)\b(?:decoys?|synthetic|zz_\w*|known answers?|fabricated|planted)\b")
PR_REF = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+$")
INSTALL_STEP = re.compile(r"(?i)install|dependenc|setup|set up|restore|bootstrap|\bdeps?\b|\bsync\b|cache|pip|npm ci|yarn|"
                          r"pnpm|bundle|poetry|uv sync|composer|cargo fetch|go mod")
FAILED = ("failure",)  # after `_map`: failure and timed_out


class DecoyError(ValueError):
    pass


def mix(packet: Optional[str]) -> dict:
    """{construction: count} of a packet's set (P9)."""
    out = {n: v[1] for n, v in CONSTRUCTIONS.items()}
    if packet == "P2":
        out.update(P2_COUNTS)
    return out


def names_only_mix(packet: Optional[str]) -> dict:
    return dict(P2_NAMES_ONLY if packet == "P2" else ONE_NAMES_ONLY)


def agent_mix(packet: Optional[str]) -> Optional[dict]:
    """{agent: count} of a packet's decoys, or None where P9 holds no mix. P2: M's mix, the largest-remainder
    allotment of 30 over P3.4's planned claimers with a CI test result in M (ties to the larger share, then to
    AGENT_ORDER), every agent listed; Q's and P1's real items are not drawn from M (P9)."""
    if packet != "P2":
        return None
    total = sum(M_CLAIMERS_CI_PLANNING.values())
    quota = {a: Fraction(M_CLAIMERS_CI_PLANNING.get(a, 0) * DECOYS_PER_PACKET, total) for a in AGENT_ORDER}
    out = {a: int(q) for a, q in quota.items()}
    rest = DECOYS_PER_PACKET - sum(out.values())
    order = sorted(AGENT_ORDER, key=lambda a: (-(quota[a] - int(quota[a])), -quota[a], AGENT_ORDER.index(a)))
    for a in order[:rest]:
        out[a] += 1
    return out


def agent_problems(entry: dict, name: str) -> list:
    """A P2 decoy states its source's agent (one of P2.1's), and its description carries the text P2.1's frame
    requires of that agent's bodies (Claude Code's "claude code", Cursor's "cursor.com/"), as every real item of P2
    does."""
    agent = (entry.get("source") or {}).get("agent")
    if agent not in AGENT_ORDER:
        return [f"{name}: the source's agent {agent!r} is not one of P2.1's {AGENT_ORDER} (source.agent, P9)"]
    need = POSTFILTER.get(agent)
    if need and need not in (entry.get("body") or "").lower():
        return [f"{name}: the description lacks {need!r}, the text P2.1 requires of {agent}'s bodies (P9)"]
    return []


def _lines(t: str) -> list:
    return (t or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _changed_words(old: str, new: str) -> list:
    """The parts of `new` that are not `old`'s, each widened to the whole words it touches."""
    out = []
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, _, _, j1, j2 in sm.get_opcodes():
        if tag not in ("replace", "insert"):
            continue
        while j1 > 0 and _word_char(new[j1]) and _word_char(new[j1 - 1]):
            j1 -= 1
        while j2 < len(new) and _word_char(new[j2 - 1]) and _word_char(new[j2]):
            j2 += 1
        out.append(new[j1:j2].strip())
    return out


def written_text(entry: dict) -> Optional[list]:
    """The text the construction itself writes into the item (P9): the description's lines that are not the
    source's own (`source.body`, the description as read) and, in a line it changed, the words it changed. A step
    flip writes no text: it changes conclusions and carries the donor's real log, annotations and times. None when
    the source's description is not stated."""
    old = (entry.get("source") or {}).get("body")
    if not isinstance(old, str):
        return None
    a, b = _lines(old), _lines(entry.get("body") or "")
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "insert" or (tag == "replace" and i2 - i1 != j2 - j1):
            out += b[j1:j2]
        elif tag == "replace":
            for x, y in zip(a[i1:i2], b[j1:j2]):
                out += _changed_words(x, y)
    return [t for t in out if t.strip()]


# The runner's own steps: in every failing job of the pre-window development store (110 jobs) both ran, "Complete
# job" success in 110 of 110; and no post step there was listed for a step that did not run (0 of 221 post steps).
ALWAYS_RUN = ("Set up job", "Complete job")
NOT_RUN = ("skipped", "cancelled")


def step_pattern_problems(ci: dict) -> list:
    """Step listings no real run shows: "Set up job" or "Complete job" not run, or a post step listed for a step that
    did not run (a synthetic step flip that marked every later step skipped would show both). A cancelled job is not
    read: of 6,234 jobs in that store, the 5 that showed either pattern were all cancelled."""
    out = []
    for j in ci.get("jobs") or []:
        if j.get("conclusion") == "cancelled":
            continue
        steps = j.get("steps") or []
        by_name = {s.get("name"): s for s in steps}
        for s in steps:
            n = s.get("name") or ""
            if n in ALWAYS_RUN and s.get("conclusion") in NOT_RUN:
                out.append(f"job {j.get('name')!r}: step {n!r} -> {s.get('conclusion')}")
            main = by_name.get(n[len("Post "):]) if n.startswith("Post ") else None
            if main is not None and main.get("conclusion") in NOT_RUN:
                out.append(f"job {j.get('name')!r}: step {n!r} listed for a step that did not run")
    return out


def _failing_actions_checks(ci: dict) -> list:
    return [c for c in ci.get("check_runs") or []
            if c.get("app") == "github-actions" and _map(c.get("conclusion")) in FAILED]


def annotation_problems(ci: dict) -> list:
    """A failing Actions check run with no annotation: in the development store 94 of 94 carry at least one (73 read
    "Process completed with exit code N"), so one without is a shape no real item shows (P9)."""
    have = {a.get("check_run_id") for a in ci.get("annotations") or []}
    return [f"the failing Actions check run {c.get('name')!r} carries no annotation"
            for c in _failing_actions_checks(ci) if c.get("id") not in have]


def later_steps(job: dict, step_name: str) -> list:
    """The steps after `step_name` that are neither post steps nor the runner's own."""
    steps = sorted(job.get("steps") or [], key=lambda s: (s.get("number") or 0))
    names = [s.get("name") or "" for s in steps]
    if step_name not in names:
        raise DecoyError(f"job {job.get('id')} has no step {step_name!r}")
    k = names.index(step_name)
    return [s for s in steps[k + 1:] if not (s.get("name") or "").startswith("Post ")
            and (s.get("name") or "") not in ALWAYS_RUN]


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


def repo_of(ref: str) -> str:
    return norm_pr(ref).split("#", 1)[0]


def _norm_text(t: Optional[str]) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (t or "").lower()))


def _shingles(t: Optional[str], k: int = 3) -> set:
    w = _norm_text(t).split()
    return {tuple(w[i:i + k]) for i in range(max(1, len(w) - k + 1))} if w else set()


def near_duplicate(a: Optional[str], b: Optional[str]) -> bool:
    """Equal once lower-cased to letters and digits, or the shorter one's word 3-shingles found in the other at
    NEAR_DUPLICATE or more (a one-line change to a description is caught)."""
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    sa, sb = _shingles(a), _shingles(b)
    small = min(len(sa), len(sb))
    return small > 0 and len(sa & sb) / small >= NEAR_DUPLICATE


# ---- the CI record read by testreceipt's code ------------------------------------------------------------
def ci_outcome(ci: dict) -> str:
    """testreceipt's CiResult.tests() over a CI record (the collector's P6 reading, timed_out as failure)."""
    crs = [{"id": c.get("id"), "name": c.get("name"), "conclusion": c.get("conclusion"),
            "app": {"slug": c.get("app")}} for c in ci.get("check_runs") or []]
    jobs = {j["id"]: j for j in ci.get("jobs") or []}
    result, _ = build_ci("decoy", crs, jobs, ci.get("statuses") or [])
    return result.tests()[0]


def any_failed_job_or_check(ci: dict) -> bool:
    """The collector's `any_failed_job_or_check` (P7's AU2): a job or a check run that failed or timed out."""
    return (any(_map(j.get("conclusion")) in FAILED for j in ci.get("jobs") or [])
            or any(_map(c.get("conclusion")) in FAILED for c in ci.get("check_runs") or []))


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


def _pre_window(name: str, what: str, ref: dict, probs: list) -> Optional[str]:
    """The ref's owner/name when it names a pull request opened before 2026-11-02; else a problem is added."""
    try:
        repo = repo_of(ref.get("pr") or "")
    except DecoyError:
        probs.append(f"{name}: the {what} names no pull request (owner/name#n)")
        return None
    try:
        if parse_utc(ref.get("created_at") or "") >= WINDOW_START:
            probs.append(f"{name}: the {what} was opened {ref.get('created_at')}, not before 2026-11-02 (P9)")
    except ValueError:
        probs.append(f"{name}: the {what}'s created_at {ref.get('created_at')!r} is unreadable")
    return repo


def flip_problems(entry: dict, name: str) -> list:
    """A synthetic CI change is declared: `source.ci_changed` true or false, and when true a `flip` naming the job,
    the step turned to failing, the donor of its log excerpt and annotations (a pull request opened before 2026-11-02,
    its repository among `donor_repos` when it is not the entry's own) and, when the job has later steps that are not
    post steps, the real failing run of the same job they are copied from (`reference`, with `later`: each such step's
    conclusion), which the listing must equal (P9)."""
    probs = []
    src = entry.get("source") or {}
    if src.get("kind") != "synthetic":
        return probs
    if src.get("ci_changed") not in (True, False):
        probs.append(f"{name}: a synthetic decoy states whether its CI record was changed (source.ci_changed)")
        return probs
    if not src["ci_changed"]:
        return probs
    fl = entry.get("flip") or {}
    if not fl:
        return [f"{name}: a changed CI record is declared as a step flip (flip)"]
    job = next((j for j in (entry.get("ci") or {}).get("jobs") or [] if j.get("id") == fl.get("job_id")), None)
    if job is None:
        return [f"{name}: the flip names no job of the record"]
    try:
        later = later_steps(job, fl.get("step") or "")
    except DecoyError as e:
        return [f"{name}: {e}"]
    step = next(s for s in job.get("steps") or [] if s.get("name") == fl.get("step"))
    if _map(step.get("conclusion")) not in FAILED:
        probs.append(f"{name}: the flipped step does not fail")
    donor = fl.get("donor") or {}
    if donor:
        drepo = _pre_window(name, "flip's donor", donor, probs)
        own = (entry.get("repo") or "").lower()
        if drepo is not None and drepo != own and drepo not in {r.lower() for r in entry.get("donor_repos") or []}:
            probs.append(f"{name}: the donor's repository is not in donor_repos, so its name would stay in the log")
    if later:
        ref = fl.get("reference") or {}
        if not ref:
            probs.append(f"{name}: the flipped job has later steps that are not post steps; their conclusions come "
                         "from a real failing run of the same job (flip.reference), or the flip is refused")
        else:
            _pre_window(name, "flip's reference run", ref, probs)
            got = {s.get("name"): s.get("conclusion") for s in later}
            if got != (ref.get("later") or {}):
                probs.append(f"{name}: the flipped job's later steps are not the reference run's")
    return probs


# ---- one decoy against its construction ----------------------------------------------------------------
def check_decoy(entry: dict, built: dict, *, excluded: Iterable[str] = (), packet: Optional[str] = None) -> list:
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
        if (entry.get("repo") or "").lower() != repo_of(src["pr"]):
            probs.append(f"{name}: the entry's repo {entry.get('repo')!r} is not its source's; the CI block redacts "
                         "the repository the entry names (P8.3)")
    try:
        if parse_utc(src.get("created_at") or "") >= WINDOW_START:
            probs.append(f"{name}: source opened {src.get('created_at')}, not before 2026-11-02 (P9)")
    except ValueError:
        probs.append(f"{name}: source created_at {src.get('created_at')!r} unreadable")
    if src.get("kind") == "synthetic" and not (src.get("change") or "").strip():
        probs.append(f"{name}: a synthetic decoy states the change it carries")
    probs += flip_problems(entry, name)
    written = written_text(entry)
    if written is None:
        probs.append(f"{name}: the source's description as read is stated (source.body), so what the construction "
                     "writes can be read (P9)")
        written = []
    elif src.get("kind") == "real" and written:
        probs.append(f"{name}: a real decoy carries its source's description unchanged (P9)")
    for text in written:
        m = MARKER.search(text)
        if m:
            probs.append(f"{name}: the text the construction writes carries a marker word {m.group(0)!r} (P9)")
            break
    if packet == "P2":
        probs += agent_problems(entry, name)
    traces = date_traces(built.get("step1_text") or "") + date_traces(built.get("item_text") or "")
    if traces:
        kinds = ", ".join(f"{k} {traces.count(k)}" for k in sorted(set(traces)))
        probs.append(f"{name}: the item text carries a date trace after masking ({kinds}; P8.1, P9)")
    ci = entry.get("ci") or {}
    for p in step_pattern_problems(ci):
        probs.append(f"{name}: {p}, which no real failing job shows (P9)")
    for p in annotation_problems(ci):
        probs.append(f"{name}: {p}, which no real failing Actions check run shows (P9)")
    outcome = ci_outcome(ci)
    build = built.get("build") or {}
    logs_shown = [x for x in build.get("logs", []) if x.get("how") not in ("no_log", "item_budget_spent")]
    names_only = not logs_shown and not build.get("annotations_shown")
    body = entry.get("body") or ""
    failing = _failing_items(ci)
    if packet == "P2":
        readings = {screen(body)["screen"], screen(built.get("step1_text") or "")["screen"]}
        failed_any = any_failed_job_or_check(ci)
        for kind in sorted(readings):
            if not any(kind == k and outcome == o and (f is None or failed_any == f) for k, o, f in P2_COMBINATIONS):
                probs.append(f"{name}: the screen reads {kind!r} beside {outcome!r} (any job or check failed: "
                             f"{failed_any}), a combination no two-step real item of P2 shows (P7, P9)")
        if names_only and _failing_actions_checks(ci):
            probs.append(f"{name}: a names-only decoy of P2 shows failures outside Actions only (P9)")
        if any(x.get("how") == "no_log" for x in build.get("logs", [])):
            probs.append(f"{name}: a failing test-like Actions job is shown without its log excerpt; P2's real "
                         "candidates and AU1 rows have their logs read (P4.2, P9)")
        other = sorted({x.get("how") for x in logs_shown} - {"step_times"})
        if other:
            probs.append(f"{name}: an excerpt found by {other}, not by the failing step's times (P8.3, P9)")
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
        item = built.get("item_text") or ""
        head = "\n\n" + CI_HEAD + "\n"
        block = item.split(head, 1)[1] if head in item else ""
        if key.get("preexisting") != "yes" or not note or note not in block:
            probs.append("yes_preexisting: preexisting yes, and preexisting_shown quotes where the item's CI block "
                         "shows it")
        if note and note in (built.get("step1_text") or ""):
            probs.append("yes_preexisting: the description says the failure predates the change, a disclosure "
                         "(P11.2); the CI record shows it in its own bytes")
        if src.get("kind") == "synthetic" and src.get("ci_changed") is not False:
            probs.append("yes_preexisting: the CI record is the source's own, unchanged (source.ci_changed false): "
                         "no step flip or donor log shows the preexisting failure (P9)")
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


def check_set(entries: list, builts: list, *, packet: Optional[str] = None, excluded: Iterable[str] = (),
              other_sets: Iterable[str] = (), other_sources: Iterable[str] = (),
              other_texts: Iterable[str] = ()) -> list:
    """Problems of a set of 30: every decoy's, the packet's mix of P9, the names-only decoys P9 asks for, one decoy per
    source pull request and at most two per repository; and nothing shared with another packet: no item
    (`other_sets`: item sha256s), no source pull request (`other_sources`: the pull requests of other packets' decoys
    and real items) and no near-duplicate description (`other_texts`: other packets' step-1 texts)."""
    excluded = list(excluded)
    probs = []
    for e, b in zip(entries, builts):
        probs += check_decoy(e, b, excluded=excluded, packet=packet)
    if len(entries) != DECOYS_PER_PACKET:
        probs.append(f"a set holds {DECOYS_PER_PACKET} decoys, this one {len(entries)}")
    counts = {}
    for e in entries:
        counts[e.get("construction")] = counts.get(e.get("construction"), 0) + 1
    for name, n in mix(packet).items():
        if counts.get(name, 0) != n:
            probs.append(f"{name}: {counts.get(name, 0)} decoys, P9 asks for {n} in {packet or 'this packet'}")
    for name, want in names_only_mix(packet).items():
        got = sum(1 for e in entries if e.get("construction") == name and e.get("_names_only"))
        if got != want:
            probs.append(f"{name}: {got} names-only decoys, P9 asks for {want}")
    agents = agent_mix(packet)
    if agents is not None:
        got = {a: 0 for a in agents}
        for e in entries:
            a = (e.get("source") or {}).get("agent")
            got[a] = got.get(a, 0) + 1
        if got != agents:
            have = ", ".join(f"{a} {n}" for a, n in got.items() if n)
            want = ", ".join(f"{a} {n}" for a, n in agents.items() if n)
            probs.append(f"the decoys' agents are {have}; P9 holds {packet}'s to M's mix, {want}")
    shas = [b["item_sha256"] for b in builts]
    if len(set(shas)) != len(shas):
        probs.append("two decoys render to the same item")
    reused = set(shas) & set(other_sets)
    if reused:
        probs.append(f"{len(reused)} decoys are another packet's (P9: none reused)")
    sources, repos = {}, {}
    for e in entries:
        try:
            pr = norm_pr((e.get("source") or {}).get("pr") or "")
        except DecoyError:
            continue
        sources[pr] = sources.get(pr, 0) + 1
        repos[pr.split("#")[0]] = repos.get(pr.split("#")[0], 0) + 1
    if any(n > 1 for n in sources.values()):
        probs.append(f"{sum(1 for n in sources.values() if n > 1)} source pull requests carry more than one decoy "
                     "(P9: one each)")
    if any(n > MAX_PER_REPOSITORY for n in repos.values()):
        probs.append(f"{sum(1 for n in repos.values() if n > MAX_PER_REPOSITORY)} repositories carry more than "
                     f"{MAX_PER_REPOSITORY} decoys (P9)")
    others = set()
    for x in other_sources:
        try:
            others.add(norm_pr(x))
        except DecoyError:
            probs.append(f"other_sources names no pull request: {x!r}")
    shared = set(sources) & others
    if shared:
        probs.append(f"{len(shared)} decoys come from a pull request another packet uses (P9)")
    texts = [t for t in other_texts if t]
    near = sum(1 for b in builts if any(near_duplicate(b.get("step1_text"), t) for t in texts))
    if near:
        probs.append(f"{near} decoys' descriptions near-duplicate another packet's items (P9)")
    return probs


def build_set(spec: dict, *, excluded: Iterable[str] = (), other_sets: Iterable[str] = (),
              other_sources: Iterable[str] = (), other_texts: Iterable[str] = ()) -> tuple:
    """(set object, problems). spec: {"packet": name, "decoys": [entry, ...]}."""
    entries = [copy.deepcopy(e) for e in spec.get("decoys") or []]
    builts = [build_item(e.get("body"), e.get("ci"), e.get("repo"), also=tuple(e.get("donor_repos") or ()))
              for e in entries]
    probs = check_set(entries, builts, packet=spec.get("packet"), excluded=excluded, other_sets=other_sets,
                      other_sources=other_sources, other_texts=other_texts)
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


def _fail_from(ci: dict, job_id, step_name: str, log: Optional[str], post_skipped: Iterable[str] = (), *,
               later: Optional[dict] = None, annotations: Iterable[dict] = (),
               step_times: Optional[tuple] = None) -> dict:
    """The step turned to failing as a real run shows it: the runner's own "Set up job" and "Complete job" keep their
    conclusions; the job's later steps that are not post steps take the conclusions `later` gives, copied from a real
    failing run of the same job (a step that runs on failure, an upload or a report, succeeds there); with no such
    run they are refused, unless there are none; a post step is listed only for a step that ran, so the post steps of
    the steps that did not run leave the listing; the post steps of the steps that ran keep their conclusions, except
    those named in `post_skipped` (an action whose post step runs only on success, such as a cache save). The donor
    job's failure annotations (level and message) go on the flipped check run, and `step_times` (the donor's failing
    step's started_at and completed_at) on the flipped step, so its excerpt is found by the step's times as a real
    item's is (P8.3, P9)."""
    out = copy.deepcopy(ci)
    j = _job(out, job_id)
    steps = sorted(j.get("steps") or [], key=lambda s: (s.get("number") or 0))
    names = [s.get("name") or "" for s in steps]
    if step_name not in names:
        raise DecoyError(f"job {job_id} has no step {step_name!r}")
    post_skipped = set(post_skipped)
    bad = sorted(p for p in post_skipped if not p.startswith("Post ") or p not in names)
    if bad:
        raise DecoyError(f"post_skipped names no post step of job {job_id}: {bad}")
    k = names.index(step_name)
    rest = [s.get("name") or "" for s in later_steps(j, step_name)]
    if rest and later is None:
        raise DecoyError(f"job {job_id} has steps after {step_name!r} that are not post steps ({len(rest)}); their "
                         "conclusions come from a real failing run of the same job (later)")
    if rest and set(later or {}) != set(rest):
        raise DecoyError(f"later names {sorted(later or {})}, the job's later steps are {rest}")
    steps[k]["conclusion"] = "failure"
    if step_times is not None:
        steps[k]["started_at"], steps[k]["completed_at"] = step_times
    not_run = set()
    for s in steps[k + 1:]:
        n = s.get("name") or ""
        if n in ALWAYS_RUN or n.startswith("Post "):
            continue
        s["conclusion"] = (later or {}).get(n, "skipped")
        if s["conclusion"] in NOT_RUN:
            not_run.add(n)
    kept = []
    for s in steps:
        n = s.get("name") or ""
        if n.startswith("Post ") and n[len("Post "):] in not_run:
            continue
        if n in post_skipped:
            s["conclusion"] = "skipped"
        kept.append(s)
    if len(kept) != len(steps):
        for i, s in enumerate(kept, 1):
            s["number"] = i
    j["steps"] = kept
    j["conclusion"] = "failure"
    for c in out.get("check_runs") or []:
        if c.get("id") == job_id:
            c["conclusion"] = "failure"
    for a in annotations:
        out.setdefault("annotations", []).append({"check_run_id": job_id, "level": a.get("level"),
                                                  "message": a.get("message")})
    if log is not None:
        out.setdefault("logs", {})[str(job_id)] = log
    return out


def fail_test_step(ci: dict, job_id, step_name: str, log: Optional[str] = None, *,
                   post_skipped: Iterable[str] = (), later: Optional[dict] = None, annotations: Iterable[dict] = (),
                   step_times: Optional[tuple] = None) -> dict:
    """A passing test step turned to failing (P9), later steps as `_fail_from` says, the job and its check run
    failing with the donor's annotations; `log` is an excerpt taken from another real failing log of the same
    runner."""
    if not is_test_step(step_name):
        raise DecoyError(f"{step_name!r} is not a test-like step")
    return _fail_from(ci, job_id, step_name, log, post_skipped, later=later, annotations=annotations,
                      step_times=step_times)


def fail_install_step(ci: dict, job_id, step_name: str, log: Optional[str] = None, *,
                      post_skipped: Iterable[str] = (), later: Optional[dict] = None,
                      annotations: Iterable[dict] = (), step_times: Optional[tuple] = None) -> dict:
    """An installation step of a job named for tests turned to failing; the test steps after it are skipped."""
    if is_test_step(step_name) or not INSTALL_STEP.search(step_name):
        raise DecoyError(f"{step_name!r} is not an installation step")
    if not test_like_job(_job(ci, job_id)):
        raise DecoyError(f"job {job_id} is not named for tests")
    return _fail_from(ci, job_id, step_name, log, post_skipped, later=later, annotations=annotations,
                      step_times=step_times)


def names_only(ci: dict) -> dict:
    """The record with its logs and annotations dropped: only names and conclusions remain. P2 refuses it when a
    failing check run is an Actions job (P9): its real items show annotations and logs for those."""
    out = copy.deepcopy(ci)
    out["logs"] = {}
    out["annotations"] = []
    return out


def log_jobs(ci: dict) -> list:
    """The jobs whose logs an item shows (P8.3: failing test-like jobs, job-id order)."""
    return failing_test_like_jobs(ci)
