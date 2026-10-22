# -*- coding: utf-8 -*-
"""population — from one GH Archive hour's events to the agent pull-request records the plan admits.

A pure function of the hour's decoded lines, the frozen identity table and the plan's parameters.
It reads no network, no clock and no API.

What an event must be to be admitted, in this order (the earliest step that fails names the reason):

  1. a PullRequestEvent whose payload.action is 'opened' (every other event and action is skipped;
     in 2026 the archive also carries 'merged', 'closed', 'reopened', 'labeled', 'assigned', ...);
  2. agent work by the frozen identity table: the opening account (actor.login) or the head
     branch (payload.pull_request.head.ref) matches an admitted rule (identity_table.json);
  3. every field a record needs is present and well formed (missing_fields:<name> otherwise);
  4. not the lab's own: the repository's owner is not in excluded_owners and the opening account
     is not in excluded_actor_logins (excluded_owner, excluded_actor);
  5. not young, which is how planting is excluded: the opening account's numeric id is at most
     account_id_ceiling and the repository's numeric id at most repo_id_ceiling
     (account_too_young, repo_too_young);
  6. created inside the window [t0, t1) by the event's own created_at (created_at_outside_window).

Age without a creation date. GH Archive carries no created_at for an account or a repository: the
actor object is {id, login, display_login, gravatar_id, url, avatar_url} and the repo object is
{id, name, url}. What it does carry is the numeric id, and GitHub assigns account ids, and
separately repository ids, in increasing order of creation (assumption A-1, checkable by a stranger
with api.github.com/users/<login> and /repositories/<id>, one request each). So the plan fixes a
calibration hour, the hour that starts young_days before t0, and takes the largest actor id and the
largest repository id seen in any event of that hour (`id_ceilings`). Every account created after
that hour has a larger id than every account active in it, so 'id above the ceiling' contains
'created after t0 minus young_days', and therefore 'created after t0'. It also catches some accounts
created a little before the calibration hour (newer than every account active in it): the rule errs
toward excluding. A stranger re-derives the ceilings from the same public hour. What the rule does
not exclude: an old account in an old repository, which anyone holding one can still use to plant;
the per-repository and per-account caps (pool.py) bound how much one such holder can add.

What a 2026 payload carries (observed; `fields_census` re-records it over every hour streamed):
event {actor, created_at, id, payload, public, repo, type} (org when the repository has one);
payload {action, number, pull_request} plus {label, labels} or {assignee, assignees} on those
actions; pull_request {base, head, id, number, url}; head and base {ref, repo, sha}; head.repo
{id, name, url}. Not carried: title, body, the author object, created_at of the pull request,
draft, the file list or any diff. A pre-2025-10 payload carries the whole pull request object; when
it does, the record keeps its title and body as `as_opened`, and accuse.py prefers them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "identity_table.json")
TABLE_SCHEMA = "styxx.sealed-ahead/identity-table/v0"
RECORD_SCHEMA = "styxx.sealed-ahead/pool-record/v0"
MATCHES = ("actor_login", "head_ref_prefix")
_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DIGITS = re.compile(r"^[0-9]+$")
_REPO = re.compile(r"^[^/\s]+/[^/\s]+$")


def _sibling(name):
    key = "sealed_ahead_" + name
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


common = _sibling("common")


# ---------------------------------------------------------------- the frozen identity table

def load_identity_table(path: str = TABLE_PATH) -> dict:
    with open(path, "rb") as fh:
        table = json.loads(fh.read().decode("utf-8"))
    probs = table_problems(table)
    if probs:
        raise ValueError("identity table refused: " + "; ".join(probs))
    return table


def table_problems(table) -> list:
    if not isinstance(table, dict) or table.get("schema") != TABLE_SCHEMA:
        return [f"schema must be {TABLE_SCHEMA}"]
    rules = table.get("admitted")
    if not isinstance(rules, list) or not rules:
        return ["admitted must be a non-empty list"]
    probs, seen = [], set()
    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            probs.append(f"admitted[{i}] is not an object")
            continue
        for k in ("rule", "family", "match", "value", "evidence", "evidence_status", "third_party_can_forge"):
            if k not in r:
                probs.append(f"admitted[{i}] lacks {k}")
        if r.get("match") not in MATCHES:
            probs.append(f"admitted[{i}].match must be one of {MATCHES}")
        if not isinstance(r.get("value"), str) or not r.get("value"):
            probs.append(f"admitted[{i}].value must be a non-empty string")
        if not r.get("evidence"):
            probs.append(f"admitted[{i}] names no evidence")
        if r.get("rule") in seen:
            probs.append(f"rule {r.get('rule')!r} appears twice")
        seen.add(r.get("rule"))
    return probs


def rules_of(table: dict) -> list:
    """The operative part of the table: what matching reads, in matching order."""
    return [{"rule": r["rule"], "family": r["family"], "match": r["match"], "value": r["value"]}
            for r in table["admitted"]]


def rules_sha256(table: dict) -> str:
    return common.canon_sha256(rules_of(table))


def _fold(s: str) -> str:
    """Logins compare without ASCII case (GitHub treats them so); anything non-ASCII compares exactly."""
    return s.lower() if s.isascii() else s


def match_identity(actor_login, head_ref, rules: list) -> list:
    """The rules an opened event matches, in table order. Prefixes compare exactly, case included."""
    out = []
    for r in rules:
        if r["match"] == "actor_login":
            if isinstance(actor_login, str) and _fold(actor_login) == _fold(r["value"]):
                out.append(r)
        elif r["match"] == "head_ref_prefix":
            if isinstance(head_ref, str) and head_ref.startswith(r["value"]):
                out.append(r)
    return out


# ---------------------------------------------------------------- what the archive carries

def _keys(obj) -> str:
    return json.dumps(sorted(obj.keys())) if isinstance(obj, dict) else type(obj).__name__


class Census:
    """Which keys the events actually carry, as key sets with counts. No names, no rates. Streams."""

    PARTS = ("event", "actor", "repo", "pr_payload", "pull_request", "head", "base", "head_repo")

    def __init__(self):
        self.c = {k: Counter() for k in self.PARTS}
        self.actions, self.body, self.n = Counter(), Counter(), 0

    def add(self, e) -> None:
        self.n += 1
        self.c["event"][_keys(e)] += 1
        self.c["actor"][_keys(e.get("actor"))] += 1
        self.c["repo"][_keys(e.get("repo"))] += 1
        if e.get("type") != "PullRequestEvent":
            return
        p = e.get("payload") or {}
        self.c["pr_payload"][_keys(p)] += 1
        self.actions[str(p.get("action"))] += 1
        pr = p.get("pull_request") or {}
        self.c["pull_request"][_keys(pr)] += 1
        self.c["head"][_keys(pr.get("head"))] += 1
        self.c["base"][_keys(pr.get("base"))] += 1
        self.c["head_repo"][_keys((pr.get("head") or {}).get("repo"))] += 1
        self.body["has_body" if isinstance(pr, dict) and "body" in pr else "no_body"] += 1

    def result(self) -> dict:
        return {"events": self.n, "pr_actions": dict(sorted(self.actions.items())),
                "pr_body": dict(sorted(self.body.items())),
                "keysets": {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0])) for k, v in self.c.items()}}


class Ceilings:
    """The largest account id and repository id in one hour's events (any event type). Streams."""

    def __init__(self):
        self.acc = self.repo = self.n = 0

    def add(self, e) -> None:
        self.n += 1
        a = (e.get("actor") or {}).get("id")
        r = (e.get("repo") or {}).get("id")
        if type(a) is int and a > self.acc:
            self.acc = a
        if type(r) is int and r > self.repo:
            self.repo = r

    def result(self) -> dict:
        return {"events": self.n, "account_id_ceiling": self.acc, "repo_id_ceiling": self.repo}


def fields_census(events) -> dict:
    c = Census()
    for e in events:
        c.add(e)
    return c.result()


def id_ceilings(events) -> dict:
    c = Ceilings()
    for e in events:
        c.add(e)
    return c.result()


# ---------------------------------------------------------------- one event

class Context:
    """The parts of the plan population reads, parsed once."""

    def __init__(self, params: dict, table: dict):
        if rules_sha256(table) != params["identity_rules_sha256"]:
            raise ValueError("the identity table's rules digest is not the one the parameters name: "
                             f"{rules_sha256(table)} != {params['identity_rules_sha256']}")
        self.rules = rules_of(table)
        self.t0 = common.parse_utc(params["t0"])
        self.t1 = common.parse_utc(params["t1"])
        self.account_ceiling = params["account_id_ceiling"]
        self.repo_ceiling = params["repo_id_ceiling"]
        self.owners = {_fold(o) for o in params["excluded_owners"]}
        self.actors = {_fold(a) for a in params["excluded_actor_logins"]}


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def classify(event, ctx: Context, hour: str):
    """('skip', why) | ('exclude', reason) | ('admit', record). Pure."""
    if not isinstance(event, dict) or event.get("type") != "PullRequestEvent":
        return "skip", "not_pr_event"
    if _get(event, "payload", "action") != "opened":
        return "skip", "not_opened"
    login = _get(event, "actor", "login")
    head_ref = _get(event, "payload", "pull_request", "head", "ref")
    matched = match_identity(login, head_ref, ctx.rules)
    if not matched:
        return "skip", "not_agent"
    pr = _get(event, "payload", "pull_request")
    number = _get(event, "payload", "number")
    if number is None:
        number = _get(pr, "number")
    fields = {
        "event_id": event.get("id"),
        "created_at": event.get("created_at"),
        "actor_id": _get(event, "actor", "id"),
        "actor_login": login,
        "repo_id": _get(event, "repo", "id"),
        "repo": _get(event, "repo", "name"),
        "number": number,
        "head_sha": _get(pr, "head", "sha"),
        "head_ref": head_ref,
        "base_sha": _get(pr, "base", "sha"),
        "base_ref": _get(pr, "base", "ref"),
    }
    checks = {
        "event_id": lambda v: isinstance(v, str) and bool(_DIGITS.match(v)),
        "created_at": lambda v: isinstance(v, str),
        "actor_id": lambda v: type(v) is int and v > 0,
        "actor_login": lambda v: isinstance(v, str) and bool(v),
        "repo_id": lambda v: type(v) is int and v > 0,
        "repo": lambda v: isinstance(v, str) and bool(_REPO.match(v)),
        "number": lambda v: type(v) is int and v > 0,
        "head_sha": lambda v: isinstance(v, str) and bool(_SHA.match(v)),
        "head_ref": lambda v: isinstance(v, str) and bool(v),
        "base_sha": lambda v: isinstance(v, str) and bool(_SHA.match(v)),
        "base_ref": lambda v: isinstance(v, str) and bool(v),
    }
    for k, ok in checks.items():
        if not ok(fields[k]):
            return "exclude", f"missing_fields:{k}"
    try:
        created = common.parse_utc(fields["created_at"])
    except ValueError:
        return "exclude", "missing_fields:created_at"
    if _fold(fields["repo"].split("/", 1)[0]) in ctx.owners:
        return "exclude", "excluded_owner"
    if _fold(login) in ctx.actors:
        return "exclude", "excluded_actor"
    if fields["actor_id"] > ctx.account_ceiling:
        return "exclude", "account_too_young"
    if fields["repo_id"] > ctx.repo_ceiling:
        return "exclude", "repo_too_young"
    if not (ctx.t0 <= created < ctx.t1):
        return "exclude", "created_at_outside_window"
    head_repo = _get(pr, "head", "repo") or {}
    as_opened = None
    if isinstance(pr, dict) and "body" in pr:
        title, body = pr.get("title"), pr.get("body")
        as_opened = {"title": title if isinstance(title, str) else None,
                     "body": body if isinstance(body, str) else None}
    families = {r["family"] for r in matched}
    record = {
        "schema": RECORD_SCHEMA,
        "event_id": fields["event_id"],
        "hour": hour,
        "opened_at": common.fmt_utc(created),
        "repo_id": fields["repo_id"],
        "repo": fields["repo"],
        "number": fields["number"],
        "head_sha": fields["head_sha"],
        "head_ref": fields["head_ref"],
        "head_repo_id": head_repo.get("id") if type(head_repo.get("id")) is int else None,
        "head_repo": head_repo.get("name") if isinstance(head_repo.get("name"), str) else None,
        "base_sha": fields["base_sha"],
        "base_ref": fields["base_ref"],
        "actor_id": fields["actor_id"],
        "actor_login": login,
        "family": matched[0]["family"] if len(families) == 1 else "mixed",
        "rules": [r["rule"] for r in matched],
        "as_opened": as_opened,
    }
    return "admit", record


class BadLines:
    """Lines that do not decode to a JSON object: counted and hashed as they pass, never kept."""

    def __init__(self):
        self.n = 0
        self._h = hashlib.sha256()

    def add(self, raw: bytes) -> None:
        if self.n:
            self._h.update(b"\n")
        self._h.update(raw)
        self.n += 1

    def result(self) -> dict:
        return {"bad_json_lines": self.n, "bad_json_lines_sha256": self._h.hexdigest()}


def iter_events(lines, bad: BadLines):
    """Raw archive lines (bytes, newline kept or not) -> decoded event objects, one at a time."""
    for raw in lines:
        raw = raw.rstrip(b"\r\n")
        if not raw.strip():
            continue
        try:
            e = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            bad.add(raw)
            continue
        if not isinstance(e, dict):
            bad.add(raw)
            continue
        yield e


def scan_events(events, ctx: Context, hour: str, observers=()) -> dict:
    """One hour, streamed: the admitted records and the hour's counts (by reason, and by family for
    sizing only; no rates). Each observer's add(event) sees every event (Census, Ceilings)."""
    records, skipped, excluded, families = [], Counter(), Counter(), Counter()
    n = 0
    for e in events:
        n += 1
        for o in observers:
            o.add(e)
        kind, out = classify(e, ctx, hour)
        if kind == "admit":
            records.append(out)
            families[out["family"]] += 1
        elif kind == "exclude":
            excluded[out] += 1
        else:
            skipped[out] += 1
    return {"hour": hour, "events": n, "records": records,
            "skipped": dict(sorted(skipped.items())), "excluded": dict(sorted(excluded.items())),
            "admitted": len(records), "admitted_by_family": dict(sorted(families.items()))}


def scan_lines(lines, ctx: Context, hour: str, observers=()) -> dict:
    """scan_events over raw lines, with the undecodable lines counted and hashed."""
    bad = BadLines()
    out = scan_events(iter_events(lines, bad), ctx, hour, observers)
    out.update(bad.result())
    return out


if __name__ == "__main__":  # pragma: no cover
    t = load_identity_table()
    print(json.dumps({"identity_rules_sha256": rules_sha256(t), "rules": rules_of(t)}, indent=1))
