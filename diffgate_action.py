"""GitHub Action entry for styxx.diffgate — checkout-free, injection-safe.

Reads the event payload from GITHUB_EVENT_PATH (the summary text never touches a shell),
fetches the diff from the GitHub API with the workflow token, gates the summary against it
with ``styxx.diffgate.gate_diff_text``, writes a job-summary table, emits ::error::
annotations for contradictions, and exits per the gate verdict and the strict/soft-fail
inputs. Supports ``pull_request`` (body vs PR diff) and ``push`` (head commit message vs
compare diff). Anything else: reports and passes.

soft-fail (STYXX_SOFT_FAIL, which action.yml fills from its soft-fail input) is read by
``_soft_fail``, and only an explicit "false" blocks. The value is compared case-insensitively
with surrounding whitespace stripped. "false" blocks: the job fails on a contradicted claim, and
with strict on an UNCHECKABLE one too. "true" reports: every verdict goes to the job summary and
the annotations. Unset reads as "true", the input's default. Every other value, the empty string
included, reports as "true" does, and the script prints a warning naming the value it did not
recognise. In report mode the gate's verdicts never fail the job; an error the script does not
catch (an unreadable event file, for one) still does.
The reason the default reports, with its receipts, is in action.yml's description and the
CHANGELOG entry "the GitHub Action reports by default".
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

from styxx.diffgate import gate_diff_text

# PATH-2a (NOTE_path2a_third_pass_2026_09_30): the form of a reason the overlay writes, at its start, and the kinds
# it may move. Written out here, not imported, so this script reads a styxx without the overlay too
# (tests/test_diffgate_path2a.py pins the kinds to the module's REACH). Run as the Action runs it,
# `python <action path>/diffgate_action.py`, it imports the styxx package beside it, at the ref the workflow names
# (NOTE_path2a_sixth_pass_2026_09_30, I-1), not the one pip installs.
_OVERLAY_WHY = re.compile(r"(?:VERIFIED|CONTRADICTED) withheld by PATH-2a \((?:#97|#121|#97, #121|#101)\): ")
_OVERLAY_MAIN = ". main's reading: "
_OVERLAY_KINDS = frozenset({"file_created", "file_deleted", "file_touched", "files_changed_count", "only_touches",
                            "tests_added", "symbol_added"})


def api(url: str, accept: str) -> str:
    req = urllib.request.Request(url, headers={
        "Accept": accept, "User-Agent": "styxx-diffgate-action",
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def _write_summary(lines) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def _soft_fail(raw: str | None) -> tuple[bool, str | None]:
    """Read the soft-fail input. Returns ``(soft, unrecognised)``.

    Only an explicit "false" blocks. The value is compared case-insensitively with surrounding
    whitespace stripped: "false" gives ``(False, None)`` and "true" gives ``(True, None)``. Unset
    (``None``) is the input's default, "true". Every other value, the empty string included,
    reports as "true" does and comes back as ``unrecognised`` so the caller can name it in a
    warning.
    """
    if raw is None:
        return True, None
    value = raw.strip().lower()
    if value == "false":
        return False, None
    if value == "true":
        return True, None
    return True, raw


def main() -> int:
    strict = os.environ.get("STYXX_STRICT", "false").lower() == "true"
    # Only an explicit "false" blocks (see _soft_fail); any value it does not recognise reports, and is named here.
    soft, unrecognised = _soft_fail(os.environ.get("STYXX_SOFT_FAIL"))
    if unrecognised is not None:
        shown = repr(unrecognised).replace("%", "%25")      # repr escapes CR and LF; a workflow command needs % escaped
        print(f"::warning title=styxx diffgate - soft-fail value not recognised::soft-fail is {shown}, which is "
              "neither \"true\" nor \"false\" (compared case-insensitively, surrounding whitespace stripped), so "
              "this check reports and does not fail the job. Set soft-fail: \"false\" to fail it.")
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = json.loads(open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8").read())

    if event_name == "pull_request":
        summary = (event.get("pull_request") or {}).get("body") or ""
        diff_url = (event.get("pull_request") or {}).get("url", "")
        what = f"PR #{(event.get('pull_request') or {}).get('number', '?')} body"
    elif event_name == "push":
        summary = (event.get("head_commit") or {}).get("message") or ""
        diff_url = (event.get("compare", "")
                    .replace("github.com", "api.github.com/repos", 1)
                    .replace("/compare/", "/compare/", 1))
        # canonical form: repository.compare_url template
        tpl = (event.get("repository") or {}).get("compare_url", "")
        if tpl:
            diff_url = tpl.replace("{base}", event.get("before", "")) \
                          .replace("{head}", event.get("after", ""))
        what = "head commit message"
    else:
        print(f"styxx diffgate: event {event_name!r} not gated (pull_request/push only)")
        return 0

    if not summary.strip():
        print(f"styxx diffgate: empty {what} — nothing to gate")
        return 0
    try:
        diff = api(diff_url, "application/vnd.github.diff")
    except Exception as e:  # noqa: BLE001
        # This branch used to `return 0` under a comment saying "a broken fetch
        # must not fake a verdict" — and 0 IS the passing verdict. A gate that
        # could not read the diff has not cleared anything, so under `strict` it
        # now fails rather than passing quietly.
        print(f"::warning title=styxx diffgate - could not fetch the diff::"
              f"{e}. The gate did not run; this is not a pass.")
        _write_summary(["## styxx diffgate — DID NOT RUN", "",
                        f"The diff could not be fetched: `{e}`.", "",
                        "_This is not a clean result. A gate with no diff to read "
                        "cannot contradict anything, which is true of every "
                        "summary ever written._"])
        return 1 if (strict and not soft) else 0

    g = gate_diff_text(summary, diff, strict=strict)

    # `measured` is False when the fetched text yielded no file statuses and no
    # added lines — an empty diff, an error payload served with HTTP 200, an
    # HTML interstitial. PASS/FAIL cannot carry "this gate did not run".
    if not g.measured:
        print(f"::warning title=styxx diffgate - UNMEASURED::"
              f"{g.why_unmeasured}. The gate did not run; this is not a pass.")
        _write_summary(["## styxx diffgate — UNMEASURED", "",
                        f"**The gate did not run.** {g.why_unmeasured}", "",
                        f"Every claim in the {what} is reported UNCHECKABLE, not "
                        "verified. A PASS here would mean *nothing contradicted "
                        "the summary* — which is true of any summary when there "
                        "is no diff to read.", "",
                        "_Set `strict: true` and `soft-fail: \"false\"` to fail the job "
                        "on this._"])
        return 1 if (strict and not soft) else 0

    lines = ["## styxx diffgate — the summary vs the diff", "",
             f"**{g.verdict}** · {len(g.claims)} claim(s) checked on the {what}", ""]
    if g.claims:
        lines += ["| verdict | claim | evidence |", "|---|---|---|"]
        for c in g.claims:
            mark = {"VERIFIED": "✅", "CONTRADICTED": "❌", "UNCHECKABLE": "❓"}[c.verdict]
            # PATH-2a (NOTE_path2a_second_pass_2026_09_30): a reason the overlay wrote is shown whole, since its
            # leading 100 characters are the withheld verdict and the phrase, and main's reading comes after them.
            # Only a reason the overlay wrote (NOTE_path2a_third_pass_2026_09_30, A-2): an UNCHECKABLE claim of a
            # kind it may move, whose reason starts with its form. No reason main writes for those kinds starts so;
            # a reason that only contains the words, such as a DECLARE-1 MALFORMED one, is cut as main cuts it.
            # I-3 (NOTE_path2a_sixth_pass_2026_09_30): the overlay's own words whole, and main's reading after them cut
            # as main cuts a reason, at 100 characters, so a long path cannot carry the table past GitHub's step
            # summary limit.
            ours = (c.verdict == "UNCHECKABLE" and c.kind in _OVERLAY_KINDS
                    and _OVERLAY_WHY.match(c.why) is not None)
            if ours:
                head, sep, mains = c.why.partition(_OVERLAY_MAIN)
                why = head + sep + mains[:100]
            else:
                why = c.why[:100]
            lines.append(f"| {mark} {c.verdict} | {c.text[:80]} | {why} |")
    else:
        lines += ["_No diff-shaped claims found. The gate checks a closed template set "
                  "(touched/created/deleted paths, added functions/tests, file counts, "
                  "only-touches scopes, tests-pass); prose outside it is not judged._"]
    # This footer used to say "zero false accusations across both public validation
    # corpora". The README withdrew that claim (re-run at 7.46.0 the committed sweep
    # found four, all false accusations), and a footer that kept asserting it on every
    # gated PR in every adopter's repo was the same present-tense mistake one level down.
    # Which of the two the job did is said here, because the default changed: soft-fail reports
    # unless the workflow sets it to "false" (action.yml; CHANGELOG "the GitHub Action reports by
    # default"). The figures are the receipts' own; the CHANGELOG entry says where each comes from.
    if not soft:
        mode = ("_This workflow sets `soft-fail: \"false\"`, so the job fails on a ❌, and on a ❓ "
                "only with `strict: true`. ")
    elif unrecognised is not None:
        mode = ("_soft-fail has a value this Action does not recognise (the warning annotation names "
                "it), so this check reports and does not fail the job, as it does by default; only "
                "`soft-fail: \"false\"` fails it on a ❌, and on a ❓ only with `strict: true`. ")
    else:
        mode = ("_soft-fail is on, as it is by default, so this check reports and does not fail the "
                "job; a workflow that sets `soft-fail: \"false\"` fails it on a ❌, and on a ❓ only "
                "with `strict: true`. ")
    p = "https://github.com/fathom-lab/styxx/blob/main/papers/closed-model-frontier/"
    lines += ["", mode + "A path claim the diff does not show is reported ❓ UNCHECKABLE, not "
              "accused: the 100 accusations EXTERNAL-1 sampled from 71,016 external agent-authored "
              "PRs, 85 of them path claims, measured precision 0.23 against a preregistered 0.95 "
              f"floor ([EXTERNAL-1]({p}RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md), "
              f"2026-08-31; receipts [`external1_adjudication.json`]({p}external1_adjudication.json), "
              f"[`external1_summary.json`]({p}external1_summary.json)), and a repaired path accuser "
              "measured 0.16 on a held-out sample "
              f"([V14]({p}RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md), 2026-09-01; "
              f"receipt [`v14_adjudication.json`]({p}v14_adjudication.json)), so that accusation is "
              "withheld until a held-out repair clears the floor. Of the kinds that still accuse, "
              "`only_touches` is the one with a measured precision: 0.25, 2 of 8 accusations "
              f"correct ([PATH-1]({p}RESULT_path1_only_touches_repair_2026_09_17.md), 2026-09-17, "
              f"re-derived by [SCOPE-1]({p}RESULT_scope1_ABANDONED_2026_09_18.md), 2026-09-18; "
              f"receipt [`scope1_footprint.json`]({p}scope1_footprint.json)). The withdrawn "
              "zero-false-accusation claim, and what replaced it, are in the "
              "[styxx README](https://github.com/fathom-lab/styxx/blob/main/README.md)._"]
    _write_summary(lines)

    failing = False
    for c in g.claims:
        if c.verdict == "CONTRADICTED":
            failing = True
            print(f"::error title=styxx diffgate - contradicted claim::{c.text[:120]} - {c.why}")
        elif c.verdict == "UNCHECKABLE" and strict:
            failing = True
            print(f"::error title=styxx diffgate - uncheckable under strict::{c.text[:120]} - {c.why}")
        elif c.verdict == "UNCHECKABLE":
            print(f"::warning title=styxx diffgate - uncheckable::{c.text[:120]} - {c.why}")
    print(f"styxx diffgate: {g.verdict} - {len(g.claims)} claim(s), "
          f"{sum(1 for c in g.claims if c.verdict == 'CONTRADICTED')} contradicted")
    if failing and not soft:
        return 1
    if failing and soft:
        print("::warning::styxx diffgate found failures but soft-fail is on, as it is by default, so "
              "the job passes; set soft-fail: \"false\" to fail it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
