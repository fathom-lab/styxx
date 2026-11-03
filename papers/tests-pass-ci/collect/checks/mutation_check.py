# -*- coding: utf-8 -*-
"""Break one rule of the collector at a time, in a copy, and check that the tests catch each break.

    py -3.12 checks/mutation_check.py <scratch dir> [pytest target ...]

With no pytest target the copy's own tests/ folder runs (the development layout under
_direction_a_2026_11_02/collector). In the lab's repository the collector's tests live in tests/:
run from the repository root and pass them, e.g.

    py -3.12 papers/tests-pass-ci/collect/checks/mutation_check.py <scratch> tests/test_tests_pass_ci_collector_*.py

The copy is named to those tests by the environment variable TPC_COLLECT_DIR. It sits at
<scratch>/mut/tests-pass-ci/collect with the PREREG copied beside its folder, as in the repository, so code that
reads the PREREG (the seats' rubric, the scorer's table) reads the same bytes. Nothing outside the scratch
directory is written. Rules 21 to 39 are the packet and decoy builder's, the seat runners' and the scorer's; rules
40 and 41 are the log excerpt's step boundaries, repaired while packet P2's decoys were built; the rules after them
answer the open findings of 2026-11-02 (dates, step flips, the shuffle, the seal guard, the signed PREREG, the salts),
and the last group the second review of 2026-11-02 (P2's shapes, annotations, flips, dates, repository names, reuse
across packets, the pinned main and the signing commit, the committed digests, and the reviewer's ten surviving
mutants R1 to R4, R10, R17, R18, R29, R31 and R32).

A rule may name a test file (its fifth field) that runs ahead of the others; every target still runs until one fails, so the hint
only orders the run and never narrows it.
"""
import os
import pathlib
import shutil
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent
M = [
    ("seal check never raises", "tpc_seal.py", "        if self.in_window(created_at) and not self.open:", "        if False:"),
    ("seal skips the ancestry check", "tpc_seal.py", "        if not self._on_main(self.repo, commit):", "        if False:"),
    ("seal reads any signature line", "tpc_seal.py", "        if not signed(text):", "        if False:"),
    ("seal reads the markers anywhere", "tpc_seal.py",
     "    return bool(lines) and all(ln.startswith(SIGNED) and not ln.startswith(UNSIGNED) for ln in lines)",
     "    return SIGNED in text and UNSIGNED not in text"),
    ("seal ignores TO-FILL rows", "tpc_seal.py", "        if to_fill_rows(text):", "        if False:"),
    ("seal ignores the running code", "tpc_seal.py", "        if sealed_code[0] != running:", "        if False:"),
    ("seal takes a future anchor", "tpc_seal.py", "        if self._now() < t:", "        if False:"),
    ("read_row skips the guard", "tpc_read.py", '    guard.check(row.get("created_at"), f"CI read of',
     '    (lambda *a: None)(row.get("created_at"), f"CI read of'),
    ("batch skips the guard", "tpc_read.py", '        guard.check(r.get("created_at"), f"body read of',
     '        (lambda *a: None)(r.get("created_at"), f"body read of'),
    ("stage_read drops the sealed filter", "tpc_collect.py", "    if sealed and not ctx.guard.open:", "    if False:"),
    ("cap by u", "tpc_collect.py", '    for r in sorted(rows, key=lambda x: (x["v"], x["u"])):',
     '    for r in sorted(rows, key=lambda x: x["u"]):'),
    ("M uses P_AGENT for all", "tpc_draw.py", "    in_m = in_d and below(h, p_main(agent))", "    in_m = in_d"),
    ("timed_out not mapped", "tpc_screen.py", '    return "failure" if conclusion == "timed_out" else conclusion',
     "    return conclusion"),
    ("slice formula label", "tpc_frame.py", '(("tests-pass-ci/slice/" + d.isoformat())',
     '(("tests-pass-ci/slices/" + d.isoformat())'),
    ("postfilter dropped", "tpc_read.py", "    if need and need not in body.lower():", "    if False:"),
    ("au1 logs dropped", "tpc_read.py",
     '    au1_eligible = row["in_M"] and sc["screen"] == "none" and outcome == "tests failed"',
     "    au1_eligible = False"),
    ("rule R ignores staleness", "tpc_rule.py", "    item3 = stale is False", "    item3 = True"),
    ("rule R reads local claims", "tpc_rule.py", " or LOCAL.search(text):", ":"),
    ("dry run asks for databaseId", "tpc_const.py", 'DRYRUN_FIELDS = "number createdAt',
     'DRYRUN_FIELDS = "databaseId number createdAt'),
    ("frame keeps forks", "tpc_frame.py", '    return (row.get("stars") or 0) >= MIN_STARS and row.get("fork") is False',
     '    return (row.get("stars") or 0) >= MIN_STARS'),
    # the packet and decoy builder, the seat runners, the scorer
    ("images kept in the description", "tpc_items.py", '    text = MD_IMAGE.sub("", text)', "    text = text"),
    ("excerpt ignores the step's times", "tpc_items.py", "        sp = _step_span(lines, s)", "        sp = None"),
    ("decoy mix unchecked", "tpc_decoys.py", "        if counts.get(name, 0) != n:", "        if False:"),
    ("stand-in hash accepted", "tpc_beacon.py", "    if bh == STAND_IN_HEX:", "    if False:"),
    ("window draw under any Bitcoin block", "tpc_collect.py", "    if not why and not ctx.dev:", "    if False:"),
    ("AU4 may be cut", "tpc_packet.py", '    never = set(in_m_cands) | set(au["AU4"]) | set(rule_r)',
     "    never = set(in_m_cands) | set(rule_r)"),
    ("top-ups cut by lowest w", "tpc_packet.py", "                k = keep_topups.pop()  # the highest w",
     "                k = keep_topups.pop(0)"),
    ("shuffle ignores the seed", "tpc_packet.py", "    rng = random.Random(shuffle_seed(seed_hex, packet, salt))",
     "    rng = random.Random(0)"),
    ("spool not read back", "tpc_seats.py", "    if back != data:", "    if False:"),
    ("inconsistent step-2 label accepted", "tpc_seats.py", '    if not consistent(scope, obj["label"]):', "    if False:"),
    ("dev server on any packet", "tpc_seats.py",
     '    if getattr(seat, "dev", False) and packet not in DEV_PACKETS and not allow_dev:', "    if False:"),
    ("step 2 asked after a step-1 NO_ANSWER", "tpc_seats.py", '                if s1.get("answer") is None:',
     "                if False:"),
    ("decoys scored on the side", "tpc_score.py", '            ok = bool(s2) and s2.get("label") == k["label"]',
     '            ok = bool(s2) and (s2.get("label") == "WHOLE_CONTRADICTED") == (k["side"] == "YES")'),
    ("NO-side bar at 12", "tpc_const.py", "QUALIFY_NO = 14  #", "QUALIFY_NO = 12  #"),
    ("one voter decides", "tpc_score.py", "        if n >= 2:", "        if n >= 1:"),
    ("undecided whole candidates dropped", "tpc_score.py",
     "    whole_up = wilson(k + d_w, n_eff)[1] if n_eff > 0 else 1.0", "    whole_up = wilson(k, n_eff)[1] if n_eff > 0 else 1.0"),
    ("n_eff is n", "tpc_score.py", "    n_eff = math.floor(n_pass * s_lo) + c_fail", "    n_eff = n"),
    ("answers need no seal", "tpc_score.py", "        if not sp.exists():", "        if False:"),
    ("P2 decoys not held to the sealed set", "tpc_score.py", "            if d is None:", "            if False:"),
    # the excerpt's step boundaries (repaired while P2's decoys were built)
    ("excerpt runs into the post steps", "tpc_items.py",
     "        if (t is not None and t > t1) or lines[i][1].startswith(POST_STEPS):",
     "        if t is not None and t > t1:"),
    ("excerpt runs past the step's error line", "tpc_items.py", "    return errs[-1] + 1 if errs else end",
     "    return end"),
    # dates masked in every item (P8.1)
    ("dates kept in the items", "tpc_items.py",
     "    return EPOCH.sub(_epoch_sub, MONTH_ABBR.sub(DATE_TOKEN, DATES.sub(DATE_TOKEN, text)))", "    return text"),
    ("a decoy may carry a date", "tpc_decoys.py",
     '    traces = date_traces(built.get("step1_text") or "") + date_traces(built.get("item_text") or "")',
     "    traces = []", "items_decoys"),
    # step flips as real runs list them
    ("a step flip skips the runner's own steps", "tpc_decoys.py",
     '        if n in ALWAYS_RUN or n.startswith("Post "):', "        if False:", "items_decoys"),
    ("a step flip keeps the posts of skipped steps", "tpc_decoys.py",
     '        if n.startswith("Post ") and n[len("Post "):] in not_run:', "        if False:", "items_decoys"),
    ("impossible step listings accepted", "tpc_decoys.py",
     '    for p in step_pattern_problems(ci):', "    for p in []:", "items_decoys"),
    # P8.5: the order says nothing about which items are decoys
    ("the shuffle starts from the reals then the decoys", "tpc_packet.py", "    entries.sort(key=lambda e: e[2])",
     "    pass"),
    ("the shuffle keyed by the public seed alone", "tpc_packet.py",
     '    return int(hashlib.sha256((seed_hex + ":" + packet + ":" + salt.hex()).encode("utf-8")).hexdigest(), 16)',
     '    return int(hashlib.sha256((seed_hex + ":" + packet).encode("utf-8")).hexdigest(), 16)'),
    # the seal guard: the rows filled at signing, the file it reads, the fetch, the signed text
    ("seal skips the rows filled at signing", "tpc_seal.py", "        if unfilled:", "        if False:"),
    ("seal requires the collector row only", "tpc_seal.py", "SIGNING_ROWS = (COLLECTOR_ROW_PREFIX, P2_SET_ROW_PREFIX)",
     "SIGNING_ROWS = (COLLECTOR_ROW_PREFIX,)"),
    ("seal reads any prereg_path", "tpc_seal.py", "        if path != PREREG_REPO_PATH:", "        if False:"),
    ("seal opens after a failed fetch", "tpc_seal.py", '        if st["fetch"] != "ok":', "        if False:"),
    ("signed text handed to any working copy", "tpc_seal.py", '            if got != self.state["prereg_blob_sha256"]:',
     "            if False:"),
    ("P2 build reads the working copy's lock row", "tpc_packet.py",
     "            prereg = guard.signed_prereg(PREREG_FILE)",
     '            prereg = PREREG_FILE.read_text(encoding="utf-8")'),
    ("scorer reads the working copy's lock row", "tpc_score.py",
     '            return guard.signed_prereg(PREREG_FILE), "the signed blob, through the seal guard"',
     '            return PREREG_FILE.read_text(encoding="utf-8"), "the signed blob, through the seal guard"'),
    ("P2's packet key digest never needed", "tpc_score.py",
     '        kp = parse_utc(p["key_public_at"])  # every packet\'s key digest public before its seats read (section 14)',
     '        kp = parse_utc(p.get("key_public_at") or p["key_released_at"])', "score"),
    # where salts and keys may be written (P9)
    ("a salt may sit inside a git working tree", "tpc_packet.py", "    if salt:", "    if False:"),
    ("secrets written where git does not ignore them", "tpc_packet.py", "    if r.returncode == 0:\n        return None",
     "    if True:\n        return None"),
    ("the key's salt written beside the key", "tpc_packet.py", 'salt_dir / f"key_{p}.salt"', 'out / f"key_{p}.salt"'),
    # the second review of 2026-11-02: dates (finding 4)
    ("the mask leaves Unix times", "tpc_items.py", "    return EPOCH.sub(_epoch_sub, MONTH_ABBR.sub(DATE_TOKEN, DATES.sub",
     "    return (lambda f, t: t)(_epoch_sub, MONTH_ABBR.sub(DATE_TOKEN, DATES.sub", "items_decoys"),
    ("the mask leaves month abbreviations", "tpc_items.py", "_epoch_sub, MONTH_ABBR.sub(DATE_TOKEN, DATES.sub",
     "_epoch_sub, (lambda a, t: t)(DATE_TOKEN, DATES.sub", "items_decoys"),
    ("the decoy date check misses Unix times", "tpc_items.py", "        if TRACE_EPOCH[0] <= s < TRACE_EPOCH[1]:",
     "        if False:", "items_decoys"),
    # repository names (finding 5)
    ("runner paths keep the repository", "tpc_items.py",
     "    text = RUNNER_WORK.sub(lambda m: m.group(1) + REPO_TOKEN + m.group(3) + REPO_TOKEN, text)", "    pass",
     "items_decoys"),
    ("short names stay in the CI block", "tpc_items.py", "    for t in _name_tokens(repos):", "    for t in []:",
     "items_decoys"),
    ("a donor's repository stays", "tpc_items.py", "    repos = [r for r in (repo,) + tuple(also) if r]",
     "    repos = [r for r in (repo,) if r]", "items_decoys"),
    ("an entry may name another repository", "tpc_decoys.py",
     '        if (entry.get("repo") or "").lower() != repo_of(src["pr"]):', "        if False:", "items_decoys"),
    # P2's shapes (findings 1 and 2)
    ("P2 takes any claim and CI combination", "tpc_decoys.py",
     "            if not any(kind == k and outcome == o and (f is None or failed_any == f) for k, o, f in P2_COMBINATIONS):",
     "            if False:", "items_decoys"),
    ("P2 keeps Q's and P1's mix", "tpc_decoys.py",
     'P2_COUNTS = {"yes_log": 10, "yes_names_only": 1, "no_passing": 0, "no_lint_build": 4}', "P2_COUNTS = {}",
     "items_decoys"),
    ("P2 keeps two names-only decoys of no_part_beside's", "tpc_decoys.py",
     'P2_NAMES_ONLY = {"no_disclosed": 1, "no_part_beside": 0}', 'P2_NAMES_ONLY = {"no_disclosed": 1, "no_part_beside": 1}',
     "items_decoys"),
    ("a failing Actions check run needs no annotation", "tpc_decoys.py", "    for p in annotation_problems(ci):",
     "    for p in []:", "items_decoys"),
    ("P2 names-only inside Actions", "tpc_decoys.py", "        if names_only and _failing_actions_checks(ci):",
     "        if False:", "items_decoys"),
    ("P2 shows a failing test job without its log", "tpc_decoys.py",
     '        if any(x.get("how") == "no_log" for x in build.get("logs", [])):', "        if False:", "items_decoys"),
    ("P2 takes excerpts found by markers", "tpc_decoys.py",
     '        other = sorted({x.get("how") for x in logs_shown} - {"step_times"})', "        other = []", "items_decoys"),
    # step flips (finding 3)
    ("a flip skips every later step", "tpc_decoys.py", "    if rest and later is None:", "    if False:", "items_decoys"),
    ("a changed CI record needs no flip", "tpc_decoys.py",
     '        return [f"{name}: a changed CI record is declared as a step flip (flip)"]', "        return []",
     "items_decoys"),
    ("a donor's repository need not be redacted", "tpc_decoys.py",
     '        if drepo is not None and drepo != own and drepo not in {r.lower() for r in entry.get("donor_repos") or []}:',
     "        if False:", "items_decoys"),
    ("a flip's later steps unchecked against its reference", "tpc_decoys.py",
     '            if got != (ref.get("later") or {}):', "            if False:", "items_decoys"),
    # reuse within a set and across packets (finding 6)
    ("a source pull request may carry two decoys", "tpc_decoys.py", "    if any(n > 1 for n in sources.values()):",
     "    if False:", "items_decoys"),
    ("a repository may carry three decoys", "tpc_decoys.py",
     "    if any(n > MAX_PER_REPOSITORY for n in repos.values()):", "    if False:", "items_decoys"),
    ("another packet's sources accepted", "tpc_decoys.py", "    shared = set(sources) & others", "    shared = set()",
     "items_decoys"),
    ("near-duplicate descriptions accepted", "tpc_decoys.py",
     '    near = sum(1 for b in builts if any(near_duplicate(b.get("step1_text"), t) for t in texts))', "    near = 0",
     "items_decoys"),
    ("Q takes claimers with nothing failing", "tpc_packet.py",
     "    rows = [r for r in _read_rows(root) if _claim_ci(r) and _ci_shows_failure(r)",
     "    rows = [r for r in _read_rows(root) if _claim_ci(r)", "packet"),
    ("Q takes another packet's sources", "tpc_packet.py",
     "            and f\"{(r.get('repo') or '').lower()}#{int(r['number'])}\" not in ex]", "            and True]",
     "packet"),
    ("the decoys CLI ignores other packets' sources", "tpc_packet.py",
     '        out["other_sources"] += [(e.get("source") or {}).get("pr") for e in _json(p)["decoys"]',
     '        out["other_sources"] += [] and [(e.get("source") or {}).get("pr") for e in _json(p)["decoys"]', "packet"),
    ("a git error reads as outside any working tree", "tpc_packet.py", "    if marked is not None:", "    if False:",
     "packet"),
    # the seal guard reads main from the pinned URL (finding 8; R1 to R4 are the review's mutants)
    ("R1 git_on_main returns True", "tpc_seal.py", "    return r.returncode == 0\n\n\ndef git_signing_commit",
     "    return True\n\n\ndef git_signing_commit", "seal_store"),
    ("R2 git_fetch returns ok", "tpc_seal.py", "def git_fetch(repo: Path, url: str = PINNED_URL) -> str:",
     'def git_fetch(repo: Path, url: str = PINNED_URL) -> str:\n    return "ok"', "seal_store"),
    ("R3 ancestry checked against HEAD", "tpc_seal.py",
     '        r = _git(repo, ["merge-base", "--is-ancestor", commit, sha])',
     '        r = _git(repo, ["merge-base", "--is-ancestor", commit, "HEAD"])', "seal_store"),
    ("R4 blob read from HEAD", "tpc_seal.py", '        r = _git(repo, ["show", f"{commit}:{path}"])',
     '        r = _git(repo, ["show", f"HEAD:{path}"])', "seal_store"),
    ("the fetch follows origin's refspec", "tpc_seal.py",
     '        r = _git(repo, ["fetch", "--quiet", "--no-tags", url, f"+refs/heads/main:{SEAL_REF}"], timeout=180)',
     '        r = _git(repo, ["fetch", "--quiet", "origin"], timeout=180)', "seal_store"),
    ("main read by a short name", "tpc_seal.py",
     '        r = _git(repo, ["show-ref", "--verify", "--hash", SEAL_REF])',
     '        r = _git(repo, ["rev-parse", "origin/main"])', "seal_store"),
    ("origin's URL unchecked", "tpc_seal.py", "        if r.returncode != 0 or norm_url(got) != norm_url(url):",
     "        if False:", "seal_store"),
    ("a URL rewrite is followed", "tpc_seal.py",
     '        if any(k.endswith((".insteadof", ".pushinsteadof")) for k in keys):', "        if False:", "seal_store"),
    # the signing commit only, and the PREREG never changed after it (finding 9)
    ("a later commit opens the guard", "tpc_seal.py", "        if signing != commit:", "        if False:", "seal_store"),
    ("the PREREG may change after signing", "tpc_seal.py", "        if tip != blob:", "        if False:", "seal_store"),
    ("the signing commit is the latest signed one", "tpc_seal.py",
     '        r = _git(repo, ["log", "--reverse", "--format=%H", sha, "--", path], timeout=120)',
     '        r = _git(repo, ["log", "--format=%H", sha, "--", path], timeout=120)', "seal_store"),
    # the committed digests (finding 10) and the review's surviving mutants R10, R17, R18, R29, R31, R32 (finding 11)
    ("the committed digests unchecked", "tpc_score.py",
     '            if probs:\n                raise ScoreError(f"the committed digests refuse',
     '            if False:\n                raise ScoreError(f"the committed digests refuse', "score"),
    ("the key line need not be committed", "tpc_score.py", "    elif not any(_holds(o, want) for o in lines):",
     "    elif False:", "score"),
    ("answer digests need not be committed", "tpc_score.py",
     "        if lines is None or not any(_holds(o, w) for o in lines):", "        if False:", "score"),
    ("a seat may answer before the key digest is public", "tpc_score.py",
     "                if fa is None or not kp < parse_utc(fa):", "                if False:", "score"),
    ("seats read an uncommitted packet", "tpc_seats.py",
     '    if not (published or key_committed)(pk["packet"], pk_sha):', "    if False:", "seats"),
    ("R10 the packet's decoy count unchecked", "tpc_score.py", "        if n != len(decoy_set):", "        if False:",
     "score"),
    ("R17 cmd_p2 accepts an unfilled P2 lock row", "tpc_packet.py", "        if lock is None:", "        if False:",
     "packet"),
    ("R18 the scorer loads P2's set without the lock row", "tpc_score.py",
     "                                          lock_digest=None if (smoke or dry_run) else lock)",
     "                                          lock_digest=None)", "score"),
    ("R29 only the side compared between key and set", "tpc_score.py",
     '            for f in ("side", "construction", "label"):', '            for f in ("side",):', "score"),
    ("R31 ledger rows for other packet bytes accepted", "tpc_score.py",
     '        if other:\n            raise ScoreError(f"{lp.name} answers other packet bytes',
     '        if False:\n            raise ScoreError(f"{lp.name} answers other packet bytes', "score"),
    ("R32 a stand-in packet scored for real", "tpc_score.py",
     '    if public.get("beacon", {}).get("kind") == "stand_in" and not dry_run:', "    if False:", "score"),
]


def main() -> int:
    scratch = pathlib.Path(sys.argv[1]).resolve()
    targets = sys.argv[2:]
    base = scratch / "mut"
    work = base / "tests-pass-ci" / "collect"
    missed = 0
    for rule in M:
        name, f, old, new = rule[:4]
        hint = rule[4] if len(rule) > 4 else None
        if base.exists():
            shutil.rmtree(base)
        shutil.copytree(SRC, work, ignore=shutil.ignore_patterns("__pycache__", "checks"))
        for prereg in SRC.parent.glob("PREREG_*.md"):
            shutil.copy2(prereg, work.parent / prereg.name)
        p = work / f
        s = p.read_bytes().decode("utf-8")
        assert s.count(old) == 1, (name, s.count(old))
        p.write_bytes(s.replace(old, new).encode("utf-8"))
        env = dict(os.environ, TPC_COLLECT_DIR=str(work))
        if targets:
            ordered = sorted(targets, key=lambda t: 0 if hint and hint in t else 1)
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + ordered
            cwd = None
        else:
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"]
            cwd = work
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        missed += int(r.returncode == 0)
        print(("CAUGHT " if r.returncode else "MISSED ") + name + " :: " + last, flush=True)
    print(f"{len(M) - missed} of {len(M)} caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
