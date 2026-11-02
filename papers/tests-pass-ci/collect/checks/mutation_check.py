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
answer the open findings of 2026-11-02 (dates, step flips, the shuffle, the seal guard, the signed PREREG, the salts).
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
    ("shuffle ignores the seed", "tpc_packet.py", "    rng = random.Random(shuffle_seed(seed_hex, packet))",
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
    ("dates kept in the items", "tpc_items.py", "    return DATES.sub(DATE_TOKEN, text)", "    return text"),
    ("a decoy may carry a date", "tpc_decoys.py", "        if DATES.search(text):", "        if False:"),
]


def main() -> int:
    scratch = pathlib.Path(sys.argv[1]).resolve()
    targets = sys.argv[2:]
    base = scratch / "mut"
    work = base / "tests-pass-ci" / "collect"
    missed = 0
    for name, f, old, new in M:
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
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"] + targets
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
