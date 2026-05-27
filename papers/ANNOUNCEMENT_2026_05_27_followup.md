# Announcement follow-up thread — styxx 7.7.7 + gauntlet + Zenodo DOI

**Channel:** @fathom_lab. Format: standalone or reply-thread off the 7.7.3 thread — both work. Voice: same as `ANNOUNCEMENT_2026_05_27.md` (lowercase, honest, scoped, no hype).

**Permanent citation:** [10.5281/zenodo.20418532](https://doi.org/10.5281/zenodo.20418532)
**Release:** https://github.com/fathom-lab/styxx/releases/tag/v7.7.7
**Leaderboard:** https://github.com/fathom-lab/styxx/blob/main/LEADERBOARD.md

---

**1/**
```
follow-up to last week's 7.7.3 thread: the seven-method empirical floor is no longer just a paper.

it's a pip-installable public challenge with concrete numeric baselines, CI auto-verification on submissions, and a permanent academic DOI.

styxx 7.7.7 is live ⤵️
```

---

**2/**
```
styxx gauntlet — run any detection or classification method against the labeled benchmark.

we tested seven methods on the dark core. all seven closed-negative.
those bars are now public. beat them or join the floor.

pip install styxx==7.7.7
styxx gauntlet --method <module:attr>
```

---

**3/**
```
styxx leaderboard — see the current floor from the terminal.

the board has 4 concrete reference rows already:
• Baseline-001 (the seven-method floor)
• Baseline-002 (our classifier, 1/3 bars)
• Baseline-003 (length heuristic, 0/3 — anchors the bottom)
• Baseline-004 (random class, ~chance)
```

---

**4/**
```
the submission protocol is on origin.

fork → write a method.py → run styxx gauntlet → open a PR.

CI re-runs your method against the bundled benchmark, compares scores to what you reported (1e-3 float tolerance). mismatches fail the PR.

the leaderboard is trustworthy by construction.
```

---

**5/**
```
styxx critique — audit + register-fix suggestions, every suggestion carries a mandatory scope_bound naming its documented limit.

the tool can't ship a register rule without acknowledging where that rule doesn't apply. test-enforced. the discipline pattern as code.
```

---

**6/**
```
the work is now permanently citable.

zenodo DOI: 10.5281/zenodo.20418532
concept (always-latest): 10.5281/zenodo.19326174

v24 in the chain. predecessor v23 (10.5281/zenodo.20130041) is preserved.

the credibility compounds across versions, not within any one.
```

---

**7/**
```
the bar:

K1 folklore F1 ≥ 0.70 (in-distribution)
K2 4-way accuracy ≥ 0.65
K3 cross-corpus folklore F1 ≥ 0.60 (the load-bearing test)

clean install in 60 seconds:
pip install styxx==7.7.7
styxx leaderboard --rows-only

submit your method. the synthesis gets revised if you beat us.
```

---

## Notes for operator

- 7 tweets total. Each verified ≤280 chars X-effective (URLs auto-shorten to 23).
- **Tweet 1** explicitly references the prior 7.7.3 thread — works as either a standalone announcement OR a reply-thread.
- **Tweet 6** leads with the citation block — the Zenodo DOI is the durable academic anchor; surfacing it explicitly invites researchers who'd want to cite.
- **Tweet 7** closes with the install one-liner + the explicit floor-revising frame.
- Optional 8th tweet pinning a specific link (paper, LEADERBOARD, or the v24 Zenodo page) — operator's call.

## What this thread does NOT do

- Repeat the seven-method arc results from the original 7.7.3 thread (anyone reading this will click back through). The new thread is *what's new since*, not a fresh primer.
- Promise reach, replication, or "phenomenal."
- Use hype-class adjectives.

## Optional addition: visual asset

The cognometric card at `papers/agent-self-audit/cognometric-card-claude-2026-05-27.png` could attach to tweet 1 for visual weight; or a screenshot of `styxx leaderboard --rows-only` showing the four reference baselines could attach to tweet 3. Operator's call.
