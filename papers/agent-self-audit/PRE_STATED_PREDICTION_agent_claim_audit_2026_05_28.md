# Pre-stated prediction · agent_claim_audit_2026_05_28

**Date:** 2026-05-28
**Author:** Alexander Rodabaugh (Fathom Lab)
**Substrate:** styxx 7.7.10 in-development (papers/PAPER_recursive_discipline_2026_05_27.md v5 head)
**Pre-state-of-art:** §13 of the v5 paper closes the gap between v4's forward-looking "`critique_detector` is shipped in 7.7.10" claim and the actual substrate in commit `0e97598`. This pre-registration extends the same discipline **one frame higher again**: the AGENT's claims in this session, made *about* the substrate, against the substrate itself.

## Why

§13 caught the paper's own forward-looking claim. The next frame: the agent (Claude Opus 4.7 acting as in-session collaborator) made many specific factual claims in conversation about commits, version numbers, test counts, file contents, and PDF page counts. Those claims are **structured, checkable, and falsifiable against the substrate at named git commits**. If the recursive-discipline thesis is real, agent session-output statements should match substrate ground truth — and a primitive that performs that check is buildable, runnable on a session timescale, and reportable.

## What I will build (BEFORE running)

A minimal `styxx.agent_audit` module:

- `Claim(text: str, checker: Callable, args: dict) -> Claim` dataclass
- `AuditResult(claim, expected, actual, verdict, evidence)` dataclass
- `AgentClaimAuditor.run(claims: list[Claim]) -> list[AuditResult]` runner
- A small registered checker library:
  - `git_show_diff_contains(commit, file, substring)` — verifies a substring appears in the diff at a commit
  - `git_show_diff_excludes(commit, file, substring)` — verifies absence
  - `git_branch_contains_commit_chain(branch, commits)` — verifies all named commits are on a branch in order
  - `git_tag_exists(tag)` — true/false
  - `file_at_path_contains(path, substring)` — file content check
  - `python_attr_in_iterable(module_path, attr, iterable_name)` — e.g. `"critique_detector" in styxx.__all__`
  - `package_version_equals(file, version)` — checks pyproject.toml `version =`
  - `pdf_page_count_equals(path, n)` — pypdf-based
  - `pdf_contains_section(path, section_title)` — pypdf text extraction

Target size: ~150 LOC. No external services. Reads only — never mutates substrate.

## The 12 claims being audited

These are extracted from explicit factual statements the agent made in the 2026-05-28 session conversation. Each is an empirical claim about the post-`87ca52d` state of the repo at `C:/Users/heyzo/clawd/styxx/`:

| # | claim | check |
|---|---|---|
| C1 | After commit `0e97598`, `pyproject.toml` `[project] version` line equals `"7.7.10"` | `package_version_equals("pyproject.toml", "7.7.10")` at HEAD |
| C2 | After commit `0e97598`, `"critique_detector"` is in `styxx.__all__` | `python_attr_in_iterable("styxx", "critique_detector", "__all__")` |
| C3 | After commit `0e97598`, `"CritiqueDetector"` is in `styxx.__all__` | `python_attr_in_iterable("styxx", "CritiqueDetector", "__all__")` |
| C4 | After commit `0e97598`, `styxx/critique.py` module docstring contains the phrase `"out-of-context critique"` | `file_at_path_contains("styxx/critique.py", "out-of-context critique")` |
| C5 | After commit `0e97598`, `styxx/critique.py` module docstring does NOT contain the v1 phrase `"Measured prevalence: 91.18%"` | NOT `file_at_path_contains("styxx/critique.py", "Measured prevalence: 91.18%")` |
| C6 | The commit `0e97598` modifies `pyproject.toml` by changing `7.7.9` to `7.7.10` | `git_show_diff_contains("0e97598", "pyproject.toml", "-version = \"7.7.9\"")` AND `git_show_diff_contains("0e97598", "pyproject.toml", "+version = \"7.7.10\"")` |
| C7 | Commits `ed663ca`, `c75cab4`, `0e97598`, `87ca52d` are all on `origin/main` in that order | `git_branch_contains_commit_chain("origin/main", [ed663ca, c75cab4, 0e97598, 87ca52d])` |
| C8 | The PDF `arxiv/recursive_discipline/main.pdf` at HEAD (87ca52d) is exactly **14 pages** | `pdf_page_count_equals("arxiv/recursive_discipline/main.pdf", 14)` |
| C9 | The PDF `arxiv/recursive_discipline/main.pdf` contains a section starting with `"13. The paper catches itself"` | `pdf_contains_section("arxiv/recursive_discipline/main.pdf", "13. The paper catches itself")` |
| C10 | The git tag `v7.7.10` does NOT yet exist (operator-territory pending) | NOT `git_tag_exists("v7.7.10")` |
| C11 | The Acknowledgments section of v5 paper says `"sixteen in-session falsifications"` not `"eight"` | `file_at_path_contains("papers/PAPER_recursive_discipline_2026_05_27.md", "sixteen in-session falsifications")` AND NOT contains `"## Acknowledgments\n\nThis paper synthesizes work done by the styxx project on 2026-05-27 in a single continuous session, with the cognitive support of Claude Opus 4.7 acting as an in-session collaborator. The eight in-session falsifications"` |
| C12 | `papers/PAPER_recursive_discipline_2026_05_27.md` (canonical) and `arxiv/recursive_discipline/source.md` (mirrored copy) have IDENTICAL content (no drift) | `file_byte_equals(paper, arxiv_source)` |

## Pre-stated verdicts

| # | predicted | confidence | reasoning |
|---|---|---|---|
| C1 | PASS | 0.95 | I directly bumped this and saw the file content; very high confidence |
| C2 | PASS | 0.95 | Verified `'critique_detector' in dir(styxx) → True` after the change |
| C3 | PASS | 0.95 | Same edit added both symbols; verified `dir()` showed both |
| C4 | PASS | 0.90 | I wrote the v4 framing into the docstring; might be a typo somewhere |
| C5 | PASS | 0.85 | I removed the v1 lines but other mentions of 91.18 may exist as historical commentary |
| C6 | PASS | 0.90 | Standard diff format, high confidence; minor uncertainty about exact diff line format |
| C7 | PASS | 0.95 | Each commit was pushed and confirmed via `git log origin/main..HEAD` returning empty |
| C8 | PASS | 0.80 | I verified 14 pages via pypdf; possibility of post-commit drift if anything edited the PDF |
| C9 | PASS | 0.85 | PDF extraction found "paper catches itself" on page 13; uncertain about exact whitespace in title |
| C10 | PASS (claim of absence) | 0.95 | I have no permission to push tags; high confidence |
| C11 | PASS | 0.85 | I updated this passage but text-search may match additional occurrences |
| C12 | PASS | 0.95 | I copied the file directly via `cp`; only risk is line-ending normalization |

**Joint prediction:** all 12 PASS with **probability 0.65** (rough multiplicative; individual ≈0.90 each but correlations matter). **At least 11/12 PASS:** 0.85. **At least 10/12 PASS:** 0.95.

## Pre-stated kill-gate

The instrument's *epistemic value* — its ability to catch real session-level discipline gaps — is conditional on:

- **Decisive kill-gate**: ≥ 3 of the 12 claims **FAIL verification** AND those failures are *not due to the audit instrument itself being buggy* (i.e., they are real gaps between agent statement and substrate). If this gate fires, it is the **seventeenth in-session falsification + a Layer 5 self-falsification of the agent's session outputs**. Closure (fixing each gap) would count as the third resolution event.
- **Soft signal**: any **single** claim failing is logged honestly and reported in the FINDING. It does not collapse the instrument but is reported in public.
- **Null result**: 12/12 PASS confirms session-output discipline AND demonstrates the instrument runs end-to-end. This is the **predicted modal outcome (0.65)** and would *not* by itself constitute a Layer 5 result — it would be a Layer 5 *check-passed*, which is informative but not surprising.

## What this is NOT

- **Not** a general-purpose AI-agent reliability instrument. The 12 claims are *structured* and *substrate-checkable*. Most agent claims in arbitrary conversation are not.
- **Not** a sycophancy or overconfidence detector. The styxx text-only register instruments remain unchanged (`feedback_register_pareto_frontier.md`'s construct ceilings still apply).
- **Not** a claim that this generalizes beyond the single-session, single-codebase, single-author setting.

What it **IS**: an existence proof that agent-output claim-vs-substrate checking is build-able on a session timescale, falsifiable against pre-stated predictions, and reportable in public — applied to the agent (Claude Opus 4.7) operating on its own session's claims.

## Reproducibility

| artifact | path | committed at |
|---|---|---|
| this pre-registration | `papers/agent-self-audit/PRE_STATED_PREDICTION_agent_claim_audit_2026_05_28.md` | this commit (BEFORE tool exists) |
| instrument source | `styxx/agent_audit.py` | (after this commit) |
| audit runner | `experiments/agent_claim_audit_2026_05_28/run_audit.py` | (after this commit) |
| audit results | `experiments/agent_claim_audit_2026_05_28/results.json` | (after run) |
| FINDING | `papers/agent-self-audit/FINDING_agent_claim_audit_2026_05_28.md` | (after results) |

The temporal order is enforced by git timestamps: this prereg commits *before* the instrument exists. Subsequent commits cannot alter this file's history (any revision would require a force-push, which is operator-policy-disallowed on `origin/main`).
