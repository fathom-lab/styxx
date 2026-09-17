# diffgate as a pre-commit hook — the commit message cannot lie about the staged diff

Four lines in your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/fathom-lab/styxx
    rev: <a release tag or commit that contains .pre-commit-hooks.yaml>
    hooks:
      - id: diffgate-commit-msg
```

then

```
pre-commit install --hook-type commit-msg
```

pre-commit builds styxx into its own environment from this repository at `rev`; nothing is
installed into yours. From then on every `git commit` reads the message you or your agent wrote
against `git diff --cached` and refuses the commit on a CONTRADICTED claim. This is what it
looks like when the message lies (real output, verbatim — pre-commit strips the first line's indent; a
temporary repository with one modified file and one new test):

```
$ git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."
styxx diffgate (the commit message vs the staged diff).......................Failed
- hook id: diffgate-commit-msg
- exit code: 1

[ok ] file_touched         diff status 'M' for 'src/retry.py'
  [LIE] tests_added          diff adds 1 test functions, claim says 3
  [LIE] only_touches         paths outside 'src': ['tests/test_retry.py']
styxx diffgate: FAIL — 2 claim(s) in the message contradict the staged diff. fix the message or the diff; `git commit --no-verify` overrides.
```

and when it does not:

```
$ git commit -m "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed."
styxx diffgate (the commit message vs the staged diff).......................Passed
```

A message with no diff-shaped claim passes on scope, silently. Git's own `#` commentary in the
template is not read as prose.

## What it judges, and what it will not

The closed template set of `styxx.diffgate`, nothing else: touched / created / deleted paths,
"N files changed", "added N tests", "adds function X", "only touches prefix/", "tests pass".
Every other sentence is never read, and the PASS line says how many. "tests pass" stays
UNCHECKABLE on purpose: a hook that ran your suite would be a second place for the agent's shell
to write the verdict it wants. A path the diff does not show is UNCHECKABLE, not an accusation
(EXTERNAL-1: that accusation measured precision 0.23 against a 0.95 floor on 71,016
agent-authored PRs and is withheld until a held-out repair clears the floor). One class of
false VERIFIED is known and open as #101: a changed `def` line counts as an added definition,
so "adds function X" and "added N tests" can verify against a signature edit or a re-indent.

## Rung

Weak, and said so: this hook shares a filesystem and a shell with the agent that wrote the
message. `git commit --no-verify` skips it, an agent can edit `.pre-commit-config.yaml`, and
pre-commit hooks are installed per clone by the person, never by the diff. It is the habit, not
the enforcement. The enforcement is the same gate run where the agent's shell cannot reach: the
GitHub Action (`uses: fathom-lab/styxx@main`, checkout-free, reads the PR body and diff from the
API) and the CLI on a checkout you control.

## The same gate, other doors

`integrations/git/commit-msg` is this hook as a single file for people who do not use
pre-commit (`cp` it into `.git/hooks/`). `integrations/claude-code/diffgate-hook` blocks the
`git commit` / `gh pr create` command itself inside Claude Code, before it runs. The console
script this hook runs is `styxx-diffgate-commit-msg COMMIT_MSG_FILE`, in `styxx/diffgate_hook.py`.
