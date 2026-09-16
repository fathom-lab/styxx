# git hook — the commit message cannot lie about the staged diff

One file, `commit-msg`, that runs `styxx.diffgate` on the message against `git diff --cached`
before every commit lands. A message that says "added 3 tests" over a diff that adds one is
refused with the line named; a message with no diff-shaped claim passes on scope and says so.

```
pip install styxx
cp integrations/git/commit-msg .git/hooks/commit-msg && chmod +x .git/hooks/commit-msg
```

```
$ git commit -m "Refactored src/retry.py. Added 3 tests. Only touches files under src/."
  [ok ] file_touched         diff status 'M' for 'src/retry.py'
  [LIE] tests_added          diff adds 1 test functions, claim says 3
  [LIE] only_touches         paths outside 'src': ['tests/test_retry.py']
styxx diffgate: FAIL — 2 claim(s) in the message contradict the staged diff. fix the message or the diff; `git commit --no-verify` overrides.
```

What it reads is the same closed template set as the CLI and the GitHub Action, and what it will
not do is the same: a path the diff does not show is UNCHECKABLE, not an accusation (EXTERNAL-1,
see the README). "tests pass" is UNCHECKABLE in a hook by design — running the suite here would
hand the agent's shell a second place to write the verdict it wants.

Pairs with the "write the commit message before the code" habit: the message becomes a
preregistration, and this is the half that checks whether it came true.

Git never installs hooks from a checkout, so this is opt-in per clone. For a whole team, point
`core.hooksPath` at a directory that holds it.
