<!--
This is the styxx gauntlet submission template. Use it when submitting a
new method to the public-challenge leaderboard. See submissions/GAUNTLET.md
for the full protocol.

For a non-gauntlet PR (bug fix, doc change, feature), delete this template
and write your own description.
-->

## Submission summary

- **Name:** *(short display name for the leaderboard, e.g., `My-Method-v1`)*
- **Submitter:** *(your name / affiliation / GitHub handle)*
- **Task:** classification | detection
- **Method spec:** `submissions.<your-dir>.method:<predict|detect>`
- **Code:** *(link to your repo or branch where the method is developed)*

## Reported gauntlet result

Paste the **JSON output** from your local `styxx gauntlet` run here (or attach as `submissions/<your-name>/_local_result.json`). The CI workflow will re-run the gauntlet and compare scores against your `submission.json`'s `reported` block (1e-3 float tolerance for floats).

<details><summary>local gauntlet output</summary>

```json
PASTE the result.json from `styxx gauntlet ... --format json > result.json` here
```

</details>

## Submission checklist

- [ ] My method's directory exists at `submissions/<my-name>/`.
- [ ] `submissions/<my-name>/method.py` exposes either `predict(question: str) -> dict` (classification) or `detect(question: str, response: str) -> dict` (detection).
- [ ] `submissions/<my-name>/submission.json` includes `name`, `submitter`, `method`, `task`, `description`, `code_url`, and `reported` (with `overall_pass`, `n_passed`, `n_total_bars`, `metrics`, `bar_results`).
- [ ] If my method requires extra Python dependencies, they are listed in `submissions/<my-name>/requirements.txt`.
- [ ] I ran `styxx gauntlet --method submissions.<my-name>.method:<fn> --task <task> --format json` locally and the reported numbers in `submission.json` match what the CI will produce within 1e-3 float tolerance.
- [ ] I have **NOT** modified `papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json`, `styxx/_data/`, the bars in `styxx/gauntlet.py`, or any pre-registration document.
- [ ] My method's training data does not include benchmark items (honor system — independent papers may catch contamination).

## About this method

*(a paragraph or two describing the approach, intended audience, and what you expect the result to mean. include caveats / scope_bound if your method has known failure modes — the project values honest scope statements over claimed wins.)*

## Citation

*(if you want a specific citation format for the leaderboard row, propose it here. otherwise we use a default `<Submitter> (<year>). <Name>. styxx gauntlet leaderboard, <PR-link>.`)*

---

CI will re-run `styxx gauntlet` against your method on the bundled benchmark JSON and verify your reported scores match. A maintainer reviews + decides leaderboard placement after CI passes. See `submissions/GAUNTLET.md` for the full protocol.
