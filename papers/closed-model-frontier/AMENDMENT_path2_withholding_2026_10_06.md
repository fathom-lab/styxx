# AMENDMENT — PATH-2's repair is not what lands; withholding is. The operator's decision.

Fathom Lab · 2026-10-06 · Amends `PREREG_path2_resolution_2026_09_17.md`, which is committed on the
branch of #161 (`fix/diffgate-path-resolution`, commit `97e84502`) and is not on `main`, because #161
never merged. Written by the lead, on the operator's instruction of 2026-10-06, after the operator
read the state of #161 and #187 and chose among the options put to them. Committed alone, before the
merge of #187.

## What PREREG_path2 promised

A repair of three defects of the diff gate: #97 (a base-name match shadows an exact path), #121 (a
dotfile and its undotted twin share one path key) and #101 (a changed definition counts as an added
one). Its unit gate G-P1 names the repaired verdicts on the reproductions: VERIFIED where `main` is
wrong today. Its failure clause: a failure of G-P1 to G-P5 blocks the corpus run.

## What happened

- #161 wrote the repair. It went through 15 review passes and never met its merge bar, of which one
  condition, from pass six on, was no verdict worse than `main`'s. Its licences for #97 and #121 were
  withdrawn at pass fifteen. It stays open as the record of the attempt.
- #187 (PATH-2a) does not repair. It leaves `main`'s reader unchanged and withholds, as UNCHECKABLE,
  any decided verdict that one of the three mechanisms can have made wrong. Its pass twelve
  (`36de3501`) is the only one of its passes that all four of the lab's own reviews returned as ship.

## The decision

1. **G-P1 is not met, and this branch does not claim it.** No reproduction gets the repaired
   VERIFIED. #97, #101 and #121 stay open: this amendment does not mark them fixed.
2. **Withholding is accepted as what lands now.** The cost and coverage are those `web/gate/README.md`
   states at the head of #187, with what each figure measures and whether a test pins it. On the
   pinned known-answer world (1,706 cases), every one of the 1,206 false verdicts attributable to the
   three mechanisms is withheld, and 253 of 5,897 right verdicts are withheld with them. On decorated
   descriptions the share of right verdicts withheld is higher (the README gives it). The pinned
   table in `tests/test_diffgate_path2a_truth.py` gives 329 false verdicts on that world that the
   mechanisms do not explain; 93 of them are withheld, and 236 stand, as they do on `main`.
3. **The removal of the gate-agreement switch is confirmed.** It was the lead's decision of
   2026-10-04, carried out at pass nine, with the exchange figures the README gives: the command line
   and the bookmarklet now give different gate verdicts on some inputs where `main`'s two agree.
4. **The boundary of bar A is confirmed as the lead restated it at pass eleven.** The record leaves
   the overlay as `main`'s, but for abstentions in reach, for any rule code that uses what it is
   handed as data. Reflection is not covered: a class reached through `type()`, a function's globals,
   a frame, or a patched built-in. The README states that limit, and a test pins one such route as
   leaving the relation.
5. **The operator options not taken stand as disclosed**, O-1 to O-16. Among them are O-10 (the
   single-prefix family of git renderings, pinned as kept), O-13 (the five joint #121 reproductions)
   and O-16 (the description-side guards). They stay open for a later decision; none is taken by
   this amendment.

## What does not change

- PREREG_path2's corpus gates G-C1 to G-C5 were never run, and they are not run for #187.
- The path accusation stays withheld. Nothing here restores a licence #161 withdrew.
- No committed receipt, certificate, PREREG, RESULT, NOTE or charon log is edited. The preregistration
  itself is not edited; this file is the amendment.

## What this amendment does not say

That the three defects are fixed, or that withholding is as good as repair. Withholding is cheaper
to make safe. It is not the answer PREREG_path2 asked for. Repair remains the open goal.
