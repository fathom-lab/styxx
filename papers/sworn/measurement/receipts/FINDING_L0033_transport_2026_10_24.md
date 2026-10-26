# Finding for the tranche A RESULT: item L-0033 fails at the Claude transport in all three seats (2026-10-24)

- What: L-seat1, L-seat2 and L-seat3 of the Claude family (claude-opus-5-5, Amendment 6) each record one error, on
  item L-0033: "the transport's bytes were not the JSON envelope the transport promises" (after the runner's two
  retries). Every other Claude item, and every local item, has no error.
- Cause (measured): for packet L, `seat_claude.py` `prompt_for` passes the item text verbatim as the value of `-p`
  (`return item["text"]`), and L-0033 is the only item of either packet whose text starts with "-"
  ("- The read≠write submission source at ..."). The claude CLI reads an argument that starts with "-" as an option,
  so the call returns no JSON envelope. Length is not the cause (L-0033 is 2,199 JSON chars; L-0008, 2,918, passed).
- Tranche A: no re-run (the other seats' answers exist; a re-run now is a post-data change). The RESULT prints the
  three errors, this cause, and scores L-0033's Claude family by the PREREG's own rule for an item with errors.
- Tranche B: owed before its data, a dated AMENDMENT that routes a prompt starting with "-" safely (e.g. the item
  passed after `--`, or through stdin) without editing the locked `seat_claude.py` (a route like Amendment 6's for the
  twins), with a test that a leading-dash item answers.
- Local family: L-seat1/2/3 leave 14, 14 and 16 of 59 items unparsed (the 3B model's L format), R 0; reported as
  measured.
