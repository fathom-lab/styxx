# the size line of each pass-3 reader's prompt

Excerpts, verbatim, of the last instruction in each of the four prompts that launched the pass-3 readers on
2026-09-14. They are committed because `CORRECTION_sand_neighbours_pass3_2026_09_14.md` said "each reader was
told its files' sizes" and the lab's verification of the same night found nothing in the tree that showed it.
The full prompts are in the lab's session record, not in this repository; each also carried the frozen
protocol's pricing rule, the fingerprint clause's object and the return schema.

reader-1 (L01, L02, L08, L15):

> chars_read must be the number of characters you actually read (the file sizes are: L01 81839, L02 88365, L08 67659, L15 10133).

reader-2 (L03, L04, L05, L06):

> chars_read must be the number of characters you actually read (the file sizes are: L03 105414, L04 40686, L05 27508, L06 59992).

reader-3 (L07, L09, L10, L13):

> chars_read must be the number of characters you actually read (the file sizes are: L07 42248, L09 122531, L10 50459, L13 36617).

reader-4 (L11, L12, L14, L16, L17):

> For L16 and L17 the "elements" object may be omitted. chars_read must be the number of characters you actually read (the file sizes are: L11 80213, L12 95326, L14 86729, L16 24162, L17 15981).

Every reader returned exactly the sizes it was given (`sand_prior_art_survey_pass3_correction_2026_09_14.json`,
`self_reports`). A reported count equal to the count one was told is not a measurement of reading.
