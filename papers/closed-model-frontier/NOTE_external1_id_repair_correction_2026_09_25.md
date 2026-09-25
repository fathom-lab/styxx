# Note — two corrections to NOTE_external1_id_repair_2026_09_25.md

2026-09-25. `NOTE_external1_id_repair_2026_09_25.md` is committed and is not edited. Two of its
sentences are corrected here, and one thing it did not cover is recorded. This note is committed
alone, ahead of the commit that changes `external1_packet.py` and its tests to make the
corrections true. `external1_packet.json`, `external1_key_SEALED.json` and
`external1_key_digest.txt` are untouched by both.

## "which `build --as-published` still reproduces exactly"

That held on Linux and not on Windows. `build` wrote its three outputs with `Path.write_text`,
which writes the platform's line separator. Git stores `external1_packet.json` and
`external1_key_digest.txt` with LF (`git ls-files --eol` reports `i/lf`; the packet's blob is
378,104 bytes and contains no CR, while its CRLF checkout on Windows is 392,352). On Windows the
rebuilt files carried CRLF, so they matched the committed blobs line for line, not byte for
byte. The test that pins the bytes `--as-published` writes had been computed on Windows, so it
pinned that CRLF rendering, which a build on Linux — every CI job — does not write.

In the commit that follows this note, `build` writes LF on every platform. The pin is re-derived
from LF bytes: the exact text the pre-repair builder (the version of `external1_packet.py` that
built the published packet) passed to `Path.write_text`, which is what it wrote on Linux, on a
synthetic corpus that now carries a non-ASCII character as the committed packet does. The
repaired `build --as-published` was checked to write those bytes exactly. The exact-bytes guard
reads a CRLF checkout of the record as the record, and no other difference. From that commit the
sentence holds as written on both platforms.

## "a test is not what holds it — the module's own layout is"

From the commit that follows this note, a test does hold it:
`tests/test_external1_packet_ids.py::test_the_cited_sampling_line_is_still_line_63` asserts that
line 63 of `external1_packet.py` is `sample_acc = rng.sample(acc, N_ACC)` and that line 45 of
`ANALYSIS_base_rate_ceiling_2026_09_01.md` still cites it, and its failure message names that
citation. The rest of the paragraph stands: if the line moves, the honest fix is a new note
naming the new line, never an edit to the ANALYSIS.

## Not in the earlier note: the fresh-clone refusal

The round that wrote the earlier note also made `build` refuse, in both modes, the state every
clone is in: the packet and the digest committed, the sealed key gitignored and absent. That
refusal blocked `build --as-published`, the command the module's RECIPE prescribes for
regenerating the record, from exactly that state. In the commit that follows, only a plain build
refuses there. `--as-published` is held by the exact-bytes check instead: it writes nothing
unless the packet and the digest it builds are the ones on disk, and the digest is the key's
salted SHA-256, so when they match, the key it writes is the sealed key.
