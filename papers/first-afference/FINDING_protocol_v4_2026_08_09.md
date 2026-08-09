# FINDING — protocol v4: the composition check ships, and its red team found holes in the foundation every prior verdict stood on

Fathom Lab · 2026-08-09 · prereg: `PREREG_protocol_v4_composition_2026_08_09.md` (frozen before
the implementation existed) · receipts: `protocol_v4_result.json`,
`protocol_v4_redteam_audit.json` · scored by `styxx.protocol`.

## Verdict

The exam returned **`PROCEED_TO_RED_TEAM__not_yet_shippable`**, and the red team — after **four
rounds**, two of which broke the fixes of the round before — returned **FIXES HOLD, shippable**.
v4 ships in 7.35.0. This is the first instrument in this program to clear the full pre-release
gauntlet since the rule was adopted: three modules were stopped by it; this one survived it.

## What v4 is

E1 (cycle 159) produced a gate that passed on a value belonging to a candidate another gate had
disqualified in the same run — every component individually correct, the composition wrong, and
nothing in the machinery looking at relationships between gates. v4 makes the composition
declarable and machine-checked: a gate judging an aggregate over a set declares `agg`, `over`,
and optionally `excluding` in the frozen gates block, and `score()` recomputes the aggregate over
the declared population minus the declared exclusions, refusing on mismatch. **The E1 retro-case
is the proof**: E1's own G1, rewritten with a v4 declaration and scored against the committed
`e1_result.json`, refuses with the violation named — the metric quotes 0.1436, the declared
minimum over eligible candidates recomputes to 0.2554.

Its limit is stated in the prereg and repeated here: this checks **declared** composition against
the receipt's own internal consistency. An author who declares nothing, or a runner that forges
the declared fields, is not caught. A ratchet, not a proof.

## What the red team actually found, which is bigger than v4

The two ship-blocking defects in round 1 were not in the new mechanic. They were in the
gates-block parser **every protocol version since v1 has shared**: the fence regex took the first
match anywhere in the file, so a gates block hidden in an HTML comment — invisible in rendered
markdown — could shadow the visible one; and `json.loads` silently keeps the last duplicate key,
so a block could declare `"excluding"` twice, show a reader both, and honour only the decoy.

**A frozen document that can show a reader one thing and score another is not frozen.** Every
committed verdict in this corpus had depended on a parser with those holes. The corpus was
measured clean — zero multi-fence preregs, zero duplicate keys, zero non-ASCII keys or fence
info strings, across every committed gates block — so no history is affected and no fix rewrote
any of it. But the class was real, reachable, and old.

## Four rounds, two broken fixes, one structural lesson

- **Round 1** found the class (hidden fence, duplicate key). The fix counted exact-lowercase
  backtick fences and rejected byte-equal duplicate keys.
- **Round 2 broke it**: a `~~~gates` fence — identical to ```` ```gates ```` in any renderer —
  carried the honest declaration while a hidden backtick block was the machine's single match;
  and a homoglyph key (`exсluding`, Cyrillic с) slipped byte-equality to make two live
  declarations. The fix normalized fence info strings (case, format characters) and refused
  non-ASCII keys.
- **Round 3 broke it again with one character**: Cyrillic `а` in `gаtes` — a letter, not a
  format character, untouched by casefolding — made a visible honest-looking block invisible to
  the counter while the ASCII decoy scored. The diagnosis was structural: **two scanners with
  two definitions of "a gates fence," where every divergence is a shadowing channel.**
- **Round 4 verified the rewrite**: one line-based tokenizer counts, validates, and extracts
  from the same match; non-ASCII fence info refuses as a class ("no normalization table is
  trusted to enumerate it"); comment spans include unterminated comments; the closing fence is
  required. Every prior vector refuses. Zero unsafe regressions. Extraction byte-identical to
  frozen history.

The lesson generalises past parsers: **patching the instance an adversary shows you leaves the
class open, and the class is only closed when the two authorities — what a reader sees and what
the machine executes — are computed by one procedure.** It took an adversary two broken fixes to
force that design, which is exactly why the rule requires the adversary.

## What ships and what is documented

7.35.0 ships protocol v4 with: composition declarations (`agg`/`over`/`excluding`),
`check_metrics` resolving composition paths pre-run, the single-scanner gates-block parser, the
non-ASCII refusals, typed refusals for malformed exclusion lists, and `numbers.Real` accepted
consistently in both number guards. Twenty-seven regression tests pin every attack class from
all four rounds.

Documented fail-safe residuals (both only ever refuse, never mis-score): an unpaired `<!--`
inside a display code fence above the gates block false-refuses; non-plain fence forms
(`~~~gates`, case variants, extra backticks, 1–3-space indents) render as fences but refuse by
design, because the plain unindented lowercase form is the one form renderer and parser read
identically.

## Limits

The composition check cannot verify intent or provenance (D3, disclosed in the prereg): `over`
pointing at a decoy dict, or a forged exclusion list, scores. The residual defence is unchanged
and is the same one every layer of this program ends at — an adversary who reads the prereg
before the run. The four-round audit measured this lab's parser, not other markdown renderers'
exact fence semantics; the CommonMark rules used (3-space indent, tab as 4) are the spec's, and
divergences from real renderers, if any exist, are in the refusing direction.

*Frozen before the implementation existed; the exam's first run caught its own pairing bug and
disclosed it; two fixes broken by the verifier before the third held; shipped only on the
adversary's explicit verdict. Every number grounds in the two receipts.*
