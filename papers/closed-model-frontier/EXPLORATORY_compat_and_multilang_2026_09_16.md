# EXPLORATORY — two questions asked of the EXTERNAL-1 corpus before any rule is frozen

Fathom Lab · 2026-09-16 · Not a measurement of the instrument; a census of the corpus, run once,
written down before `PREREG_compat1_2026_09_16.md` so the prereg cannot pretend it did not know.
Receipts: `exploratory_ml_compat.py`, `exploratory_ml_compat.json`. Counts only; no PR named.

## 1. Test counts outside Python are rare; counts of "test cases" are as common as counts of tests

Of 71,016 eligible PRs, 99 descriptions say "added N tests" or "added N test functions/methods";
85 of those diffs touch no Python. By language of the diff: C# 36, JS/TS 21, Python 14, Java 5,
Rust 4, Go 3, the rest scattered. A per-language definition pattern on the added lines
(`it(`/`test(`; `func Test`; `#[test]`; `@Test`; `[Fact]`/`[Test]`; `def test_`) equals the claimed
N in 31 of the 85 non-Python cases (C# 16 of 36, JS/TS 10 of 21, Python 5 of 14 by this
pattern). Agreement is not truth and disagreement is not a lie: a `describe` block with three
`it(` is three tests to one author and one to another. Another 100 descriptions count "test
cases", "test files" or "test scenarios", which BC-2 verifies only when the `def test_` count
happens to match. The multi-language counter is worth a preregistration of its own; it moves
under a hundred claims on this corpus and is not the lever.

## 2. One in eight PRs claims compatibility, and one in sixteen of those removes a public name

**8,467 of 71,016 descriptions (11.9%) carry a compatibility claim**: "backward compatibility"
4,834, "no breaking changes" 1,310, "backward compatible" 906, "no functional changes" 366,
"non-breaking" 320, "backwards compatibility" 267, "no behavioral changes" 82, "fully compatible"
58, "zero functional changes" 16, and variants. The gate reads none of them today; they are the
sentence a reviewer most wants checked and the sentence that failed loudest in public this week
(a pull request titled "zero behavior change" whose review found removed public names that
external plugins imported).

A first-cut mechanical reading — a top-level public definition present in the removed lines of a
file (Python `def`/`class` at column 0; JS/TS `export function|class|const|…`; Go exported `func`
or `type`; Rust `pub fn|struct|enum|trait`; Java `public … name(`) whose name does not appear
anywhere in the added lines of the same language — finds **531 of the 8,467 claims (6.3%) sitting
on a diff that drops at least one public name**: JS/TS 243, Python 155, Go 79, Java 40, Rust 34.
165 drop exactly one name, 81 two, 119 ten or more (whole-module removals). Whether each is a
breaking change is not decided here: a dropped name can be dead, private-by-convention,
re-exported from elsewhere, or deliberately removed with the claim being about wire behaviour.
That is why COMPAT-1 attaches the names as evidence and never accuses.

## What this means for the prereg that follows

The compatibility claim gets a template and a reading that is *evidence, not verdict*: the claim
is UNCHECKABLE either way, and the reason names the public definitions the diff removed and does
not re-define, with their files, so a reviewer checks in seconds what the description asserted in
one clause. Whether that evidence is precise enough to ever accuse is a later, blind measurement
under the EXTERNAL-1 protocol, not this cycle's.
