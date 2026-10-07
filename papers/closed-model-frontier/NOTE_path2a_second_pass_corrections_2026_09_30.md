# NOTE — PATH-2a, second pass: where the code departs from the pass-2 note, and why

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`. The pass-2 note
(`papers/closed-model-frontier/NOTE_path2a_second_pass_2026_09_30.md`, commit `d801d6d3`) was committed alone before
this pass's code, and is not edited. Writing the code showed four places where its design withheld more than it had to,
or still let the two ports part. **This note records each departure before the code is committed, and is committed
alone, ahead of it.** The figures quoted here as reasons were measured in scratch while the code was written; the
figures at the head that carries the code go into `web/gate/README.md` and the CHANGELOG entry.

Every departure keeps the abstain-only relation (A) as it was: each changes only which decided claims become
UNCHECKABLE, and with which phrase. None of them turns a claim the pass-2 design withheld into a false verdict that an
attribution variant would read right; the committed truth test is the check.

## 1. `tests_added`: each port's count is read exactly, instead of the "apart" guard

**The pass-2 note said:** when a counted test pairs with a removed definition and the two line views' added lines
differ, or an added line holding `def test_` holds a divergent character, abstain with `split`, in both ports.

**The code does:** it reads, from the bytes, the count each port's `main` makes. CPython's `main` counts
`^\s*def test_` over the added lines of line view 0 with CPython's `\s`; the port's counts it over view 1 with the port's
`\s`, and its `^` also holds after U+2028 and U+2029. Both white-space sets are constants in both blocks
(`_P2A_PY_SPACE`, `_P2A_JS_SPACE`), pinned by enumeration against `str.isspace`, CPython's regex `\s`, node's `\s` and
`trim()`. Per view the overlay counts the `def test_` sites that view's `main` counts, and how many of them name a test a
removed line may define. The count for its own port must equal `main`'s `got`, or the claim abstains as `unreproduced`.
Then, for each port whose count gives the claim the verdict it has here, PREREG R-101's interval
`[got - changed, got]` is read with that port's figures: `tests` when every such reading holds the claim, `split` when
only one does.

**Why.** The guard withheld 150 more claims than pass 1 on the committed inputs (136 CONTRADICTED, 14 VERIFIED),
among them right verdicts CPython's `main` read exactly: #161's f2 cases, where a vertical tab or form feed ahead of
`def test_a` is a line break to CPython and white space to the port, and "Added 0 tests." is VERIFIED in CPython and
true. Read exactly, `split` fires on 4 committed claims.

**Why the two ports still decide alike.** Both blocks compute the same two counts from the same bytes. Where the two
`main`s give a claim the same verdict, the set of ports read is the same, so the decision and the phrase are the same.
The lockstep tests pin, on every committed input and in both ports, that the view-0 count is CPython `main`'s count,
the view-1 count is the port `main`'s count, and the two blocks agree on both.

## 2. `extract` for a path holding a code point from 0x80 up: only where the port could verify its reading

**The pass-2 note said:** a path claim whose path holds such a code point abstains.

**The code does:** it abstains only when some registration with the claimed status (any, for a touched claim) has an
ASCII base name that the claimed path, folded and with `\` read as `/`, holds.

**Why.** The broad rule withheld 158 right verdicts on the committed truth world (all of them path claims whose path
holds a code point from 0x80 up, most of them in the family of `.énv.json` beside `énv.json`, which the port's
template reads as `nv.json`);
the narrowed rule withholds 76.

**Why the two ports still decide alike.** The port's template reads only ASCII, so its path is an ASCII piece of this
one, ending at an extension. `main` verifies a path only on an entry at tier 0, 1 or 2, and every tier implies equal
base names, so the entry's base name is the ASCII base name of the port's path, which this path holds. So wherever the
port's `main` verifies the claim, the condition holds here and the claim abstains in both. Where it does not, the two
`main`s give the claim different verdicts, which is outside (C).

## 3. `extract` reads the whole run of path characters around the occurrence

**The pass-2 note said:** an ASCII path or name abstains when an occurrence of it in the claim text is immediately
preceded or followed by a code point from 0x80 up.

**The code does:** the run of path characters around the occurrence (ASCII word characters, `.`, `/`, `-`, backslash,
and every code point from 0x80 up) must not hold such a code point.

**Why.** On the reviewer's calm fuzz, "README.md/straße.md" is read whole by CPython and as "README.md" by the port,
whose ASCII `\b` holds before the `/`. Nothing outside ASCII touches "README.md". That gave 7 verdict splits.

## 4. `divergent` is read before `extract` (path claims and `only_touches`)

**The pass-2 note said:** `extract` ahead of `divergent`.

**Why.** When a file header holds a divergent character, the two ports register different paths, so the test of
section 2 can differ between them. `divergent` is computed from the bytes alike in both ports, so reading it ahead of
`extract` makes both give the same phrase. One phrase split on the reviewer's hostile fuzz came from this.

## 5. Two facts the pass-2 note did not know

- **The git door.** None of #161's recorded `--name-status` outputs carries a rename or a copy; 16 carry type changes.
  Renames, a copy and a type change are therefore read at a real git door (`tests/test_diffgate_path2a.py`, repositories
  built with plumbing), where the reviewer's `parts[1]` plant is refused. The truth test at the git door reads the 96
  reproductions that carry their own `--name-status`.
- **The abstention pins depend on the interpreter's Unicode tables** as well as on the path flavour. `main`'s own reading
  of #161's Unicode 14 and 16 probes moves with them (on Unicode 16, `main` reads the name `fo` + U+105C0 whole and
  CONTRADICTS it). The pins are measured for Unicode 15.0 (CPython 3.12) and 16.0 (3.14), and for 13.0 and 14.0 (CI's 3.9
  to 3.11) by re-reading the inputs with their five Unicode 14 letters mapped to code points no version assigns. An
  interpreter with another Unicode version fails the test and names the version, rather than passing on a figure no one
  measured.

## 6. What does not change

The reconstruction property, REACH, the reason form, the error fallback, the phrase table of the pass-2 note (no key
is added or removed here), the operator options and the disclosed gaps. G-P1 is still not met, and that is still the
operator's decision.
