# papers/darkflobi-gate — the preregistration of darkflobi under the gate

fathom lab, 2026-10-23. This folder holds the frozen half of the plan `PLAN_darkflobi_under_the_gate.md`
(`WT:_darkflobi_gate_2026_10_22/`, sha256 `665823b5…`): the rule the measurement is counted under and the
preregistration that names the population, the three cells, the gates block, the canaries, the panel and the
printable sentences, before the gate is wired to any exit. Nothing in this folder runs a gate, sends a message, or
reads a session transcript.

## what is here

| file | what it is |
|---|---|
| `RULES_darkflobi_gate_2026_10_23.md` | the rule file: *claim*, *bound* (the two-form `read_from` grammar), *public destination* (the keyed-hash predicate), the exits, the claim shapes and which are checked, the media rule (absent, said as such), the anchor rules, the refusal reason codes and matrix, the selector, the gateway's own fixed strings, the census normalisation rule, the ledger fields the measurement reads |
| `PREREG_darkflobi_gate_2026_10_23.md` | the preregistration: the rule file's sha256, the three sentences under test, the population by destination (anchor channel, canary chat and operator DM excluded by keyed hash), the three cells, the result dict, the gates block (sha256 `4ed27cb98f72…`, 6,354 bytes), the canaries (four a day), the two-family panel, the gate's own checks, one printable paragraph per verdict, the lock-hash table, the order of operations, the decisions assumed, §15 for dated amendments before the merge |
| `lock_hashes.py` | standard library (and `git` for `pin-tree`); prints the lock table from the box (`table`), a folder's tree digest (`tree DIR`, refusing rather than skipping a path it cannot read whole), the snapshot's digest from the pin's own blobs (`pin-tree`), and re-derives every lock row of the PREREG from the box (`check PREREG.md`: SAME, CHANGED, BUILT SINCE, NOT BUILT, ABSENT, UNREADABLE per row, then UNLOCKED for every box file no row covers and BYTECODE for every `__pycache__` folder or `.pyc` file where the lock covers `W:gate/`; `--strict` fails on ABSENT, UNLOCKED and BYTECODE too) |
| `keys/sworn_measurement_darkflobi_gate_canary_seed.sha256` | the canary seed's salted digest by `seal_key.py`'s rule; the seed is sealed on the box under `W:gate/canary_sealed/` (PREREG §7) and this digest is public from the pull request's opened-at |
| `scan_gates.py` | standard library plus the repository's own `styxx.protocol`; prints the gates block's digest, builds `Experiment` with the power-basis and non-vacuity refusals on, enumerates the 1,024 assignments, scores one synthetic result per realisable assignment, and prints the Wilson arithmetic behind G_C and G_N |
| `protocol_scan_2026_10_23.txt` | the scanner's output over the committed PREREG, with the PREREG's blob sha256 and commit (written in the commit after the PREREG's) |
| `tests/test_darkflobi_gate_prereg.py` (under the repository's `tests/`) | pins the block's digest and byte count, the one-fence and no-duplicate-key rules, totality and reachability of the table with the row counts, the eleven verdicts through `Experiment`, the Wilson numbers the power bases quote, the rule file's hash as the PREREG prints it, the lock table's form, the snapshot row against the pin's own blobs, the rule file's selector cases and refusal codes against the PREREG's, the canary seed digest's row, the canary section's cadence and catch rule, the census rule's four end forms on their cases, the printable per verdict, the two helper scripts on synthetic input (the walker's exclusions, its refusals, a path past 260 characters, the bytecode scan), and that none of the files prints the words the lab never prints; with `STYXX_GATE_W_ROOT` naming the box it also compares the selector list with `selector.py`, the codes with what `gate.py`, the plugin and its helper emit (read with `ast`), the ledger kinds with `ledger.py`, the census forms and constants with `reconcile.py`, the canary section with `canary_shadow.py` and its schedule, and the lock rows with the files |

## what the tests check, and do not

They check bytes and arithmetic: that the gates block is the one the measurement design scanned, that its table is
total and every row reachable, that the scorer walks a receipt of the stated shape to every verdict, that the rule
file is the one the PREREG hashes, that the snapshot row is the digest the pin's blobs give, that the lock table parses
and (when `STYXX_GATE_W_ROOT` names the box's `W:` folder) agrees with the files there, that the rule file's selector
codes, kinds, census rule and canary plan are the pieces' own (on the box), and that the helper scripts do what their
docstrings say on a synthetic tree. They
run offline, on CPU, in well under two minutes, and skip the `Experiment` construction when the PREREG is not committed
in the checkout (the protocol refuses a draft on disk).

They do not check that any gate leg, plugin, ledger writer, canary harness, reconciler or census does what the plan
says (each piece under `W:gate/` ships with its own tests), that the hashes were computed before anything else (the
pull request's server timestamp does), that the operator's decisions were signed, or anything about the window's
data, which does not exist when this folder is written.

## how to run

    py -3.12 -m pytest tests/test_darkflobi_gate_prereg.py -q
    py -3.12 papers/darkflobi-gate/scan_gates.py
    py -3.12 papers/darkflobi-gate/lock_hashes.py pin-tree
    py -3.12 papers/darkflobi-gate/lock_hashes.py check papers/darkflobi-gate/PREREG_darkflobi_gate_2026_10_23.md --strict

The lint: `py -3.12 scripts/surface_lint.py --files papers/darkflobi-gate/*.md` prints 0 hits over this folder's
documents at the commit that added them; the result of a run made after a file was written cannot be carried in
that file.

## what this folder does not establish

That any post is true; that the agent is honest; a recall; a priority. The plan's §9 and the PREREG's §14 say the
rest.
