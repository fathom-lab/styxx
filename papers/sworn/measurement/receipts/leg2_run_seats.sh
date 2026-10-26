#!/usr/bin/env bash
# Leg 2, tranche A: the six Claude seat files on the amended alias, then the nine trivial twins (PREREG section 4,
# Amendment 6 items 2-4). Run from this Claude Code session's environment, in the run worktree checked out at a main
# that contains Amendment 6 (#235). Sequential, one quiet window. Stops on the first seat that is not written, is VOID,
# or records another alias. Never edits a file; seat files land under papers/sworn/measurement/seat_outputs/claude/.
set -u
WT=/c/Users/heyzo/clawd/wt/leg2-run
M=papers/sworn/measurement
ALIAS=claude-opus-5-5
LOG=/c/Users/heyzo/clawd/wt/_next_level_2026_10_12/tonight/leg2/seat_run_$(date -u +%Y%m%dT%H%M%SZ).log
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd "$WT" || exit 2
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }

say "run worktree head $(git rev-parse --short HEAD); amendment present: $(git ls-files "papers/sworn/AMENDMENT_sworn_measurement_leg2_seat_alias_2026_10_24.md" | wc -l)"
[ "$(git ls-files papers/sworn/AMENDMENT_sworn_measurement_leg2_seat_alias_2026_10_24.md | wc -l)" = 1 ] || { say "REFUSED: Amendment 6 is not committed at this head"; exit 3; }
[ -z "$(git status --short -- papers/sworn | grep -v '^??' )" ] || { say "REFUSED: tracked changes under papers/sworn"; exit 3; }

# Item 4: the seats start on a later day than 2026-10-24 -> the one-item check runs again over a fresh synthetic copy.
if [ "$(date -u +%Y-%m-%d)" != "2026-10-24" ]; then
  TC="transport_check_$(date -u +%Y_%m_%d)_opus"
  if [ ! -d "$M/$TC" ]; then
    mkdir -p "$M/$TC/keys"
    for f in population.json packet_L.json packet_R.json decoys_L.json packets_digest.txt; do cp "$M/dryrun/$f" "$M/$TC/"; done
    cp "$M"/dryrun/keys/* "$M/$TC/keys/"; cp "$M"/dryrun/SYN*.json "$M/$TC/"
  fi
  say "transport check on the day: $TC"
  (cd "$M" && py -3.12 seat_claude.py --packet L --seat 1 --transport-check --max-items 1 --dir "$TC" --model "$ALIAS") 2>&1 | tee -a "$LOG" | tail -2
  grep -q '"verdict": "VOID' "$M/$TC/seat_outputs/claude/L-seat1.json" 2>/dev/null && { say "STOP: the day's transport check is VOID"; exit 4; }
fi

for P in L R; do
  for N in 1 2 3; do
    OUT="$M/seat_outputs/claude/$P-seat$N.json"
    if [ -f "$OUT" ]; then say "skip $P-seat$N: already written"; continue; fi
    say "start $P-seat$N"
    py -3.12 "$M/seat_claude.py" --packet "$P" --seat "$N" --model "$ALIAS" >> "$LOG" 2>&1
    rc=$?
    if [ ! -f "$OUT" ]; then say "STOP: $P-seat$N wrote no seat file (exit $rc); it can be re-run"; exit 5; fi
    read -r MODEL VERDICT ITEMS ERRS < <(py -3.12 -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d['substrate']['model'], d['verdict'], len(d['items']), len(d['errors']))" "$OUT")
    say "done $P-seat$N: model=$MODEL verdict=$VERDICT items=$ITEMS errors=$ERRS (exit $rc)"
    [ "$MODEL" = "$ALIAS" ] || { say "STOP: $P-seat$N recorded alias $MODEL"; exit 6; }
    case "$VERDICT" in VOID*) say "STOP: $P-seat$N is $VERDICT"; exit 7;; esac
  done
done

for D in D01 D02 D03 D04 D05 D06 D07 D08 D09; do
  say "twin $D"
  py -3.12 -c "import functools, sys; sys.path.insert(0, 'papers/sworn/measurement'); import seat_claude as SC; SC.cli = functools.partial(SC.cli, model='$ALIAS'); import twin_trivial as T; sys.exit(T.main(sys.argv[1:]))" --doc "$D" >> "$LOG" 2>&1
  say "twin $D exit $?"
done
say "ALL DONE: six seat files and nine twins; next the local seats (GPU, by day), then score.py and verdict_input.py --tranche A"
