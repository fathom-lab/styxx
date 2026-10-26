#!/usr/bin/env bash
# Leg 2, tranche A, the local family (PREREG Amendment 1): Qwen/Qwen2.5-3B-Instruct, bf16, --device cuda, by day,
# each seat under the box's GPU lease, out by 22:00 local. Usage: leg2_run_local.sh L|R [first_estimate_seconds]
# Runs seats 1-3 of the packet in order, skipping seat files already written; starts a seat only when its expected
# end (the previous seat's measured duration, else the given estimate) falls before 22:00 local. Stops on a seat
# that writes no file or records another model.
set -u
P=${1:?packet L or R}
EST=${2:-5400}
WT=/c/Users/heyzo/clawd/wt/leg2-run
M=papers/sworn/measurement
MODEL=Qwen/Qwen2.5-3B-Instruct
PY=C:/Users/heyzo/AppData/Local/Programs/Python/Python312/python.exe
BOX=C:/Users/heyzo/clawd/wt/_box
LOG=/c/Users/heyzo/clawd/wt/_next_level_2026_10_12/tonight/leg2/local_run_${P}_$(date +%Y%m%dT%H%M%S).log
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd "$WT" || exit 2
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
deadline=$(date -d "today 22:00" +%s)
say "local seats, packet $P, head $(git rev-parse --short HEAD), deadline 22:00 local, first estimate ${EST}s"
for N in 1 2 3; do
  OUT="$M/seat_outputs/local/$P-seat$N.json"
  [ -f "$OUT" ] && { say "skip $P-seat$N: already written"; continue; }
  now=$(date +%s)
  if [ $((now + EST)) -gt "$deadline" ]; then say "STOP for today: $P-seat$N expected to end after 22:00 (estimate ${EST}s)"; exit 0; fi
  say "start $P-seat$N (expected ~$((EST/60)) min)"
  t0=$(date +%s)
  "$PY" "$BOX/lease.py" hold --reserve-gib 6 --holder "leg2 local $P-seat$N" --launcher-allow "seat_local.py --packet $P --seat $N" \
    -- "$PY" "$M/seat_local.py" --packet "$P" --seat "$N" --model "$MODEL" --dtype bf16 --device cuda >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s); EST=$((t1 - t0))
  if [ ! -f "$OUT" ]; then say "STOP: $P-seat$N wrote no seat file (exit $rc, ${EST}s); it can be re-run"; exit 5; fi
  read -r MOD NAMED VERDICT ITEMS ERRS < <("$PY" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); s=d['substrate']; print(s['model'], s.get('named_in_design'), d['verdict'], len(d['items']), len(d['errors']))" "$OUT")
  say "done $P-seat$N in ${EST}s: model=$MOD named_in_design=$NAMED verdict=$VERDICT items=$ITEMS errors=$ERRS (exit $rc)"
  [ "$MOD" = "$MODEL" ] || { say "STOP: $P-seat$N recorded model $MOD"; exit 6; }
done
say "packet $P: all three local seats written"
