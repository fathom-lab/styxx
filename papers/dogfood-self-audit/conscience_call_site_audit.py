"""The live conscience is not the conscience gate. Measured on his own traffic.

styxx spent a self-audit cycle building `needs_revision` to be DISCIPLINED. Its docstring
is explicit: overconfidence is a construct-ceiling detector that "saturates on any
declarative phrasing" and so is scored but not trusted to gate; reference-less deception
is excluded from the composite outright; the firing condition is a CONJUNCTION so that a
documented non-discriminative axis "can never be the SOLE reason a draft is told to
revise". That is the 2026-05-24 alarm-fatigue fix, and it is real work.

darkflobi's live send path then calls it like this (darkflobi_glimmer.py:520):

    fired = bool(getattr(v, "fired", False) or getattr(v, "needs_revision", False))

`ConscienceVerdict.fired` is a LIST of advice items (conscience.py:53), and a non-empty
list is truthy. So the live flag is `advice_is_non_empty OR needs_revision` -- a
disjunction in which the first term is strictly looser than the second. **`needs_revision`
cannot change the outcome of that expression.** Every calibration decision above is
bypassed at the call site by one `or`.

The symptom was already visible and was read as weather rather than as a bug. The comment
four lines below that one says "the register gate fires on nearly every turn", and this
morning's live-traffic probe measured the conscience firing on 67% of turns and scoring
deception 0.9993 on "gn flobi". Deception is not in the composite and not in the gate
keys -- it cannot raise needs_revision at all -- yet it fires the live flag, because
producing ADVICE is all the live flag requires.

WHAT THIS SCRIPT MEASURES, rather than argues: for every logged turn that carries
conscience scores, recompute the calibrated `needs_revision` with the real function and
the real reply text, and compare it to what the live path actually did.

    python conscience_call_site_audit.py
"""
from __future__ import annotations

import io
import json
import os
import sys

LOG = r"C:\Users\heyzo\.styxx\glimmer-day-zero\darkflobi_glimmer_log.jsonl"


def rows():
    out = []
    for line in io.open(LOG, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        # The verdict may sit at the top level or under a nested key depending on when
        # the turn was logged; take the first shape that carries real scores.
        for cand in (r, r.get("verdict") or {}, r.get("conscience") or {}):
            if isinstance(cand, dict) and isinstance(cand.get("scores"), dict) \
                    and cand["scores"]:
                out.append({
                    "scores": {k: float(v) for k, v in cand["scores"].items()
                               if isinstance(v, (int, float))},
                    "live_fired": bool(cand.get("fired")),
                    "reply": str(r.get("reply") or r.get("shipped") or ""),
                    "prompt": str(r.get("question") or r.get("prompt") or ""),
                })
                break
    return out


def main():
    sys.path.insert(0, r"C:\Users\heyzo\clawd\styxx")
    from styxx.cognometrics import _cogn_needs_revision, _cogn_gate_keys

    data = rows()
    if not data:
        print("  no logged turns carry conscience scores -- nothing measurable here,")
        print("  which is itself worth knowing before any claim is made about them")
        return

    print(f"  trusted gate keys (ungrounded live path): {_cogn_gate_keys()}")
    print(f"  n = {len(data)} logged turns carrying conscience scores\n")

    live_only = calibrated = both = neither = 0
    syc_above = 0
    for d in data:
        nr = bool(_cogn_needs_revision(d["scores"], response=d["reply"],
                                       prompt=d["prompt"]))
        lf = d["live_fired"]
        syc_above += (d["scores"].get("sycophancy", 0.0) > 0.30)
        if lf and nr:
            both += 1
        elif lf:
            live_only += 1
        elif nr:
            calibrated += 1
        else:
            neither += 1

    n = len(data)
    print(f"  live flag fired                : {both + live_only}/{n} "
          f"({(both + live_only) / n:.1%})")
    print(f"  calibrated needs_revision would: {both + calibrated}/{n} "
          f"({(both + calibrated) / n:.1%})")
    print(f"  sycophancy > 0.30 (the necessary condition): {syc_above}/{n} "
          f"({syc_above / n:.1%})\n")
    print(f"    both agree fire      {both}")
    print(f"    LIVE ONLY            {live_only}   <- softened by the live path; the")
    print(f"                              calibrated gate would have passed them")
    print(f"    calibrated only      {calibrated}   <- must be 0: the live flag ORs")
    print(f"                              needs_revision in, so it cannot miss one")
    print(f"    both agree quiet     {neither}")

    if calibrated:
        print("\n  UNEXPECTED: the live flag missed a calibrated firing. That should be")
        print("  impossible given the disjunction -- investigate before trusting any of")
        print("  the above.")
    else:
        print(f"\n  {live_only} of {n} turns ({live_only / n:.1%}) were softened on the")
        print("  strength of an advice list alone. The conjunction styxx built to stop")
        print("  exactly that is present in the code, correct, and unreachable.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "CONSCIENCE_CALL_SITE_AUDIT.json")
    io.open(out, "w", encoding="utf-8", newline="\n").write(json.dumps({
        "n": n, "trusted_gate_keys": _cogn_gate_keys(),
        "live_fired": both + live_only, "calibrated_would_fire": both + calibrated,
        "sycophancy_above_gate_threshold": syc_above,
        "live_only": live_only, "calibrated_only": calibrated,
        "both": both, "neither": neither}, indent=1) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
