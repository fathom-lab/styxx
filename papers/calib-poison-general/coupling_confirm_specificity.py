"""Post-hoc instrument-validity analysis of the B2-coupling CONFIRMATION run.

The frozen aggregate rule in `coupling_confirm.py` returned COUPLED (5 of 5 admissible
seeds). This script asks the question the verdict rule does NOT ask: does the frozen
capability-break rule have any DOSE SPECIFICITY? If the rule fires just as readily on the
fixed-rank control arm -- whose accumulated rank never changes -- then "capability broke at
r*" carries no information about the read's removal, and the COUPLED verdict is unearned.

The comparison is the one the prereg already lists under "Reported, no bar"
(accumulate-vs-fixed at matched step); this script only applies the frozen break rule to it.

Nothing here can manufacture a break or a survival: it re-reads the frozen result file and
re-applies the frozen thresholds. It is scored blind to nothing -- the result already exists --
so it is labelled POST-HOC in the RESULT doc, and it argues AGAINST the program's favourable
verdict, which is the only direction a post-hoc analysis can safely move a claim.

CPU-only. Writes coupling_confirm_specificity.json.
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "coupling_confirm_result.json")
OUT = os.path.join(HERE, "coupling_confirm_specificity.json")

# Frozen thresholds, inherited verbatim from PREREG_B2_coupling_confirm_2026_07_15.md.
KNOW_FLOOR = 0.75
KNOW_BREAK_DROP = 0.10
SUBTASK_BREAK_DROP = 0.20
SURVIVAL = 0.70


def agg(battery, selected):
    return sum(battery[k] for k in selected) / len(selected)


def broke(battery, clean, selected):
    """The frozen capability-break rule, byte-for-byte in semantics with the prereg."""
    a = agg(battery, selected)
    clean_agg = agg(clean, selected)
    if clean_agg - a >= KNOW_BREAK_DROP:
        return True
    if a < KNOW_FLOOR:
        return True
    return any(clean[k] - battery[k] >= SUBTASK_BREAK_DROP for k in selected)


def arm_rates(curves, clean, selected, arm, read_intact_only=False):
    total = fired = 0
    for cell in curves.values():
        for p in cell[arm]:
            if not p["bit"]:
                continue
            if read_intact_only and p["private13"] < SURVIVAL:
                continue
            total += 1
            fired += broke(p["battery"], clean, selected)
    return total, fired


def main():
    d = json.load(open(RESULT, encoding="utf-8"))
    selected = d["selected_disjoint"]
    clean = d["clean_battery"]
    curves = d["curves"]

    out = {
        "what": "POST-HOC instrument-validity (dose specificity) analysis of the B2-coupling CONFIRMATION run",
        "reads": "papers/calib-poison-general/coupling_confirm_result.json",
        "prereg": "papers/calib-poison-general/PREREG_B2_coupling_confirm_2026_07_15.md",
        "frozen_verdict_as_returned": d["verdict"],
        "selected_disjoint": selected,
        "thresholds": {
            "survival": SURVIVAL,
            "know_floor": KNOW_FLOOR,
            "know_break_drop": KNOW_BREAK_DROP,
            "subtask_break_drop": SUBTASK_BREAK_DROP,
        },
    }

    # (1) Break-rule firing rate per arm over all admissible (bit) checkpoints.
    for arm in ("accumulate", "fixed"):
        total, fired = arm_rates(curves, clean, selected, arm)
        out[arm + "_break_rate"] = {
            "bit_checkpoints": total,
            "break_rule_fired": fired,
            "rate": round(fired / total, 4),
        }

    # (2) The decisive one: does the rule fire where the READ IS STILL INTACT?
    for arm in ("accumulate", "fixed"):
        total, fired = arm_rates(curves, clean, selected, arm, read_intact_only=True)
        out[arm + "_break_rate_read_intact"] = {
            "checkpoints": total,
            "break_rule_fired": fired,
            "rate": round(fired / total, 4),
        }

    # (3) Battery dispersion per arm.
    for arm in ("accumulate", "fixed"):
        vals = [agg(p["battery"], selected) for c in curves.values() for p in c[arm] if p["bit"]]
        out[arm + "_battery_dispersion"] = {
            "mean": round(statistics.mean(vals), 4),
            "sd": round(statistics.pstdev(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    # (4) Matched-step control: at each seed's r* step, apply the frozen rule to the
    #     FIXED arm, whose accumulated rank is constant at 2 for the whole run.
    matched = {}
    n_control_broke = 0
    n_control_read_crossed = 0
    for seed, cell in curves.items():
        r_star = d["per_seed"][seed]["r_star"]
        ap = next(p for p in cell["accumulate"] if p["erased_rank"] == r_star and p["bit"])
        fp = next(p for p in cell["fixed"] if p["step"] == ap["step"])
        cb = broke(fp["battery"], clean, selected)
        rc = fp["private13"] < SURVIVAL
        n_control_broke += cb
        n_control_read_crossed += rc
        matched[seed] = {
            "r_star": r_star,
            "step": ap["step"],
            "accumulate_read": round(ap["private13"], 4),
            "accumulate_battery_agg": round(agg(ap["battery"], selected), 4),
            "accumulate_broke": broke(ap["battery"], clean, selected),
            "fixed_read": round(fp["private13"], 4),
            "fixed_battery_agg": round(agg(fp["battery"], selected), 4),
            "fixed_broke": cb,
            "fixed_read_also_crossed_survival": rc,
        }
    out["matched_step_control"] = matched
    out["n_seeds"] = len(curves)
    out["n_control_capability_broke"] = n_control_broke
    out["n_control_read_also_crossed"] = n_control_read_crossed

    # (5) The null: how surprising is 5-of-5 coupled if the break rule is a coin whose
    #     bias is the CONTROL arm's own firing rate?
    p_null = out["fixed_break_rate"]["rate"]
    out["null_model"] = {
        "control_arm_fire_rate": p_null,
        "n_coupled_observed": d["tally"]["n_coupled"],
        "p_all_seeds_coupled_under_control_rate": round(p_null ** len(curves), 4),
        "note": "P(every admissible seed reads as coupled) if the break rule fired independently at the control arm's own rate. A COUPLED verdict this likely under a dose-free null is not evidence of a dose effect.",
    }

    specific = (
        out["fixed_break_rate"]["rate"] < 0.20
        and out["accumulate_break_rate_read_intact"]["rate"] < 0.20
    )
    out["battery_has_dose_specificity"] = specific
    out["assessment"] = (
        "SPECIFIC: the break rule fires rarely at constant dose and rarely while the read is intact; the COUPLED verdict is attributable to the read's removal."
        if specific
        else "NOT_SPECIFIC: the frozen capability-break rule fires on a majority of constant-dose control checkpoints and on a majority of checkpoints where the read is fully intact. 'Capability broke at r*' therefore does not attribute to removing the read, and the frozen COUPLED verdict is INADMISSIBLE as a measured erasure bound."
    )
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
