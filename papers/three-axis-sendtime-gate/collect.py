"""Live + replay collection harness for the three-axis send-time gate.

Wires the REAL styxx text-axis (cogn_audit_on_send) into the full pipeline:
T + I_fd + I_rg + D_cont + P + M_jury + gate decision.

Two modes:
- collect(prompt, draft, category) — one fresh trajectory, appends to trajectories.jsonl
- replay_text_only(memory_jsonl) — backfill text-axis for existing reflex-loop msg_ids
                                    (per PROTOCOL §4.3, contributes to T denominator only)

NEVER mutates the existing styxx audit; this is a measurement layer on top.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import traceback
from typing import Any

from styxx.middleware import cogn_audit_on_send
from styxx.three_axis.regen_scorer import regenerate_and_score
from styxx.three_axis.forced_decode import forced_decode_score
from styxx.three_axis.paraphrase import paraphrase_invariance
from styxx.three_axis.meta_rate import jury
from styxx.three_axis.differential_gate import decide_dict

HERE = pathlib.Path(__file__).parent
TRAJ = HERE / "trajectories.jsonl"
LABELS = HERE / "labels.jsonl"


def real_text_axis(text: str, prompt: str = "") -> dict[str, Any]:
    """Run real styxx cogn audit on a (prompt, draft) pair. Returns flat dict."""
    _, traj = cogn_audit_on_send(
        prompt=prompt or "(none)", draft=text, max_revise=0, persist_to_chart=False,
    )
    c = traj.chosen
    scores = c.get("scores", {}) or {}
    return {
        "composite": c.get("composite", 0.0),
        "sycophancy": scores.get("sycophancy", 0.0),
        "deception": scores.get("deception", 0.0),
        "overconfidence": scores.get("overconfidence", 0.0),
        "refusal": scores.get("refusal", 0.0),
        "firing_instruments": c.get("firing_instruments", []),
        "construct_ceiling_fires": c.get("construct_ceiling_fires", []),
        "ceiling_only": c.get("ceiling_only", False),
        "passed": c.get("passed", False),
    }


def collect_one(
    *,
    system_prompt: str,
    user_prompt: str,
    draft: str,
    category: str,
    msg_id: str | None = None,
    skip_axes: set[str] | None = None,
) -> dict[str, Any]:
    """Run the full three-axis pipeline on one (prompt, draft) and write to trajectories.jsonl."""
    skip = skip_axes or set()
    row: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "msg_id": msg_id,
        "category": category,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "draft": draft,
        "protocol_lock_sha": "aaae5f4+amendment3:09ad3df",
    }

    # T — REAL text-axis
    try:
        row["T"] = real_text_axis(draft, prompt=user_prompt)
    except Exception as e:
        row["T"] = {"_error": f"{type(e).__name__}: {e}"}

    # I_fd
    if "I_fd" not in skip:
        try:
            r = forced_decode_score(
                prefix=system_prompt + "\n\n" + user_prompt + "\n\n", draft_text=draft,
            )
            row["I_fd"] = {k: v for k, v in r.items() if k != "per_token"}
        except Exception as e:
            row["I_fd"] = {"_error": f"{type(e).__name__}: {e}",
                           "_traceback": traceback.format_exc()[-500:]}

    # I_rg + D_cont
    if "I_rg" not in skip:
        try:
            row["I_rg"] = regenerate_and_score(system_prompt, user_prompt, draft)
        except Exception as e:
            row["I_rg"] = {"_error": f"{type(e).__name__}: {e}"}

    # P
    if "P" not in skip:
        try:
            p = paraphrase_invariance(
                draft, text_axis_fn=lambda t: real_text_axis(t, prompt=user_prompt), k=5,
            )
            row["P"] = {
                "k_valid": p["k_valid"],
                "P_per_construct": p["P_per_construct"],
                "P_composite": p["P_composite"],
                "paraphrase_previews": [(pp.get("text") or "")[:160] for pp in p["paraphrases"]],
                "preserved_flags": [pp.get("preserved_all_claims") for pp in p["paraphrases"]],
            }
        except Exception as e:
            row["P"] = {"_error": f"{type(e).__name__}: {e}"}

    # M_jury
    if "M" not in skip:
        try:
            m = jury(user_prompt, draft)
            row["M_jury"] = {
                "M_self": {k: v for k, v in m["M_self"].items()},
                "M_4o": {k: v for k, v in m["M_4o"].items()},
                "M_41": {k: v for k, v in m["M_41"].items()},
                "peer_mean": m["peer_mean"],
                "Delta_self": m["Delta_self"],
            }
        except Exception as e:
            row["M_jury"] = {"_error": f"{type(e).__name__}: {e}"}

    # Gate decision
    try:
        row["decision"] = decide_dict(
            text_axis=row.get("T") if isinstance(row.get("T"), dict) and "_error" not in row["T"] else None,
            meta_jury=row.get("M_jury") if isinstance(row.get("M_jury"), dict) and "_error" not in row["M_jury"] else None,
            paraphrase={"P_per_construct": row["P"].get("P_per_construct", {})}
                       if isinstance(row.get("P"), dict) and "P_per_construct" in row["P"] else None,
            i_fd=row.get("I_fd"),
            i_rg=row.get("I_rg"),
        )
        # strip duplicate axis fields the dataclass carries
        for k in ("text_axis", "meta_jury", "paraphrase", "i_fd", "i_rg"):
            row["decision"].pop(k, None)
    except Exception as e:
        row["decision"] = {"_error": f"{type(e).__name__}: {e}"}

    # Append to trajectory log
    with TRAJ.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

    return row


def replay_text_only(memory_jsonl: pathlib.Path):
    """Backfill text-axis on the n=7 existing reflex-loop drafts.

    Per PROTOCOL §4.3, post-hoc capture of meta/internal axes on these
    drafts is INVALID (the agent has already seen its own audit firings
    in conversation context). We only re-score the text-axis for the T
    denominator.
    """
    out_rows = []
    with memory_jsonl.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            drafts = entry.get("drafts", [])
            for d in drafts:
                if not d.get("SHIPPED"):
                    continue
                row = {
                    "ts_original": entry.get("ts"),
                    "msg_id": entry.get("msg_id"),
                    "category": entry.get("category", "cooperative_reply"),
                    "user": entry.get("user"),
                    "shipped_version": d.get("v"),
                    "replay_text_axis_only": True,
                    "original_scores": {
                        "composite": d.get("composite"),
                        "sycophancy": d.get("sycophancy"),
                        "overconfidence": d.get("overconfidence"),
                        "refusal": d.get("refusal"),
                        "deception": d.get("deception"),
                        "firing": d.get("firing"),
                    },
                }
                out_rows.append(row)
    return out_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--system-prompt", required=True)
    c.add_argument("--user-prompt", required=True)
    c.add_argument("--draft", required=True)
    c.add_argument("--category", required=True)
    c.add_argument("--msg-id", default=None)
    c.add_argument("--skip", nargs="*", default=[])
    r = sub.add_parser("replay")
    r.add_argument("--memory-jsonl", default="../../../memory/cognometric-trajectory.jsonl")
    args = ap.parse_args()

    if args.cmd == "collect":
        row = collect_one(
            system_prompt=args.system_prompt, user_prompt=args.user_prompt,
            draft=args.draft, category=args.category, msg_id=args.msg_id,
            skip_axes=set(args.skip),
        )
        print(json.dumps({
            "msg_id": row.get("msg_id"),
            "category": row.get("category"),
            "T_composite": (row.get("T") or {}).get("composite"),
            "T_ceiling_only": (row.get("T") or {}).get("ceiling_only"),
            "I_fd_slope": (row.get("I_fd") or {}).get("entropy_slope"),
            "D_cont": (row.get("I_rg") or {}).get("D_cont"),
            "P_composite": (row.get("P") or {}).get("P_composite"),
            "Delta_self": (row.get("M_jury") or {}).get("Delta_self"),
            "decision": (row.get("decision") or {}).get("verdict"),
            "reason": (row.get("decision") or {}).get("reason"),
        }, indent=2))

    elif args.cmd == "replay":
        rows = replay_text_only(pathlib.Path(args.memory_jsonl))
        replay_path = HERE / "trajectories_replay_text_only.jsonl"
        with replay_path.open("w", encoding="utf-8") as f:
            for r_ in rows:
                f.write(json.dumps(r_, default=str) + "\n")
        print(f"Wrote {len(rows)} replayed text-only rows to {replay_path}")
