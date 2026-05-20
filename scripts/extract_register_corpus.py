"""
Extract anonymized register-firing corpus from an agent's cognometric
trajectory log into a regression fixture.

Source:  memory/cognometric-trajectory.jsonl  (one record per shipped reply,
         each containing N drafts with per-instrument scores + which fired)
Output:  tests/fixtures/register_corpus.jsonl

Anonymization rules:
- NO prompt text, NO draft text. Only scores + features + firing label.
- Each fixture is identified by a stable hash of (msg_id, draft_v).
- Source-trajectory `note` is kept ONLY if it contains no proper nouns or
  message content (we whitelist a short structural vocabulary).

Usage:
    python scripts/extract_register_corpus.py \
        --in  ../memory/cognometric-trajectory.jsonl \
        --out tests/fixtures/register_corpus.jsonl
    # default paths assume run from styxx repo root

Run daily (cron or heartbeat). The regression test
`tests/test_register_fixtures.py` consumes the output and verifies each
historical firing still fires within ±0.05 of the recorded score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

INSTRUMENTS = ("sycophancy", "deception", "overconfidence", "refusal")

# Anything beyond these tokens in a `note` field gets the note dropped
# (paranoid: better no metadata than leaked content).
_NOTE_SAFE = re.compile(
    r"^[a-z0-9 _\-\.\,\:\(\)\/\+]+$", re.IGNORECASE
)


def _safe_note(note: str | None) -> str | None:
    if not note:
        return None
    if len(note) > 200:
        return None
    return note if _NOTE_SAFE.match(note) else None


def _fixture_id(msg_id: Any, v: int) -> str:
    raw = f"{msg_id}:{v}".encode("utf-8")
    return "rf_" + hashlib.sha256(raw).hexdigest()[:12]


def _draft_to_fixture(
    *, record: Dict[str, Any], draft: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Convert one draft into an anonymized fixture row, or None to skip."""
    # Only keep drafts where at least one instrument actually fired.
    scores = {k: float(draft.get(k, 0.0)) for k in INSTRUMENTS}
    composite = float(draft.get("composite", 0.0))
    needs_revision = bool(draft.get("needs_revision", False))
    firing_label = draft.get("firing")

    # A row is interesting if needs_revision OR any instrument >= 0.5.
    any_fire = needs_revision or any(s >= 0.5 for s in scores.values())
    if not any_fire:
        return None

    return {
        "id": _fixture_id(record.get("msg_id", "0"), int(draft.get("v", 0))),
        "styxx_version": record.get("styxx_version"),
        "ts": record.get("ts"),
        "draft_v": int(draft.get("v", 0)),
        "scores": scores,
        "composite": composite,
        "needs_revision": needs_revision,
        "firing": firing_label,
        "shipped": bool(draft.get("SHIPPED", False)),
        "note": _safe_note(record.get("note")),
        "tol": 0.05,  # regression tolerance per instrument
    }


def iter_fixtures(traj_path: Path) -> Iterable[Dict[str, Any]]:
    # utf-8-sig tolerates a BOM (PowerShell-written logs include one).
    with traj_path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for draft in rec.get("drafts", []):
                fx = _draft_to_fixture(record=rec, draft=draft)
                if fx is not None:
                    yield fx


def write_fixtures(rows: List[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deduplicate by id; later occurrences win (in case a record is appended
    # with corrected numbers — extremely rare, but be deterministic).
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_id[r["id"]] = r
    ordered = sorted(by_id.values(), key=lambda r: (r["ts"] or "", r["draft_v"]))
    with out_path.open("w", encoding="utf-8") as fh:
        for r in ordered:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return len(ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        default=str(Path(__file__).resolve().parents[2] / "memory" / "cognometric-trajectory.jsonl"),
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "register_corpus.jsonl"),
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        print(f"[register-corpus] no trajectory log at {in_path}; nothing to do.")
        return 0

    rows = list(iter_fixtures(in_path))
    n = write_fixtures(rows, out_path)
    print(f"[register-corpus] wrote {n} fixture rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
