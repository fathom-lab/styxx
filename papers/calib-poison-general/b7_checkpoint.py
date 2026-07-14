"""Crash-safe per-cell checkpointing for the B7 3B erasure run (stdlib only).

WHY THIS EXISTS. The B7 scored run is 4 cells x ~3.9 h = ~15 h on this 8GB card.
The 2026-07-13 overnight launch died after 2 cells (both seed-0, both SURVIVES)
because `b7_erasure_3b.py` wrote its result JSON only at the very END of all
cells -- a session/kernel death lost every completed cell from the result file.
This module adds crash-safety WITHOUT touching the science:

  - each completed cell dict is appended to a JSONL cache the instant it finishes;
  - the clean-guard block (computed once from the base model) is cached too;
  - a resumed launch skips (seed, alpha) cells already in the cache and computes
    the frozen verdict over the SAME set of cells (cached + fresh).

The verdict function operates on the list of cells irrespective of provenance, so
the verdict over a given set of 4 cells is byte-identical whether they came from
one process or several -- no bar, no guard, no attack, no audit is changed.

DISCLOSED NON-DETERMINISM (already covered by the B7 prereg's "bf16
non-deterministic; one run per cell"): if a launch still has work to do it
recomputes the clean guard / gold subspace / deploy direction fresh; a cached
cell keeps the reference it was scored against, a fresh cell uses the resuming
launch's reference. Each cell is an independent single run, so this changes
nothing about any cell's own admissibility or verdict. A launch with NO work
left (crashed between the last cell and the result write) reloads the cached
clean block and emits the result with no model load at all.

Smoke runs (`--smoke`) never touch these caches.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


def cell_key(cell: dict) -> tuple:
    """Stable identity of a cell = (seed, alpha)."""
    return (int(cell["seed"]), float(cell["alpha"]))


def cells_cache_path(result_path) -> Path:
    """.../b7_erasure_3b_result.json -> .../b7_erasure_3b_cells.jsonl"""
    return Path(result_path).with_name("b7_erasure_3b_cells.jsonl")


def clean_cache_path(result_path) -> Path:
    """.../b7_erasure_3b_result.json -> .../b7_erasure_3b_clean.json"""
    return Path(result_path).with_name("b7_erasure_3b_clean.json")


def load_cached_cells(path):
    """Return (cells, done_keys). Tolerates a missing file or a torn last line
    from a crash mid-write (a partial final line is dropped, not fatal)."""
    path = Path(path)
    if not path.exists():
        return [], set()
    cells = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cells.append(json.loads(line))
        except json.JSONDecodeError:
            # a crash can leave a truncated final line; drop it and keep the rest
            continue
    keys = {cell_key(c) for c in cells}
    return cells, keys


def append_cell(path, cell: dict) -> None:
    """Append one completed cell dict as a JSON line (flush+fsync for crash-safety).
    If a prior crash left a torn (newline-less) final line, start on a fresh line so
    the good cell is never concatenated onto the corrupt fragment."""
    path = Path(path)
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + json.dumps(cell) + "\n")
        f.flush()
        os.fsync(f.fileno())


def compact_cells(path, cells) -> None:
    """Rewrite the cache with exactly the given (already-parsed) cells, dropping any
    torn tail a crash may have left. Called once on resume to normalise the file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save_clean(path, block: dict) -> None:
    Path(path).write_text(json.dumps(block, indent=2) + "\n", encoding="utf-8")


def load_clean(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def remaining_cells(seeds, alphas, done_keys):
    """The (seed, alpha) pairs still to run, in the harness's canonical order."""
    return [(s, a) for s in seeds for a in alphas if (int(s), float(a)) not in done_keys]


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result = Path(d) / "b7_erasure_3b_result.json"
        cache = cells_cache_path(result)
        clean = clean_cache_path(result)
        assert cache.name == "b7_erasure_3b_cells.jsonl", cache.name
        assert clean.name == "b7_erasure_3b_clean.json", clean.name

        seeds, alphas = [0, 1], [1.0, 4.0]

        # fresh: nothing cached, everything remains, in canonical order
        cells, keys = load_cached_cells(cache)
        assert cells == [] and keys == set()
        assert remaining_cells(seeds, alphas, keys) == [(0, 1.0), (0, 4.0), (1, 1.0), (1, 4.0)]

        # simulate the overnight death: two seed-0 cells complete + get appended
        c0 = {"seed": 0, "alpha": 1.0, "private13_auroc": 0.8065, "survives_cell": True}
        c1 = {"seed": 0, "alpha": 4.0, "private13_auroc": 0.8378, "survives_cell": True}
        append_cell(cache, c0)
        append_cell(cache, c1)

        # resume: cached cells load, only seed-1 remains
        cells, keys = load_cached_cells(cache)
        assert len(cells) == 2 and keys == {(0, 1.0), (0, 4.0)}, keys
        assert remaining_cells(seeds, alphas, keys) == [(1, 1.0), (1, 4.0)]
        assert cells[0]["private13_auroc"] == 0.8065

        # a torn final line (crash mid-write) is dropped, not fatal
        with cache.open("a", encoding="utf-8") as f:
            f.write('{"seed": 1, "alpha": 1.0, "priv')  # truncated, no newline
        cells, keys = load_cached_cells(cache)
        assert len(cells) == 2 and keys == {(0, 1.0), (0, 4.0)}, keys

        # append after a torn tail must NOT concatenate onto the fragment
        compact_cells(cache, cells)          # normalise away the torn tail on resume
        assert load_cached_cells(cache)[1] == {(0, 1.0), (0, 4.0)}

        # clean-block round trip
        assert load_clean(clean) is None
        block = {"clean_private13_auroc": 0.9853, "clean_eval_knowledge": 0.9242,
                 "clean_guard_ok": True, "subspace_ranks": {"15": 2, "18": 2}}
        save_clean(clean, block)
        assert load_clean(clean) == block

        # idempotent full-completion: all four done -> nothing remains
        append_cell(cache, {"seed": 1, "alpha": 1.0, "survives_cell": True})
        append_cell(cache, {"seed": 1, "alpha": 4.0, "survives_cell": True})
        _, keys = load_cached_cells(cache)
        assert remaining_cells(seeds, alphas, keys) == []

    print("b7_checkpoint self-test PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
