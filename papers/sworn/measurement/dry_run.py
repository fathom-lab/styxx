# -*- coding: utf-8 -*-
"""The whole ladder over synthetic items, in one process, under dryrun/ (SPEC §The dry run).

population -> packets -> decoys -> keys -> canaries (one planted to be MALFORMED) -> three canned
seats per family per panel (one family built to fail Panel R) -> a trivial twin -> the scorer. It
writes ``dryrun/dry_run_result.json`` with ``"dry_run": true, "quotable": false``, prints counts and
digests, and replaces every share, interval, kappa and Q3 value with ``DRYRUN-NO-RATE``.

Refusals (SystemExit beginning ``REFUSED:``): a population entry whose stem resolves to a file in the
repository, or whose doc_id does not begin ``SYN-``; a sealed directory equal to ``$STYXX_SEALED_DIR``;
an output path outside a directory named ``dryrun/``.

CLI: ``python papers/sworn/measurement/dry_run.py [--out papers/sworn/measurement/dryrun] [--sealed DIR]``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_packets as B                            # noqa: E402
import canaries as CAN                               # noqa: E402
import common as C                                   # noqa: E402
import population as P                               # noqa: E402
import score as SCORE                                # noqa: E402
import seal_key as K                                 # noqa: E402
import seat_claude as SCL                            # noqa: E402
import seat_local as SL                              # noqa: E402
import synthetic as S                                # noqa: E402
import twin_trivial as TT                            # noqa: E402

BANNER = "DRY RUN - no quotable number"


def refuse_unless_synthetic(pop: dict, root: Path) -> None:
    for e in P.iter_documents(pop) + P.iter_excluded(pop):
        if not str(e.get("doc_id", "SYN-")).startswith("SYN-"):
            raise SystemExit("REFUSED: dry-run doc_id %r does not begin SYN-" % e.get("doc_id"))
        src = e.get("source") or {}
        if src.get("kind") != "synthetic":
            raise SystemExit("REFUSED: dry-run entry %s is not synthetic" % e["stem"])
        for suffix in (".sworn.json", ".md"):
            if (root / (e["stem"] + suffix)).exists() and C.tracked_at("HEAD", e["stem"] + suffix, root=root):
                raise SystemExit("REFUSED: dry-run stem %s resolves to a file in the repository" % e["stem"])


def main_run(out_dir=None, sealed=None, keep_sealed: bool = False) -> dict:
    out_dir = Path(out_dir or HERE / "dryrun")
    if out_dir.name != "dryrun":
        raise SystemExit("REFUSED: the dry run writes only under a directory named dryrun/")
    root = out_dir.parent            # synthetic sources are named relative to the dry run's parent
    tmp_sealed = sealed is None
    sealed = Path(sealed or tempfile.mkdtemp(prefix="sworn_measurement_dryrun_sealed_"))
    if sealed.resolve() == Path(C.SEALED).resolve():
        raise SystemExit("REFUSED: the dry run never writes to the real sealed directory")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("REFUSED: %s is not empty; a re-run is a new directory at a new commit" % out_dir)
    written: List[Path] = []
    try:
        pop = S.write_population(out_dir, root)
        refuse_unless_synthetic(pop, Path(C.ROOT))
        pop_path = C.write_json_lf(out_dir / "population.json", pop)
        K.new_salt(sealed)
        picks = S.decoy_picks(pop, root)
        pk = B.build(pop, pop_path, picks=picks, root=root, sealed=sealed, out_dir=out_dir, keys_dir=out_dir / "keys")
        cn = CAN.build(pop, root=root, sealed=sealed, keys_dir=out_dir / "keys",
                       digest_path=out_dir / "twins" / "canary_digest.txt", force_malformed_in="SYN-01")
        seats = {}
        for panel in ("L", "R"):
            for seat in range(1, C.SEATS_PER_FAMILY + 1):
                h = SCL.run(panel, seat, out_dir, dry_run=True, root=root)
                seats["claude/%s-seat%d" % (panel, seat)] = {"items": len(h["items"]), "unparsed": len(h["unparsed"])}
                h = SL.run(panel, seat, out_dir, dry_run=True, root=root)
                seats["local/%s-seat%d" % (panel, seat)] = {"items": len(h["items"]), "unparsed": len(h["unparsed"])}
        twins = {}
        for e in P.iter_documents(pop):
            st = TT.build(e["doc_id"], out_dir, dry_run=True, root=root)
            twins[e["doc_id"]] = {"spans_original": st.get("spans_original"), "spans_twin": st.get("spans_twin"),
                                  "twin_text_changed": st.get("twin_text_changed")}
        result = SCORE.fold(out_dir, sealed, out_dir / "dry_run_result.json", dry_run=True, root=root)
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                written.append(p)
                if p.name.endswith((".sworn.json", ".sworn-receipt.json")) or p.name.startswith("PREREG_"):
                    raise SystemExit("REFUSED: the dry run wrote %s" % p.name)
                if p.suffix in (".json", ".txt", ".jsonl") and b"\r" in p.read_bytes():
                    raise SystemExit("REFUSED: %s is not LF-only" % p.name)
        summary = {
            "banner": BANNER, "dry_run": True, "quotable": False,
            "population": {"documents": len(pop["documents"]), "excluded": len(pop["excluded"])},
            "packets": {k: v for k, v in pk.items() if isinstance(v, int)},
            "canaries": {"capacity": cn["capacity"], "pooled_n": cn["pooled_n"],
                         "smallest_n_clearing_bar_at_k_eq_n": cn["smallest_n_clearing_bar_at_k_eq_n"]},
            "seats": seats, "trivial_twins": twins,
            "gates": {"G_D": result["gates"]["G_D"], "G_F": result["gates"]["G_F"]["families_clearing"],
                      "G_C": {k: result["gates"]["G_C"][k] for k in ("k", "n", "per_twin", "wilson95")},
                      "withheld": result["withheld"]},
            "cells": result["cells"],
            "files": {p.relative_to(out_dir).as_posix(): C.sha256_file(p) for p in written},
            "sealed_dir_was_temporary": tmp_sealed,
        }
        C.write_json_lf(out_dir / "dry_run_summary.json", summary)
        return summary
    finally:
        if tmp_sealed and not keep_sealed:
            shutil.rmtree(sealed, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--sealed", default=None)
    a = ap.parse_args(argv)
    s = main_run(a.out, a.sealed)
    print(json.dumps({k: s[k] for k in ("population", "packets", "canaries", "seats", "gates", "cells")}, indent=1))
    print("files written: %d" % len(s["files"]))
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())

