# -*- coding: utf-8 -*-
"""The one script that knows the salt (SPEC §The sealed keys).

A key is ``{item_id: meta}`` serialised by ``common.key_bytes`` and written under
``$STYXX_SEALED_DIR`` as ``<name>.json``; the tree carries ``keys/<name>.sha256`` holding
``sha256(key_bytes + salt_utf8)  <name>.json``. Nothing here reads a document or a seat.

CLI: ``python papers/sworn/measurement/seal_key.py new-salt | seal <name> --from KEY.json |
check <name> | release <name> --every-seat-output-is-recorded``. ``check`` exits 1 on a mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C                                   # noqa: E402

SALT_NAME = "sworn_measurement_salt.txt"
KEY_PREFIX = "sworn_measurement_"
RELEASE_FLAG = "--every-seat-output-is-recorded"


def salt_path(sealed=None) -> Path:
    return Path(sealed or C.SEALED) / SALT_NAME


def new_salt(sealed=None) -> Path:
    p = salt_path(sealed)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit("REFUSED: %s exists; a new salt would orphan every committed digest" % p)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(os.urandom(32).hex() + "\n")
    return p


def read_salt(sealed=None) -> str:
    p = salt_path(sealed)
    if not p.exists():
        raise SystemExit("REFUSED: no salt at %s; run seal_key.py new-salt" % p)
    return p.read_text(encoding="utf-8").strip()


def _paths(name: str, sealed=None, keys_dir=None) -> Tuple[Path, Path]:
    if not name.startswith(KEY_PREFIX) or name.endswith(".json"):
        raise SystemExit("REFUSED: a key name begins %s and carries no suffix" % KEY_PREFIX)
    return (Path(sealed or C.SEALED) / (name + ".json"),
            Path(keys_dir or HERE / "keys") / (name + ".sha256"))


def seal(name: str, key: dict, sealed=None, keys_dir=None) -> dict:
    """Write the key to the sealed directory and its salted digest to keys/<name>.sha256."""
    kp, dp = _paths(name, sealed, keys_dir)
    kb = C.key_bytes(key)
    digest = C.salted_digest(kb, read_salt(sealed))
    kp.parent.mkdir(parents=True, exist_ok=True)
    with open(kp, "wb") as fh:
        fh.write(kb)
    dp.parent.mkdir(parents=True, exist_ok=True)
    with open(dp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%s  %s.json\n" % (digest, name))
    return {"key_path": str(kp), "digest_path": str(dp), "digest": digest, "items": len(key)}


def check(name: str, sealed=None, keys_dir=None) -> Tuple[bool, Optional[str], Optional[str]]:
    """(match, committed digest, recomputed digest)."""
    kp, dp = _paths(name, sealed, keys_dir)
    if not dp.exists():
        return False, None, None
    want, _ = C.read_digest_file(dp)
    if not kp.exists():
        return False, want, None
    got = C.salted_digest(kp.read_bytes(), read_salt(sealed))
    return want == got, want, got


def load_key(name: str, sealed=None, keys_dir=None) -> dict:
    """The key, only when its salted digest equals the committed one — the scorer's refusal."""
    ok, want, got = check(name, sealed, keys_dir)
    if not ok:
        raise SystemExit("REFUSED: answer-key digest for %s does not match the committed keys/%s.sha256 "
                         "(committed %s, recomputed %s)" % (name, name, want, got))
    kp, _ = _paths(name, sealed, keys_dir)
    return json.loads(kp.read_bytes().decode("utf-8"))


def release(name: str, sealed=None, keys_dir=None, asserted: bool = False) -> Tuple[Path, Path]:
    """Copy the plaintext key and the salt into keys/ — after every seat output is recorded."""
    if not asserted:
        raise SystemExit("REFUSED: release needs %s; the flag asserts what its name says" % RELEASE_FLAG)
    ok, _, _ = check(name, sealed, keys_dir)
    if not ok:
        raise SystemExit("REFUSED: %s does not match its committed digest; nothing released" % name)
    kp, dp = _paths(name, sealed, keys_dir)
    out_key = dp.with_name(name + ".json")
    out_salt = dp.with_name(SALT_NAME)
    shutil.copyfile(kp, out_key)
    shutil.copyfile(salt_path(sealed), out_salt)
    return out_key, out_salt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sealed", default=None)
    ap.add_argument("--keys-dir", default=None)
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("new-salt")
    s = sub.add_parser("seal")
    s.add_argument("name")
    s.add_argument("--from", dest="src", required=True)
    c = sub.add_parser("check")
    c.add_argument("name")
    r = sub.add_parser("release")
    r.add_argument("name")
    r.add_argument(RELEASE_FLAG, dest="asserted", action="store_true")
    a = ap.parse_args(argv)
    if a.verb == "new-salt":
        print("salt written: %s" % new_salt(a.sealed))
        return 0
    if a.verb == "seal":
        key = json.loads(Path(a.src).read_text(encoding="utf-8"))
        r = seal(a.name, key, a.sealed, a.keys_dir)
        print("sealed %s: %d items, digest %s -> %s" % (a.name, r["items"], r["digest"][:12], r["digest_path"]))
        return 0
    if a.verb == "check":
        ok, want, got = check(a.name, a.sealed, a.keys_dir)
        print("%s: %s (committed %s, recomputed %s)" % (a.name, "MATCH" if ok else "MISMATCH", want, got))
        return 0 if ok else 1
    k, s_ = release(a.name, a.sealed, a.keys_dir, a.asserted)
    print("released %s and %s" % (k, s_))
    return 0


if __name__ == "__main__":
    sys.exit(main())
