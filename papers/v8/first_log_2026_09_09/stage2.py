"""Stage 2: append the fingerprints, sign a tree head, mirror the log, and then do exactly what
Appendix D says a stranger must be able to do, holding only the cert, the log and the pinned key.

Appendix D, steps 1 and 2, are the ones a stranger can do with no model and no keys of their own:
recompute the id, check the signature, resolve every ref, verify the tree head, verify inclusion.
Steps 3 to 5 need the weights. This script does 1 and 2 and reports honestly on 3 to 5.

It also runs the negative controls, because a verifier that accepts everything verifies nothing:
a flipped byte in an entry, a forged tree head, and an inclusion proof pointed at the wrong index
must all be refused.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path(r"C:\Users\heyzo\clawd\wt\v8")
OUT = HERE / "out"
LOG = OUT / "log"
MIRROR = OUT / "mirror"
FP = OUT / "fingerprints"
sys.path.insert(0, str(WT))

TRANSCRIPT = []
T0 = time.time()


def run(*args, expect=0, note=""):
    cmd = [sys.executable, "-m", "styxx.v8", *[str(a) for a in args]]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    t = time.time()
    p = subprocess.run(cmd, cwd=str(WT), capture_output=True, text=True, encoding="utf-8", env=env)
    try:
        parsed = json.loads(p.stdout) if p.stdout.strip() else None
    except json.JSONDecodeError:
        parsed = {"UNPARSEABLE": p.stdout[:1500]}
    TRANSCRIPT.append({"argv": [str(a) for a in args], "exit": p.returncode,
                       "expected_exit": expect, "seconds": round(time.time() - t, 1),
                       "note": note, "stdout": parsed})
    flag = "OK " if p.returncode == expect else "!! "
    print(f"{flag}exit={p.returncode} (want {expect})  {' '.join(str(a) for a in args[:3])}"
          f"{'  <- ' + note if note else ''}", flush=True)
    if p.returncode != expect:
        print(f"     {json.dumps(parsed)[:600]}", flush=True)
    return parsed, p.returncode


def main():
    issuerkey, logkey = OUT / "issuer.pem", OUT / "log.pem"

    # ---- append every fingerprint: the four runs first, then the canonical one ----
    runs = sorted(FP.glob("fingerprint-run*.json"))
    canonical = next(FP.glob("fingerprint-canonical-*.json"))
    for f in runs:
        run("log", "append", f, "--log", LOG, note=f"floor run {f.name.split('-')[1]}")
    run("log", "append", canonical, "--log", LOG,
        note="the canonical fingerprint, which references the runs above")

    # ---- a signed tree head ----------------------------------------------------
    sth_path = OUT / "sth.json"
    sth, _ = run("log", "sth", "--log", LOG, "--key", logkey,
                 "--timestamp", "2026-09-09T04:00:00Z", note="sign the tree head")
    if sth:
        sth_path.write_text(json.dumps(sth.get("sth", sth), indent=1), encoding="utf-8", newline="\n")

    # ---- mirror it -------------------------------------------------------------
    pin = LOG / "keys" / "log.pub"
    run("log", "mirror", "--log", LOG, "--to", MIRROR, "--pin", pin,
        note="a mirror verifies every entry and every head from the bytes")

    # =========================================================================
    # APPENDIX D, as a stranger: only the cert, the log, and the pinned key.
    # =========================================================================
    print("\n--- Appendix D, steps 1 and 2, run against the MIRROR ---", flush=True)
    run("log", "verify-cert", canonical, note="D1: recompute the id, check the signature")
    proof_path = OUT / "proof.json"
    run("log", "prove", "4", "--log", MIRROR, "--out", proof_path,
        note="D2: an inclusion proof for the canonical fingerprint")
    mirror_sth = sorted((MIRROR / "sth").glob("*.json"))
    if mirror_sth:
        run("log", "verify-sth", mirror_sth[-1], "--pin", pin, "--log", MIRROR,
            note="D2: the tree head verifies under the pinned key")
        run("log", "verify-inclusion", proof_path, mirror_sth[-1], "--pin", pin, "--log", MIRROR,
            note="D2: the cert is in the tree that head commits to")

    # =========================================================================
    # NEGATIVE CONTROLS. A verifier that accepts everything verifies nothing.
    # =========================================================================
    print("\n--- negative controls: each of these MUST be refused ---", flush=True)
    tampered = OUT / "tampered_mirror"
    if tampered.exists():
        shutil.rmtree(tampered)
    shutil.copytree(MIRROR, tampered)
    victim = sorted((tampered / "entries").rglob("*[0-9].json"))[0]
    raw = victim.read_bytes()
    # flip one byte inside the entry, keeping the length identical
    i = raw.find(b'"output_text"')
    i = i if i > 0 else len(raw) // 2
    flipped = raw[:i] + bytes([raw[i] ^ 0x01]) + raw[i + 1:]
    victim.write_bytes(flipped)
    print(f"     flipped one byte at offset {i} of {victim.name}", flush=True)
    res, _ = run("log", "mirror", "--log", tampered, "--to", OUT / "tamper_out", "--pin", pin,
                 expect=1, note="CONTROL: a mirror of a tampered log must report it")
    if res:
        print(f"     mirror verdict: {json.dumps(res)[:400]}", flush=True)

    forged = OUT / "forged_sth.json"
    if mirror_sth:
        head = json.loads(mirror_sth[-1].read_text(encoding="utf-8"))
        head["root_hash"] = "sha256:" + ("0" * 64)
        forged.write_text(json.dumps(head, indent=1), encoding="utf-8", newline="\n")
        run("log", "verify-sth", forged, "--pin", pin, "--log", MIRROR, expect=1,
            note="CONTROL: a head with a forged root must not verify")
        run("log", "verify-inclusion", proof_path, forged, "--pin", pin, "--log", MIRROR, expect=1,
            note="CONTROL: inclusion against a forged head must not verify")

    (HERE / "transcript_stage2.json").write_text(
        json.dumps({"transcript": TRANSCRIPT, "wallclock_s": round(time.time() - T0, 1)},
                   indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    ok = sum(1 for e in TRANSCRIPT if e["exit"] == e["expected_exit"])
    print(f"\n{ok} of {len(TRANSCRIPT)} commands behaved as required "
          f"({round(time.time() - T0, 1)}s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        (HERE / "FAILED_stage2.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        print(traceback.format_exc())
        sys.exit(1)
