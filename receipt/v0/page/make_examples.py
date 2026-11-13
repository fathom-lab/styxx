"""Writes the page's three example inputs and records the command line checker's answers on them.

    python make_examples.py --checker src/check_v0.js --utc <time from date -u> [--node node]

Example inputs go to examples/ (a file that already exists with other bytes is never replaced:
the script stops). The answers of `node check_v0.js check` and `node check_v0.js recheck` go to
examples/expected_cli_<utc>.json, which is never overwritten either.

The examples are written from RECEIPT_v0.md's worked examples (2.5, 7.3, 7.4, 7.6, 7.8, 12.1).
They are lab-written illustrations for a lab draft; nothing in them is a measurement.
Standard library only.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "examples")

# r1: the stdout of a test run, as a harness captured it. r2: a JSON file the harness read.
R1 = b"collected 296 items\n296 passed in 4.10s\n"
R2 = b'{"p95_ms": 240, "requests": 1200}\n'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def entry(rid, data, kind_of_source):
    return {
        "id": rid,
        "sha256": sha(data),
        "kind_of_source": kind_of_source,
        "captured_at": "example value: the harness's clock, a statement nothing checks",
        "complete": True,
        "bytes": base64.b64encode(data).decode("ascii"),
    }


def jcs_ascii(v):
    """Section 3's canonical text for this script's own values only: ASCII strings, small
    integers, booleans, null, lists and dicts. Python's json with sorted keys and no spaces
    gives the same text for that subset."""
    s = json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert all(ord(c) < 0x7F for c in s)
    return s


def manifest_bytes():
    core = {
        "spec": "sworn/manifest/0.2",
        "harness": "example-harness",
        "turn": "example-turn-1",
        "minted_at": "example value: the harness's clock, a statement nothing checks",
        "authored_sha256": [],
        "receipts": {
            "r1": entry("r1", R1, "tool_stdout"),
            "r2": entry("r2", R2, "file_read"),
        },
        "rung": "L2",
    }
    full = dict(core)
    full["digest"] = sha(jcs_ascii(core).encode("ascii"))
    return (json.dumps(full, indent=1, sort_keys=False, ensure_ascii=True) + "\n").encode("ascii")


R1_SHA = sha(R1)

DOCS = {
    "example-1-held.md": (
        "# example 1: every span held\n"
        "\n"
        "The test run ended with <sworn r=\"r1\" k=\"quote\">`296 passed in 4.10s`</sworn>.\n"
        "Latency at the 95th percentile was <sworn r=\"r2#/p95_ms\" k=\"numeric\">240 ms</sworn>.\n"
        "The run log is <sworn r=\"r1\" k=\"hash\">sha256 " + R1_SHA + "</sworn>.\n"
        "The log never says <sworn r=\"r1\" k=\"absent\">`failed`</sworn>.\n"
    ).encode("ascii"),
    "example-2-failed.md": (
        "# example 2: one span failed\n"
        "\n"
        "The test run ended with <sworn r=\"r1\" k=\"quote\">`296 passed in 4.10s`</sworn>.\n"
        "Latency at the 95th percentile was <sworn r=\"r2#/p95_ms\" k=\"numeric\">250 ms</sworn>.\n"
    ).encode("ascii"),
    "example-3-unresolved.md": (
        "# example 3: the receipt cannot be resolved\n"
        "\n"
        "The benchmark finished in <sworn r=\"r3\" k=\"numeric\">12 s</sworn>.\n"
        "The manifest has no entry r3, so nothing was compared.\n"
    ).encode("ascii"),
}

MANIFEST_NAME = "example-manifest.json"


def write_once(path, data):
    if os.path.exists(path):
        with open(path, "rb") as f:
            old = f.read()
        if old != data:
            sys.exit("refusing: %s exists with other bytes; examples are history, write new names" % path)
        return "unchanged"
    with open(path, "wb") as f:
        f.write(data)
    with open(path, "rb") as f:
        if f.read() != data:
            sys.exit("write did not land: %s" % path)
    return "written"


def shown(arg):
    """An argument as the record keeps it: a path relative to this script's folder, never the
    build machine's absolute path (which names its user and means nothing elsewhere)."""
    if not os.path.isabs(arg):
        return arg
    try:
        return os.path.relpath(arg, HERE).replace(os.sep, "/")
    except ValueError:            # another drive on Windows
        return os.path.basename(arg)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, timeout=60)
    return {"argv": [shown(x) for x in cmd[1:]], "exit": p.returncode, "stdout": p.stdout.decode("utf-8"),
            "stderr": p.stderr.decode("utf-8")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checker", required=True)
    ap.add_argument("--utc", required=True, help="the time, from date -u in the same command")
    ap.add_argument("--node", default="node")
    ap.add_argument("--receipts-dir", default="examples",
                    help="folder (relative to this script's folder) for the receipts the checker prints; "
                         "a newer checker writes them to a folder of their own, e.g. examples/receipts_<sha16>")
    a = ap.parse_args()
    os.makedirs(EX, exist_ok=True)
    checker = os.path.abspath(a.checker)
    with open(checker, "rb") as f:
        checker_sha = sha(f.read())
    man = manifest_bytes()
    status = {MANIFEST_NAME: write_once(os.path.join(EX, MANIFEST_NAME), man)}
    for name, data in DOCS.items():
        status[name] = write_once(os.path.join(EX, name), data)
    out = {
        "what": "answers of the command line checker on the page's three examples",
        "status": "lab-written examples for RECEIPT v0, an unsigned draft; the checker is lab-written",
        "written_at_utc": a.utc,
        "checker": {"path": shown(checker), "sha256": checker_sha},
        "node": subprocess.run([a.node, "--version"], capture_output=True, timeout=30).stdout.decode().strip(),
        "manifest": {"file": "examples/" + MANIFEST_NAME, "sha256": sha(man)},
        "files_status": status,
        "examples": [],
    }
    mpath = os.path.join(EX, MANIFEST_NAME)
    for name, data in DOCS.items():
        dpath = os.path.join(EX, name)
        chk = run([a.node, checker, "check", dpath, "--manifest", mpath, "--name", name])
        rec_rel = a.receipts_dir.strip("/").replace(os.sep, "/") + "/" + name + ".receipt.json"
        rec_path = os.path.join(HERE, rec_rel)
        os.makedirs(os.path.dirname(rec_path), exist_ok=True)
        status[rec_rel] = write_once(rec_path, chk["stdout"].encode("utf-8"))
        rck = run([a.node, checker, "recheck", rec_path, dpath, "--manifest", mpath])
        receipt = json.loads(chk["stdout"])
        core = receipt["core"]
        out["examples"].append({
            "name": name,
            "document_file": "examples/" + name,
            "document_sha256": sha(data),
            "check": chk,
            "check_stdout_sha256": sha(chk["stdout"].encode("utf-8")),
            "receipt_file": rec_rel,
            "digest": receipt["digest"],
            "document_verdict": core["document_verdict"],
            "rows": [{"verdict": r["verdict"], "reason": r["reason"], "kind": r["kind"],
                      "receipt": r["receipt"]} for r in core["spans"]],
            "recheck": rck,
        })
    target = os.path.join(EX, "expected_cli_%s.json" % a.utc.replace(":", ""))
    data = (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if os.path.exists(target):
        sys.exit("refusing to overwrite %s" % target)
    with open(target, "wb") as f:
        f.write(data)
    with open(target, "rb") as f:
        assert f.read() == data
    print(target)
    for e in out["examples"]:
        print(e["name"], e["document_verdict"], [(r["verdict"], r["reason"]) for r in e["rows"]],
              "check exit", e["check"]["exit"], "recheck", e["recheck"]["stdout"].strip(), e["recheck"]["exit"])


if __name__ == "__main__":
    main()
