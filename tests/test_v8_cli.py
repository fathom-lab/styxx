"""``python -m styxx.v8`` -- the command surface (contract section 9, spec section 11).

Every verb is driven the way a stranger drives it: a real ``subprocess`` running
``python -m styxx.v8`` in a temporary directory, with the mock runner.  Nothing here reaches
into ``cli.run`` for the exit-code assertions, because the two things under test -- the exit
code and the single JSON object on stdout -- are properties of the *process*, not of a function
call, and an in-process test would pin neither.

The two rules the contract fixes, checked on every single invocation by ``cli()`` below:

* stdout parses as exactly one JSON **object** (never a list, never a banner, never a
  traceback), and stderr is empty;
* the exit code is a value of ``consts.EXIT``.

The ladder the module fixture builds once, entirely through the CLI where the CLI can build it:

    key generate
      -> battery pool   (the root battery, signed)
      -> prereg noise-plan --runs 5 --nuisance order=...,batch_size=...  (on record before the runs)
      -> a sensitivity result cert (no verb mints one; the fixture does)
      -> fingerprint --runs 5 --plan --sensitivity   (floor 0.0 on every channel at batch 1)
      -> battery select (from a sweep record) -> a canary battery on a different item set
      -> fingerprint on the canary   (the not-comparable partner for the exit-3 case)
      -> log init | append x N | sth | prove | mirror
      -> the four log verify-* commands, run against the MIRROR

Numbers asserted here come from the mock's construction, with the arithmetic beside them.  At
``batch_size`` 1 the mock moves nothing under a reordering, so all five runs of the plan agree
and every channel floor is exactly 0.0; one drifted item out of twelve is an exact distance of
1/12 = 0.083333333 after ``ROUND_PLACES`` rounding.

Deviations from the contract text that this module had to adapt to, all of them the code's
behaviour rather than the prose's (the modules are built and passing; the prose is not):

1. **``verify --ref`` needs the battery cert resolvable.**  Section 9 lists ``[--log dir]`` only
   under ``--diff``.  ``verify.ref`` re-runs the recipe, which means it must read the battery's
   prompts, so it refuses with ``invalid`` (exit 4) when ``recipe.battery`` does not resolve.
   The CLI therefore accepts ``--log`` / ``--resolve`` / ``--battery`` on both modes, and a
   ``--ref`` without one of them is exit 4 by construction (pinned below).
2. **A resolver is all-or-nothing.**  ``verify._unresolved_refs`` returns ``[]`` when the
   resolver is ``None`` and otherwise requires *every* ref to resolve.  So passing ``--battery``
   alone to a cert that also carries ``noise_plan``/``run``/``sensitivity`` refs is exit 4, not a
   partial resolution.  ``--log`` after the ladder is appended is the resolvable case.
3. **``EXIT`` has no usage code.**  A malformed command line is ``invalid`` (4) and a named file
   that cannot be read is ``unavailable`` (5); there is no exit 64/2-style usage code, because
   inventing one would break the contract's "exit codes exactly ``EXIT``".
4. **``log mirror`` reports rather than raises**, so a tampered source mirrors with
   ``verified: false`` and the CLI turns that into exit 4 -- the command ran, the claim did not
   check out.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import keys
from styxx.v8 import sweep as SW
from styxx.v8.consts import EXIT
from styxx.v8.log import Log
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F

REPO_ROOT = Path(__file__).resolve().parents[1]

# The exit codes the contract allows a v8 command to end on. Every assertion in this file names
# one of these by key, never by literal, so a change to `consts.EXIT` fails loudly here.
ALLOWED_CODES = frozenset(EXIT.values())

MOCK_ENVIRONMENT = MockRunner().environment()

POOL_FAMILIES = ("recall", "short-reasoning", "instruction-following", "format")
POOL_SIZE = 12

# Items the mock moves when batch_size != 1 (section 4.2 delta-2). They carry flip2 > 0, so
# section 4.4 step 1 excludes them from any canary battery `battery select` builds.
NUISANCE_ITEMS = ("p00", "p05")

# The delta-1 family: one whole pool family responds to the precision change, so the selection
# keeps precision sensitivity whichever two of that family it takes.
DELTA1_PRECISION = "fp16"
PRECISION_FAMILY = "format"

DRIFT_ITEM = "p03"
RUNS = 5
N_CANARIES = 8
K_ANCHORS = 1
PERM_SEED = 11

# 1 of 12 items differs, and Appendix B's exact distance is the fraction of item-role items
# that differ: 1/12 = 0.0833333333... -> 0.083333333 at ROUND_PLACES = 9.
ONE_OF_TWELVE = 0.083333333


def pool_items() -> list[dict]:
    """The candidate pool: 12 items over 4 families, 3 per family."""
    return [
        {
            "item_id": f"p{k:02d}",
            "prompt_text": f"pool prompt {k}",
            "family": POOL_FAMILIES[k % len(POOL_FAMILIES)],
        }
        for k in range(POOL_SIZE)
    ]


PRECISION_ITEMS = tuple(
    item["item_id"] for item in pool_items() if item["family"] == PRECISION_FAMILY
)


# --------------------------------------------------------------------------- the subprocess


def _child_env() -> dict:
    """The child's environment: the repo on the path, UTF-8 in and out, no bytecode litter.

    ``HOME``/``USERPROFILE`` are left alone deliberately -- every test passes ``--key`` and
    ``--out`` explicitly, so the CLI's ``~/.styxx`` default is never written to.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.fspath(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


CHILD_ENV = _child_env()


def cli(*args) -> tuple[int, dict]:
    """Run ``python -m styxx.v8 <args>`` and return ``(exit_code, payload)``.

    Asserts the two contract rules on every call, so no individual test has to repeat them.
    """
    argv = [sys.executable, "-m", "styxx.v8"] + [os.fspath(a) if isinstance(a, Path) else str(a) for a in args]
    proc = subprocess.run(
        argv,
        cwd=os.fspath(REPO_ROOT),
        env=CHILD_ENV,
        capture_output=True,
        timeout=600,
    )
    stderr = proc.stderr.decode("utf-8", "replace")
    assert stderr == "", f"{argv[3:]} wrote to stderr:\n{stderr}"
    stdout = proc.stdout.decode("utf-8")
    try:
        payload = json.loads(stdout)
    except ValueError as exc:  # pragma: no cover -- the failure message is the point
        raise AssertionError(f"{argv[3:]} did not print one JSON value: {exc}\n{stdout!r}") from None
    assert isinstance(payload, dict), f"{argv[3:]} printed a {type(payload).__name__}, not an object"
    assert proc.returncode in ALLOWED_CODES, (
        f"{argv[3:]} exited {proc.returncode}, which is not a value of consts.EXIT"
    )
    return proc.returncode, payload


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.fspath(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return path


def only_cert(directory: Path, prefix: str) -> Path:
    hits = sorted(p for p in directory.glob(f"{prefix}*.json"))
    assert len(hits) == 1, f"expected one {prefix}* under {directory}, got {[p.name for p in hits]}"
    return hits[0]


# --------------------------------------------------------------------------- the ladder


@pytest.fixture(scope="module")
def lab(tmp_path_factory) -> dict:
    """Build the whole ladder once, through the CLI, and hand every path to the tests."""
    root = tmp_path_factory.mktemp("v8cli")
    out: dict = {"root": root}

    # The issuer key. `key generate` makes a fresh one, but the fixture certs below have to be
    # signed by the SAME issuer for the log roster to take both, so the ladder's working key is
    # the deterministic fixture key written out as a PEM. `key generate` is exercised on its own.
    key_pem = root / "issuer.pem"
    seed, public = F.keypair("issuer")
    keys.save_private_pem(seed, key_pem)
    keys.save_public(public, root / "issuer.pem.pub")
    out["key"] = key_pem
    out["public"] = F.public_key("issuer")

    subject = F.weights_subject(environment=copy.deepcopy(MOCK_ENVIRONMENT))
    out["subject"] = subject
    certs = root / "certs"

    items = write_json(root / "pool_items.json", pool_items())
    out["items"] = items

    # 1. the pool battery -- a root cert: empty recipe, no refs (section 4.5).
    root_spec = write_json(root / "spec_root.json", {"subject": subject, "recipe": {}})
    out["root_spec"] = root_spec
    code, payload = cli(
        "battery", "pool", "--source", items, "--key", key_pem, "--subject", root_spec,
        "--created", F.CREATED, "--out", certs / "pool.json",
    )
    assert code == 0, payload
    out["pool"] = certs / "pool.json"
    out["pool_id"] = payload["id"]

    # 2. the nuisance plan, minted by `prereg noise-plan` and on record BEFORE the runs
    #    (section 5.1 step 1). The pool battery is already built, so the plan can name it.
    pool_spec = write_json(
        root / "spec_plan.json",
        {"subject": subject, "recipe": F.recipe(battery=out["pool_id"])},
    )
    code, payload = cli(
        "prereg", "noise-plan", "--runs", RUNS,
        "--nuisance", "order=a3|permuted", "--nuisance", "batch_size=1|4",
        "--subject", pool_spec, "--battery", out["pool"], "--key", key_pem,
        "--created", F.CREATED, "--out", certs / "noise_plan.json",
    )
    assert code == 0, payload
    out["plan"] = certs / "noise_plan.json"
    out["plan_id"] = payload["id"]
    out["plan_cert"] = payload["cert"]

    # The sensitivity receipt is still hand-built: no verb mints one (section 5.3).
    sensitivity = F.make_cert("result", body={"kind": "sensitivity", "deviations": []}, refs=[])
    write_json(certs / "sensitivity.json", sensitivity)
    out["sensitivity"] = certs / "sensitivity.json"
    out["sensitivity_id"] = sensitivity["id"]

    # 3. the canonical fingerprint on the pool: five runs of one plan, floor attached.
    spec = write_json(
        root / "spec.json",
        {"subject": subject, "recipe": F.recipe(battery=out["pool_id"])},
    )
    out["spec"] = spec
    fp_dir = root / "fp"
    code, payload = cli(
        "fingerprint", "--subject", spec, "--battery", out["pool"], "--key", key_pem,
        "--runs", RUNS, "--plan", out["plan"], "--sensitivity", out["sensitivity_id"],
        "--created", F.CREATED, "--out", fp_dir,
    )
    assert code == 0, payload
    out["fingerprint"] = only_cert(fp_dir, "fingerprint-canonical-")
    out["fingerprint_id"] = payload["id"]
    out["run_ids"] = list(payload["run_ids"])
    # Section 5.1 step 5: R-1 non-canonical run certs, then the canonical carrying the floor.
    out["run_certs"] = [only_cert(fp_dir, f"fingerprint-run{k}-") for k in range(1, RUNS)]
    out["floor"] = dict(payload["floor"])
    out["fingerprint_payload"] = payload

    # 4. the drifted twin: same subject, same plan, one item moved (section 6's drift leg).
    drift_dir = root / "fp_drift"
    code, payload = cli(
        "fingerprint", "--subject", spec, "--battery", out["pool"], "--key", key_pem,
        "--runs", RUNS, "--plan", out["plan"], "--sensitivity", out["sensitivity_id"],
        "--drift", DRIFT_ITEM, "--created", F.CREATED, "--out", drift_dir,
    )
    assert code == 0, payload
    out["drifted"] = only_cert(drift_dir, "fingerprint-canonical-")
    out["drifted_id"] = payload["id"]
    # The drifted twin is never appended (the log is the reference ladder), so a --diff against
    # it needs its own run certs on the resolver beside the log -- deviation 2 in the docstring.
    out["drift_dir"] = drift_dir

    # 5. a sweep record and the canary battery `battery select` builds from it. The record is
    #    built in-process: no verb runs a sweep, the CLI consumes the record a harness wrote.
    runner = MockRunner(
        nuisance_items=set(NUISANCE_ITEMS),
        precision_items={DELTA1_PRECISION: set(PRECISION_ITEMS)},
    )
    record = SW.run_sweep(
        lambda precision: runner,
        pool_items(),
        F.recipe(battery=out["pool_id"]),
        subject,
        delta1=[DELTA1_PRECISION],
        delta2=[{"batch_size": 8}, {"batch_size": 8, "order": "perm", "perm_seed": PERM_SEED}],
    )
    out["sweep"] = SW.write_record(record, root / "sweep.json")

    canary_spec = write_json(
        root / "spec_canary.json",
        {"subject": subject, "recipe": F.recipe(battery=out["pool_id"])},
    )
    out["canary_spec"] = canary_spec
    code, payload = cli(
        "battery", "select", "--sweep", out["sweep"], "--pool", items,
        "--n", N_CANARIES, "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
        "--key", key_pem, "--subject", canary_spec,
        "--selected-against", out["fingerprint_id"], "--created", F.CREATED,
        "--out", certs / "canary.json",
    )
    assert code == 0, payload
    out["canary"] = certs / "canary.json"
    out["canary_id"] = payload["id"]
    out["canary_body"] = payload["body"]

    # 6. a fingerprint on the canary battery: a different `recipe.battery`, which is what makes
    #    it not comparable to the pool fingerprint (section 2.3 -> exit 3).
    canary_fp_spec = write_json(
        root / "spec_canary_fp.json",
        {"subject": subject, "recipe": F.recipe(battery=out["canary_id"])},
    )
    canary_fp_dir = root / "fp_canary"
    code, payload = cli(
        "fingerprint", "--subject", canary_fp_spec, "--battery", out["canary"],
        "--key", key_pem, "--runs", 1, "--created", F.CREATED, "--out", canary_fp_dir,
    )
    assert code == 0, payload
    out["canary_fingerprint"] = only_cert(canary_fp_dir, "fingerprint-canonical-")

    # 7. the log: init, then append in dependency order.
    log_dir = root / "log"
    code, payload = cli(
        "log", "init", "--log", log_dir, "--key", key_pem, "--issuer", f"lab={out['public']}"
    )
    assert code == 0, payload
    out["log"] = log_dir
    out["log_id"] = payload["log_id"]

    order = (
        [out["pool"], out["plan"], out["sensitivity"]]
        + out["run_certs"]
        + [out["fingerprint"], out["canary"], out["canary_fingerprint"]]
    )
    appended = []
    for path in order:
        code, payload = cli("log", "append", path, "--log", log_dir)
        assert code == 0, (path.name, payload)
        appended.append(payload["index"])
        if len(appended) == 1:
            # A head at tree_size 1, signed before the rest arrives. An STH is per tree size and
            # the log refuses a second one for a size it already published, so the small head has
            # to be taken here rather than reconstructed afterwards.
            code, small = cli(
                "log", "sth", "--log", log_dir, "--key", key_pem,
                "--timestamp", "2026-09-08T18:30:00Z", "--out", root / "sth_small.json",
            )
            assert code == 0, small
            out["sth_small"] = root / "sth_small.json"
            out["sth_small_body"] = small["sth"]
    out["appended"] = appended
    out["size"] = len(order)

    # 8. the full tree head, so consistency has a pair to check.
    code, payload = cli(
        "log", "sth", "--log", log_dir, "--key", key_pem,
        "--timestamp", "2026-09-08T19:00:00Z", "--out", root / "sth_full.json",
    )
    assert code == 0, payload
    out["sth"] = root / "sth_full.json"
    out["sth_body"] = payload["sth"]

    code, payload = cli("log", "prove", 0, "--log", log_dir, "--out", root / "proof0.json")
    assert code == 0, payload
    out["proof"] = root / "proof0.json"

    # 9. the mirror the four verify-* commands are run against.
    mirror_dir = root / "mirror"
    code, payload = cli("log", "mirror", "--log", log_dir, "--to", mirror_dir)
    assert code == 0, payload
    assert payload["report"]["verified"] is True
    out["mirror"] = mirror_dir
    out["mirror_report"] = payload["report"]
    return out


# --------------------------------------------------------------------------- the shape of the surface


def test_the_module_runs_and_prints_one_json_object_with_no_arguments():
    code, payload = cli()
    assert code == EXIT["invalid"]
    assert payload["error"].startswith("no verb")
    assert "battery" in payload["usage"][0] or any("battery" in line for line in payload["usage"])


def test_help_and_version_are_json_and_exit_zero():
    code, payload = cli("help")
    assert code == 0
    assert payload["verbs"] == ["battery", "fingerprint", "key", "log", "prereg", "verify"]
    code, payload = cli("--version")
    assert code == 0
    assert payload["styxx"] == "8.0"


def test_an_unknown_verb_and_an_unknown_option_are_invalid_not_a_traceback():
    code, payload = cli("frobnicate")
    assert code == EXIT["invalid"]
    assert "unknown verb" in payload["error"]
    code, payload = cli("key", "show", "--nonesuch", "x")
    assert code == EXIT["invalid"]
    assert payload["error"] == "unknown option --nonesuch"
    assert payload["verb"] == "key"


def test_every_exit_code_the_cli_can_produce_is_a_value_of_the_exit_table():
    # The contract's "exit codes exactly EXIT": the CLI has no usage code of its own.
    assert ALLOWED_CODES == {0, 1, 2, 3, 4, 5}
    assert EXIT["same"] == 0 and EXIT["invalid"] == 4 and EXIT["unavailable"] == 5


# --------------------------------------------------------------------------- key


def test_key_generate_writes_a_pem_and_a_pub_that_key_show_reads_back(tmp_path):
    pem = tmp_path / "k.pem"
    code, payload = cli("key", "generate", "--out", pem)
    assert code == 0
    assert Path(payload["private_key_path"]) == pem and pem.is_file()
    pub_path = Path(payload["public_key_path"])
    assert pub_path.is_file()

    code, shown = cli("key", "show", "--key", pem)
    assert code == 0 and shown["public"] == payload["public"]
    code, shown = cli("key", "show", "--pub", pub_path)
    assert code == 0 and shown["public"] == payload["public"]


def test_key_generate_refuses_to_overwrite_without_force_and_obeys_it_with(tmp_path):
    pem = tmp_path / "k.pem"
    code, original = cli("key", "generate", "--out", pem)
    assert code == 0
    code, payload = cli("key", "generate", "--out", pem)
    assert code == EXIT["invalid"]
    assert "key generate" in payload["error"]
    code, replacement = cli("key", "generate", "--out", pem, "--force")
    assert code == 0 and replacement["public"] != original["public"]


def test_key_show_on_a_missing_file_is_unavailable_not_invalid(tmp_path):
    code, payload = cli("key", "show", "--key", tmp_path / "absent.pem")
    assert code == EXIT["unavailable"]
    assert "no such file" in payload["error"]


def test_key_needs_a_subcommand():
    code, payload = cli("key")
    assert code == EXIT["invalid"]
    assert "expected 'generate' or 'show'" in payload["error"]


# --------------------------------------------------------------------------- battery


def test_battery_pool_builds_a_root_body_and_signs_it_when_given_a_key(lab):
    code, payload = cli("battery", "pool", "--source", lab["items"])
    assert code == 0
    assert payload["kind"] == "pool-v1" and payload["items"] == POOL_SIZE
    assert payload["validate"] == []
    assert "cert" not in payload  # no --key, no cert

    code, signed = cli("log", "verify-cert", lab["pool"])
    assert code == 0 and signed["ok"] is True and signed["type"] == "battery"
    assert signed["id"] == lab["pool_id"]


def test_battery_fixed_takes_a_source_label_and_marks_every_role_item(lab, tmp_path):
    code, payload = cli(
        "battery", "fixed", "--source", lab["items"], "--label", "styxx-cli@v8",
        "--out", tmp_path / "fixed.json",
    )
    assert code == 0
    assert payload["kind"] == "fixed-v1" and payload["items"] == POOL_SIZE
    assert {item["role"] for item in payload["body"]["items"]} == {"item"}
    assert payload["body"]["source"] == "styxx-cli@v8"
    assert json.loads(Path(payload["written"]).read_text(encoding="utf-8"))["kind"] == "fixed-v1"


def test_battery_select_excludes_the_nuisance_items_and_reports_its_sensitivity(lab):
    body = lab["canary_body"]
    assert body["kind"] == "canary-v1"
    excluded = {entry["item_id"] if isinstance(entry, dict) else entry for entry in body["excluded"]}
    # The mock moves exactly these two under batch 8, so flip2 > 0 for them and only them.
    assert set(NUISANCE_ITEMS) <= excluded
    chosen = {item["item_id"] for item in body["items"]}
    assert chosen.isdisjoint(NUISANCE_ITEMS)
    assert len(chosen) == N_CANARIES
    assert {item["role"] for item in body["items"]} == {"canary"}
    # zero(i) needs exp(stay) == 1, and the mock's log-prob gaps never reach it, so the anchor
    # supply is empty and `k` is a request the selection could not fill.
    assert body["params"]["k_anchors"] == K_ANCHORS
    assert body["params"]["k_anchors_actual"] == 0
    # Section 4.6: the battery keeps precision sensitivity. The delta-1 family is `format`, and
    # the share of selected items still in it is what `sensitivity_after_exclusion` reports.
    kept = chosen & set(PRECISION_ITEMS)
    assert kept, "the selection dropped every precision-sensitive item"
    assert body["params"]["sensitivity_after_exclusion"] == round(len(kept) / N_CANARIES, 9)


def test_battery_select_takes_the_pool_cert_itself_as_its_pool(lab):
    # The contract spells it `--pool <pool cert>`; a bare items file works too, and the two
    # spellings have to agree, because the cert's body IS that item list.
    from_cert = cli(
        "battery", "select", "--sweep", lab["sweep"], "--pool", lab["pool"],
        "--n", N_CANARIES, "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
    )
    from_items = cli(
        "battery", "select", "--sweep", lab["sweep"], "--pool", lab["items"],
        "--n", N_CANARIES, "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
    )
    assert from_cert[0] == 0 and from_items[0] == 0
    assert [i["item_id"] for i in from_cert[1]["body"]["items"]] == \
           [i["item_id"] for i in from_items[1]["body"]["items"]]


def test_battery_select_is_deterministic_across_two_processes(lab, tmp_path):
    args = (
        "battery", "select", "--sweep", lab["sweep"], "--pool", lab["items"],
        "--n", N_CANARIES, "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
    )
    code_a, one = cli(*args)
    code_b, two = cli(*args)
    assert code_a == 0 and code_b == 0
    assert one["body"] == two["body"]


def test_battery_select_refuses_a_canary_cert_without_a_selected_against_ref(lab):
    code, payload = cli(
        "battery", "select", "--sweep", lab["sweep"], "--pool", lab["items"],
        "--n", N_CANARIES, "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
        "--key", lab["key"], "--subject", lab["canary_spec"],
    )
    assert code == EXIT["invalid"]
    assert "selected-against" in payload["error"]


def test_battery_select_rejects_a_non_integer_n_and_a_missing_sweep(lab, tmp_path):
    code, payload = cli(
        "battery", "select", "--sweep", lab["sweep"], "--pool", lab["items"],
        "--n", "eight", "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
    )
    assert code == EXIT["invalid"]
    assert "--n must be an integer" in payload["error"]

    code, payload = cli(
        "battery", "select", "--sweep", tmp_path / "absent.json", "--pool", lab["items"],
        "--n", N_CANARIES, "--k", K_ANCHORS, "--perm-seed", PERM_SEED,
    )
    assert code == EXIT["unavailable"]


def test_battery_needs_a_known_subcommand():
    code, payload = cli("battery", "sprinkle")
    assert code == EXIT["invalid"]
    assert "expected 'select', 'fixed' or 'pool'" in payload["error"]


# --------------------------------------------------------------------------- prereg


def test_prereg_noise_plan_names_r_the_factors_and_the_environment(lab):
    """Section 5.1 step 1: the plan fixes the run count, the enumerated nuisance factors with
    their values, and the environment, and it is on record before any floor run."""
    plan = lab["plan_cert"]
    assert plan["type"] == "prereg"
    body = plan["body"]
    assert body["kind"] == "noise-plan"
    assert body["runs"] == RUNS
    assert body["nuisance"] == [
        {"factor": "batch_size", "values": ["1", "4"]},
        {"factor": "order", "values": ["a3", "permuted"]},
    ]
    assert body["environment"] == MOCK_ENVIRONMENT
    # section 2.7: a noise-plan prereg carries subject and recipe -- it names the recipe the R
    # runs execute -- and the battery its recipe embeds is a ref, so a log resolves it.
    assert plan["subject"]["kind"] == "weights"
    assert plan["recipe"]["battery"] == lab["pool_id"]
    assert plan["refs"] == [{"role": "battery", "id": lab["pool_id"]}]
    code, payload = cli("log", "verify-cert", lab["plan"])
    assert code == 0 and payload["ok"] is True and payload["type"] == "prereg"


def test_prereg_noise_plan_derives_covers_and_not_covered_from_what_it_varied(lab):
    """Section 5.4: ``covers`` is the nuisance factors the plan varied and ``not_covered`` the
    environment fields it held fixed, so the second is every environment leaf that is not the
    first -- derived, because a field nobody listed would otherwise read as covered."""
    body = lab["plan_cert"]["body"]
    assert body["covers"] == ["batch_size", "order"]
    assert body["not_covered"] == [
        "hardware.count", "hardware.driver", "hardware.gpu",
        "runtime.backend", "runtime.framework", "runtime.version",
    ]
    # and the floor the runs produced carries the plan's two lists, not the CLI's defaults
    canonical = json.loads(lab["fingerprint"].read_bytes().decode("utf-8"))
    block = canonical["body"]["noise_floor"]
    assert block["covers"] == body["covers"]
    assert block["not_covered"] == body["not_covered"]


def test_prereg_needs_a_subcommand_and_the_only_one_is_noise_plan():
    code, payload = cli("prereg")
    assert code == EXIT["invalid"]
    assert "expected 'noise-plan'" in payload["error"]
    code, payload = cli("prereg", "study")
    assert code == EXIT["invalid"]
    assert "expected 'noise-plan'" in payload["error"]


def test_prereg_refuses_fewer_than_five_runs(lab, tmp_path):
    code, payload = cli(
        "prereg", "noise-plan", "--runs", 4, "--nuisance", "order=a3|permuted",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--created", F.CREATED, "--out", tmp_path / "plan.json",
    )
    assert code == EXIT["invalid"]
    assert "R = 5 minimum" in payload["error"]


def test_prereg_refuses_a_plan_that_enumerates_no_factor(lab):
    code, payload = cli(
        "prereg", "noise-plan", "--runs", RUNS,
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
    )
    assert code == EXIT["invalid"]
    assert "--nuisance is required" in payload["error"]

    code, payload = cli(
        "prereg", "noise-plan", "--runs", RUNS, "--nuisance", "order",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
    )
    assert code == EXIT["invalid"]
    assert "is not 'factor=value" in payload["error"]


def test_a_hand_written_noise_plan_below_five_runs_does_not_check_out(lab, tmp_path):
    """The R = 5 minimum is in schema/prereg.json's noise-plan branch as well as in the verb, so
    a plan that did not come from this CLI is refused by ``log verify-cert`` and at append."""
    thin = F.make_cert(
        "prereg",
        body={
            "kind": "noise-plan",
            "runs": 4,
            "nuisance": [{"factor": "order", "values": ["a3", "permuted"]}],
            "environment": copy.deepcopy(MOCK_ENVIRONMENT),
        },
        refs=[],
    )
    path = write_json(tmp_path / "thin_plan.json", thin)
    code, payload = cli("log", "verify-cert", path)
    assert code == EXIT["invalid"]
    assert payload["ok"] is False
    assert any("runs" in reason for reason in payload["reasons"]), payload["reasons"]


def test_prereg_takes_several_factors_and_orders_them(lab, tmp_path):
    code, payload = cli(
        "prereg", "noise-plan", "--runs", 8,
        "--nuisance", "order=a3|permuted", "--nuisance", "batch_size=1|8",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--created", F.CREATED, "--out", tmp_path / "plan.json",
    )
    assert code == 0, payload
    assert payload["nuisance"] == [
        {"factor": "batch_size", "values": ["1", "8"]},
        {"factor": "order", "values": ["a3", "permuted"]},
    ]
    assert payload["covers"] == ["batch_size", "order"]
    assert payload["runs"] == 8
    assert Path(payload["written"]).is_file()


def test_prereg_refuses_a_repeated_factor(lab):
    code, payload = cli(
        "prereg", "noise-plan", "--runs", RUNS, "--nuisance", "order=a3,order=permuted",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
    )
    assert code == EXIT["invalid"]
    assert "is given more than once" in payload["error"]


def test_prereg_refuses_a_subject_with_no_environment_to_name(lab, tmp_path):
    bare = write_json(
        tmp_path / "spec_bare.json",
        {"subject": F.weights_subject(environment={}), "recipe": F.recipe(battery=lab["pool_id"])},
    )
    code, payload = cli(
        "prereg", "noise-plan", "--runs", RUNS, "--nuisance", "order=a3|permuted",
        "--subject", bare, "--battery", lab["pool"], "--key", lab["key"],
    )
    assert code == EXIT["invalid"]
    assert "no 'environment' block to name" in payload["error"]


def test_prereg_needs_a_window_for_an_alias_subject(lab, tmp_path):
    """Sections 2.2 and 5.1: an alias floor carries a mandatory window, because a provider may
    change the model during the R-run window and the cert has to say which window that was."""
    alias = write_json(
        tmp_path / "spec_alias.json",
        {
            "subject": F.alias_subject(environment=copy.deepcopy(MOCK_ENVIRONMENT)),
            "recipe": F.recipe(battery=lab["pool_id"]),
        },
    )
    args = [
        "prereg", "noise-plan", "--runs", RUNS, "--nuisance", "region=us|eu",
        "--subject", alias, "--battery", lab["pool"], "--key", lab["key"],
        "--created", F.CREATED,
    ]
    code, payload = cli(*args)
    assert code == EXIT["invalid"]
    assert "--window" in payload["error"]

    code, payload = cli(*args, "--window", "2026-09-09T00:00:00Z,2026-09-09T06:00:00Z")
    assert code == 0, payload
    assert payload["cert"]["body"]["window"] == {
        "start": "2026-09-09T00:00:00Z",
        "end": "2026-09-09T06:00:00Z",
    }


# --------------------------------------------------------------------------- fingerprint


def test_fingerprint_without_a_plan_cannot_make_a_floor(lab):
    """F1 of `papers/v8/first_log_2026_09_09/`, from the other side: the runs of a floor must all be under a
    logged plan (section 5.1 step 1), so ``--runs R`` with no plan is refused rather than
    quietly producing R independent fingerprints."""
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 5, "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "is not a floor" in payload["error"]
    assert "prereg noise-plan" in payload["error"]


def test_fingerprint_refuses_a_run_count_the_plan_did_not_fix(lab):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 6, "--plan", lab["plan"], "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "is not the R the plan fixed" in payload["error"]


def test_fingerprint_refuses_coverage_flags_beside_a_plan_cert(lab):
    """The plan fixed ``covers`` and ``not_covered`` before the runs (sections 5.1, 5.4), so the
    command line cannot restate them after seeing what the runs did."""
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", RUNS, "--plan", lab["plan"],
        "--not-covered", "hardware.gpu", "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "the plan fixed the coverage before the runs" in payload["error"]


def test_fingerprint_refuses_a_plan_file_that_is_not_a_noise_plan(lab):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", RUNS, "--plan", lab["sensitivity"],
        "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "not a prereg" in payload["error"]


def test_fingerprint_refuses_a_plan_that_fixes_nothing(lab, tmp_path):
    """A hand-written prereg of kind noise-plan that names no R, no factors and no environment
    is not a plan: it fixes nothing before the runs, which is the whole point of section 5.1."""
    empty = F.make_cert("prereg", body={"kind": "noise-plan"}, refs=[])
    path = write_json(tmp_path / "empty_plan.json", empty)
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", RUNS, "--plan", path, "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "does not name runs, nuisance, environment" in payload["error"]


def test_fingerprint_attaches_a_zero_floor_from_five_batch_one_runs(lab):
    # At batch_size 1 the mock is order-invariant, so all five runs of the plan agree and every
    # pairwise distance is 0.0 -- the floor is the maximum of them.
    assert lab["floor"] == {"exact": 0.0, "seqlp": 0.0, "topk": 0.0}
    # Section 5.1 step 5 logs R-1 run certs and then the canonical, which IS run 0: R-1 ids,
    # all distinct, and none of them the canonical's own id (a cert cannot reference itself).
    assert len(lab["run_ids"]) == RUNS - 1 == len(set(lab["run_ids"]))
    assert lab["fingerprint_id"] not in lab["run_ids"]
    code, payload = cli("log", "verify-cert", lab["fingerprint"])
    assert code == 0 and payload["ok"] is True and payload["type"] == "fingerprint"


def test_the_runs_of_a_plan_take_the_plan(lab, tmp_path):
    """The defect of `papers/v8/vacuous_floor_2026_09_09/`, driven the way it was driven.

    A plan declaring ``batch_size 1|8|32`` and ``item_order canonical|perm11|perm12`` produced
    five runs that all carried ``body.nuisance.batch_size == 1``; only the order hash moved, and
    at batch 1 item order cannot move anything because each item is its own forward pass. Every
    pairwise distance was 0.0, the floor was 0.0 on exact, seqlp and topk, and a later
    ``verify --diff`` reported ``exceeds_floor`` on every channel -- against a zero floor, every
    difference exceeds. The cert, the plan, the log and the proofs were all valid.

    Now the runs take the assignments, the certs record them, and the floor is measured across
    the batch sizes the plan named.
    """
    code, plan = cli(
        "prereg", "noise-plan", "--runs", RUNS,
        "--nuisance", "batch_size=1|8|32",
        "--nuisance", "item_order=canonical|perm11|perm12",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--created", F.CREATED, "--out", tmp_path / "plan.json",
    )
    assert code == 0, plan

    out = tmp_path / "fp"
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runs", RUNS, "--plan", tmp_path / "plan.json",
        "--nuisance", ",".join(NUISANCE_ITEMS),  # the mock's batch-sensitive items
        "--created", F.CREATED, "--out", out,
    )
    assert code == 0, payload

    # every run takes a different assignment, and every declared factor moves
    assert payload["assignments"] == [
        {"batch_size": "1", "item_order": "canonical"},
        {"batch_size": "8", "item_order": "perm11"},
        {"batch_size": "32", "item_order": "perm12"},
        {"batch_size": "1", "item_order": "perm11"},
        {"batch_size": "1", "item_order": "perm12"},
    ]

    certs = [json.loads(p.read_bytes().decode("utf-8")) for p in sorted(out.glob("*.json"))]
    nuisances = {c["body"]["run_index"]: c["body"]["nuisance"] for c in certs}
    assert [nuisances[k]["batch_size"] for k in range(RUNS)] == [1, 8, 32, 1, 1]
    assert [nuisances[k]["item_order"] for k in range(RUNS)] == [
        "canonical", "perm11", "perm12", "perm11", "perm12"
    ]
    assert len({nuisances[k]["item_order_sha256"] for k in range(RUNS)}) == 3
    # each cert records the execution it ran under, not the one the command line carried
    recipes = {c["body"]["run_index"]: c["recipe"]["decoding"]["batch_size"] for c in certs}
    assert [recipes[k] for k in range(RUNS)] == [1, 8, 32, 1, 1]

    # 2 of 12 items move under batching -> a floor with something in it (1/6 = 0.166666667)
    assert payload["floor"]["exact"] == pytest.approx(2 / 12, abs=5e-10)
    assert all(value > 0.0 for value in payload["floor"].values())


def test_a_floor_the_runs_honoured_appends_and_one_they_ignored_does_not(lab, tmp_path):
    """The two halves of the repair meeting: the CLI applies the plan, the log checks that it
    was applied, and a hand-edited floor set that pins one factor is refused by name."""
    code, plan = cli(
        "prereg", "noise-plan", "--runs", RUNS,
        "--nuisance", "batch_size=1|8|32",
        "--nuisance", "item_order=canonical|perm11|perm12",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--created", F.CREATED, "--out", tmp_path / "plan.json",
    )
    assert code == 0, plan
    out = tmp_path / "fp"
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runs", RUNS, "--plan", tmp_path / "plan.json",
        "--nuisance", ",".join(NUISANCE_ITEMS), "--created", F.CREATED, "--out", out,
    )
    assert code == 0, payload

    log_dir = tmp_path / "log"
    code, _ = cli("log", "init", "--log", log_dir, "--key", lab["key"],
                  "--issuer", f"lab={lab['public']}")
    assert code == 0
    order = [lab["pool"], tmp_path / "plan.json"]
    order += [only_cert(out, f"fingerprint-run{k}-") for k in range(1, RUNS)]
    order += [only_cert(out, "fingerprint-canonical-")]
    for path in order:
        code, appended = cli("log", "append", path, "--log", log_dir)
        assert code == 0, (path.name, appended)

    # The same floor with every run pinned back to batch 1 -- the shape the first real run
    # produced. Re-signed, so it is a valid cert set and the floor check is the only thing that
    # can refuse it.
    seed, _public = F.keypair("issuer")
    pinned = tmp_path / "pinned"
    remap: dict[str, str] = {}
    for path in order:
        cert = json.loads(path.read_bytes().decode("utf-8"))
        core = {k: v for k, v in cert.items() if k not in ("id", "sig")}
        core["refs"] = [
            {**ref, "id": remap.get(ref["id"], ref["id"])} for ref in core.get("refs", [])
        ]
        if cert["type"] == "fingerprint":
            core["body"]["nuisance"]["batch_size"] = 1
            core["recipe"]["decoding"]["batch_size"] = 1
            block = core["body"].get("noise_floor")
            if block:
                block["runs"] = [remap.get(rid, rid) for rid in block["runs"]]
        signed = certmod.sign(core, seed)
        remap[cert["id"]] = signed["id"]
        write_json(pinned / path.name, signed)

    log2 = tmp_path / "log2"
    code, _ = cli("log", "init", "--log", log2, "--key", lab["key"],
                  "--issuer", f"lab={lab['public']}")
    assert code == 0
    for path in [pinned / p.name for p in order[:-1]]:
        code, appended = cli("log", "append", path, "--log", log2)
        assert code == 0, (path.name, appended)  # the plan and the run certs are fine
    code, refused = cli("log", "append", pinned / order[-1].name, "--log", log2)
    assert code == EXIT["invalid"]
    assert "floor:" in refused["error"] and "'batch_size'" in refused["error"]
    assert "all 5 runs ran at '1'" in refused["error"]


def test_a_plan_naming_a_factor_the_runner_cannot_set_is_refused_at_mint(lab, tmp_path):
    """Section 5.1 step 2 lets a plan name the physical GPU and driver; this process cannot set
    them. Refusing at mint is the alternative to R runs at the default under a plan that said
    otherwise -- which is what produced a valid cert for a floor that measured nothing."""
    code, plan = cli(
        "prereg", "noise-plan", "--runs", RUNS, "--nuisance", "gpu=a100|h100",
        "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--created", F.CREATED, "--out", tmp_path / "gpu_plan.json",
    )
    assert code == 0, plan
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runs", RUNS, "--plan", tmp_path / "gpu_plan.json", "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "cannot apply nuisance factor 'gpu'" in payload["error"]


def test_a_multi_run_floor_needs_the_plan_cert_not_a_bare_id(lab):
    """A cert id names a plan this process cannot read, so it cannot apply one. The old code
    accepted it and ran R times at the default."""
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runs", RUNS, "--plan", lab["plan_id"], "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "needs the plan CERT" in payload["error"]


def test_the_canonical_fingerprint_is_not_byte_identical_to_run_zero(lab):
    """F2 of `papers/v8/first_log_2026_09_09/`: ``--runs 5`` wrote a canonical cert byte-identical to run 0,
    with the same id and no ``noise_floor`` key, so five independent fingerprints were produced
    where a floor was asked for.  The canonical is now the only cert run 0 has."""
    canonical = json.loads(lab["fingerprint"].read_bytes().decode("utf-8"))
    assert canonical["body"]["run_index"] == 0
    assert canonical["body"]["noise_floor"]["per_channel"] != {}
    run_bodies = [
        json.loads(path.read_bytes().decode("utf-8"))["body"] for path in lab["run_certs"]
    ]
    assert [body["run_index"] for body in run_bodies] == list(range(1, RUNS))
    assert all("noise_floor" not in body for body in run_bodies)
    assert canonical["body"]["noise_floor"]["runs"] == lab["run_ids"]


def test_the_floor_carries_the_section_5_7_overall_size_beside_the_per_channel_alphas(lab):
    canonical = json.loads(lab["fingerprint"].read_bytes().decode("utf-8"))
    block = canonical["body"]["noise_floor"]
    # alpha_single is per channel and is a diagnostic (section 5.7); at R = 5 it is 1/11.
    assert block["per_channel"]["exact"]["pairs"] == 10
    assert block["per_channel"]["exact"]["alpha_single"] == 1 / 11
    # Every floor here is 0.0, and section 5.7 governs that case by an exceedance test on the
    # distance rather than by a ratio: 0 of the 5 runs exceeded, so the measured fraction is 0.
    assert block["alpha_overall"] == 0.0
    assert len(block["standardized_max"]) == RUNS
    assert "floor_c is 0" in block["standardization"]
    assert lab["fingerprint_payload"]["alpha_overall"] == 0.0


def test_fingerprint_without_a_plan_carries_no_floor(lab, tmp_path):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 1, "--created", F.CREATED, "--out", tmp_path / "fp",
    )
    assert code == 0
    assert payload["floor"] is None and payload["noise_plan"] is None
    # `tier` follows subject.kind: a weights subject is the white-box tier (section 3.1).
    assert payload["runs"] == 1 and payload["tier"] == "white-box"
    assert payload["channels"] == {"exact": True, "seqlp": True, "topk": True}


def test_fingerprint_names_the_mock_runner_explicitly_and_reproduces_the_same_cert(lab, tmp_path):
    # `--runner mock` is the contract's spelling and the default; both must build one cert.
    code, spelled = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runner", "mock", "--runs", 1, "--created", F.CREATED, "--out", tmp_path / "a",
    )
    assert code == 0
    code, defaulted = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runs", 1, "--created", F.CREATED, "--out", tmp_path / "b",
    )
    assert code == 0
    assert spelled["id"] == defaulted["id"]


def test_fingerprint_with_no_logprobs_marks_the_two_derived_channels_absent(lab, tmp_path):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 1, "--no-logprobs",
        "--created", F.CREATED, "--out", tmp_path / "fp",
    )
    assert code == 0
    assert payload["channels"]["exact"] is True
    assert payload["channels"]["seqlp"] is False and payload["channels"]["topk"] is False


def test_fingerprint_redacts_the_output_text_but_keeps_the_checksum(lab, tmp_path):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 1, "--redact",
        "--created", F.CREATED, "--out", tmp_path / "fp",
    )
    assert code == 0 and payload["redacted"] is True
    items = payload["cert"]["body"]["items"]
    assert all("output_text" not in item for item in items)
    assert all(len(item["output_sha256"]) == 64 for item in items)


def test_fingerprint_refuses_a_recipe_that_names_another_battery(lab, tmp_path):
    spec = write_json(
        tmp_path / "spec_wrong.json",
        {"subject": lab["subject"], "recipe": F.recipe(battery=F.fake_id("elsewhere"))},
    )
    code, payload = cli(
        "fingerprint", "--subject", spec, "--battery", lab["pool"], "--key", lab["key"],
        "--runs", 1, "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "is not the battery cert's id" in payload["error"]


def test_fingerprint_refuses_a_plan_that_is_not_a_cert_id(lab):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"], "--key", lab["key"],
        "--runs", 2, "--plan", "noise-plan-7", "--created", F.CREATED,
    )
    assert code == EXIT["invalid"]
    assert "--plan must be a cert id" in payload["error"]


def test_fingerprint_without_a_key_is_invalid_and_with_a_missing_battery_is_unavailable(lab, tmp_path):
    code, payload = cli("fingerprint", "--subject", lab["spec"], "--battery", lab["pool"])
    assert code == EXIT["invalid"]
    assert "--key is required" in payload["error"]

    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", tmp_path / "absent.json",
        "--key", lab["key"],
    )
    assert code == EXIT["unavailable"]


def test_fingerprint_rejects_an_unknown_runner(lab):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 1, "--runner", "vllm",
    )
    assert code == EXIT["invalid"]
    assert "--runner must be 'mock' or 'hf'" in payload["error"]


def test_the_hf_runner_needs_a_local_snapshot_never_a_hub_id(lab):
    code, payload = cli(
        "fingerprint", "--subject", lab["spec"], "--battery", lab["pool"],
        "--key", lab["key"], "--runs", 1, "--runner", "hf",
    )
    assert code == EXIT["invalid"]
    assert "--snapshot" in payload["error"]


# --------------------------------------------------------------------------- verify


def test_verify_ref_on_the_logged_fingerprint_is_same_and_exits_zero(lab):
    code, payload = cli("verify", "--ref", lab["fingerprint"], "--log", lab["log"])
    assert code == EXIT["same"] == 0, payload
    assert payload["verdict"] == "same"
    assert payload["mode"] == "ref" and payload["mismatched"] == []
    assert payload["result"]["kind"] == "verify"
    assert payload["result"]["coverage"] == "within"
    assert payload["report"].startswith("styxx verify --ref")


def test_verify_ref_against_a_drifted_runner_is_drift_with_a_confirmation(lab):
    code, payload = cli(
        "verify", "--ref", lab["fingerprint"], "--log", lab["log"], "--drift", DRIFT_ITEM
    )
    assert code == EXIT["drift"] == 1, payload
    assert payload["verdict"] == "drift"
    assert payload["result"]["confirmation_run"] is not None
    assert payload["result"]["per_channel"]["exact"]["distance"] == ONE_OF_TWELVE


def test_verify_ref_without_confirmation_stops_at_exceeds_floor(lab):
    code, payload = cli(
        "verify", "--ref", lab["fingerprint"], "--log", lab["log"],
        "--drift", DRIFT_ITEM, "--no-confirm",
    )
    assert code == EXIT["inconclusive"] == 2, payload
    assert payload["verdict"] == "exceeds_floor"
    assert payload["result"]["confirmation_run"] is None


def test_verify_ref_needs_a_resolvable_battery(lab):
    # Deviation 1 in this module's docstring: --ref re-runs the recipe, so it cannot proceed
    # without the battery's prompts, and says so as `invalid` rather than crashing.
    code, payload = cli("verify", "--ref", lab["fingerprint"])
    assert code == EXIT["invalid"], payload
    assert payload["verdict"] == "invalid"
    assert any("does not resolve" in reason for reason in payload["result"]["invalid_reasons"])


def test_verify_diff_of_a_cert_with_itself_is_same(lab):
    code, payload = cli("verify", "--diff", lab["fingerprint"], lab["fingerprint"], "--log", lab["log"])
    assert code == EXIT["same"] == 0, payload
    assert payload["verdict"] == "same" and payload["mode"] == "diff"


def test_verify_diff_never_says_drift_even_when_the_distance_exceeds_the_floor(lab):
    code, payload = cli(
        "verify", "--diff", lab["fingerprint"], lab["drifted"],
        "--log", lab["log"], "--resolve", lab["drift_dir"],
    )
    assert code == EXIT["inconclusive"] == 2, payload
    # Section 6: a --diff executes nothing, so it has no confirmation run and never reaches drift.
    assert payload["verdict"] == "exceeds_floor"
    assert payload["result"]["per_channel"]["exact"]["distance"] == ONE_OF_TWELVE
    assert payload["result"]["per_channel"]["exact"]["floor"] == 0.0


def test_verify_diff_across_batteries_is_a_mismatch_and_carries_no_distances(lab):
    code, payload = cli(
        "verify", "--diff", lab["fingerprint"], lab["canary_fingerprint"], "--log", lab["log"]
    )
    assert code == EXIT["mismatch"] == 3, payload
    assert payload["verdict"].startswith("mismatch")
    assert "recipe.battery" in payload["mismatched"]
    assert payload["result"]["per_channel"] == {}


def test_verify_refuses_ref_and_diff_together_and_neither_alone(lab):
    code, payload = cli(
        "verify", "--ref", lab["fingerprint"], "--diff", lab["fingerprint"], "--diff", lab["drifted"]
    )
    assert code == EXIT["invalid"]
    assert "not both" in payload["error"]

    code, payload = cli("verify")
    assert code == EXIT["invalid"]
    assert "--ref" in payload["error"] and "--diff" in payload["error"]


def test_verify_diff_needs_exactly_two_certs(lab):
    code, payload = cli("verify", "--diff", lab["fingerprint"])
    assert code == EXIT["invalid"]
    assert "exactly two cert files" in payload["error"]


def test_verify_on_a_missing_cert_is_unavailable(lab, tmp_path):
    code, payload = cli("verify", "--ref", tmp_path / "absent.json", "--log", lab["log"])
    assert code == EXIT["unavailable"]
    assert "no such file" in payload["error"]


def test_verify_on_a_tampered_cert_is_invalid(lab, tmp_path):
    raw = lab["fingerprint"].read_bytes().decode("utf-8")
    broken = tmp_path / "tampered.json"
    obj = json.loads(raw)
    obj["created"] = "2020-01-01T00:00:00Z"  # the id no longer recomputes
    write_json(broken, obj)
    code, payload = cli("verify", "--diff", broken, lab["fingerprint"], "--log", lab["log"])
    assert code == EXIT["invalid"], payload
    assert payload["verdict"] == "invalid"
    assert payload["result"]["invalid_reasons"]


def test_verify_challenge_records_the_distance_against_the_targets_floor(lab):
    code, payload = cli(
        "verify", "--ref", lab["fingerprint"], "--log", lab["log"],
        "--drift", DRIFT_ITEM, "--challenge",
    )
    assert code == EXIT["drift"] == 1, payload
    challenge = payload["challenge"]
    assert challenge["per_channel"]["exact"]["distance"] == ONE_OF_TWELVE
    assert challenge["per_channel"]["exact"]["target_floor"] == 0.0
    assert challenge["coverage"] == "within"


def test_verify_challenge_refuses_to_mint_against_another_subject(lab, tmp_path):
    """Section 9 rule 1, at the mint. C1 of papers/v8/challenge_and_attack_2026_09_09 signed a
    challenge whose `own` half was an fp16 cert against a bf16 target; a tool does not sign a
    cert it can already show is not a challenge."""
    target = json.loads(lab["fingerprint"].read_bytes().decode("utf-8"))
    own = copy.deepcopy(target)
    own["subject"] = dict(own["subject"], precision="fp16")
    own.pop("id", None)
    own.pop("sig", None)
    own = certmod.sign(own, F.keypair("issuer")[0])
    own_path = write_json(tmp_path / "own_fp16.json", own)

    code, payload = cli(
        "verify", "--ref", lab["fingerprint"], "--log", lab["log"], "--challenge",
        "--own", own_path, "--key", lab["key"], "--challenge-out", tmp_path / "ch.json",
    )
    assert code == EXIT["invalid"], payload
    assert payload["challenge_validity"] == ["cross-subject:precision"]
    assert "No match, no challenge" in payload["error"]
    assert not (tmp_path / "ch.json").exists()


def test_verify_challenge_refuses_to_sign_when_it_could_not_compute_rule_one(lab, tmp_path):
    """C-MINT-UNCOMPUTED of papers/v8/challenge_and_attack_2026_09_09.

    This used to sign, print `challenge_validity: null` to stdout and record NOTHING about the
    uncomputed rule in the signed bytes, so the cert a reader received was indistinguishable
    from one whose rule 1 had passed. It fails closed now: an unresolvable `--own` is a missing
    input, not a gap to publish.
    """
    out = tmp_path / "uncomputed.json"
    code, payload = cli(
        "verify", "--ref", lab["fingerprint"], "--log", lab["log"], "--challenge",
        "--own", F.fake_id("a cert this command cannot read"),
        "--key", lab["key"], "--drift", DRIFT_ITEM, "--challenge-out", out,
    )
    assert code == EXIT["invalid"], payload
    assert payload["challenge_validity"] is None
    assert "cannot be computed here" in payload["error"]
    assert "--own" in payload["error"] and "--log/--resolve" in payload["error"]
    assert not out.exists()  # nothing signed, nothing written
    assert "challenge_cert" not in payload


def test_verify_writes_a_signed_result_cert_that_verify_cert_accepts(lab, tmp_path):
    out = tmp_path / "result.json"
    code, payload = cli(
        "verify", "--diff", lab["fingerprint"], lab["fingerprint"], "--log", lab["log"],
        "--result-out", out, "--key", lab["key"], "--created", F.CREATED,
    )
    assert code == 0, payload
    assert Path(payload["result_written"]) == out
    code, checked = cli("log", "verify-cert", out)
    assert code == 0 and checked["ok"] is True
    assert checked["type"] == "result" and checked["id"] == payload["result_cert_id"]


def test_verify_result_out_without_a_key_is_refused(lab, tmp_path):
    code, payload = cli(
        "verify", "--diff", lab["fingerprint"], lab["fingerprint"], "--log", lab["log"],
        "--result-out", tmp_path / "r.json",
    )
    assert code == EXIT["invalid"]
    assert "--key" in payload["error"]


# --------------------------------------------------------------------------- log


def test_log_init_reports_the_log_id_and_starts_empty(tmp_path):
    pem = tmp_path / "log.pem"
    cli("key", "generate", "--out", pem)
    code, payload = cli("log", "init", "--log", tmp_path / "fresh", "--key", pem)
    assert code == 0
    assert payload["size"] == 0
    assert payload["log_id"].startswith("sha256:") and len(payload["log_id"]) == 71
    assert (tmp_path / "fresh" / "keys" / "log.pub").is_file()


def test_log_init_needs_a_key_or_a_pub(tmp_path):
    code, payload = cli("log", "init", "--log", tmp_path / "nokey")
    assert code == EXIT["invalid"]
    assert "--key" in payload["error"] and "--pub" in payload["error"]


def test_the_ladder_appended_in_dependency_order_at_consecutive_indexes(lab):
    assert lab["appended"] == list(range(lab["size"]))
    code, payload = cli("log", "append", lab["pool"], "--log", lab["log"])
    assert code == EXIT["invalid"]
    assert "refused" in payload["error"] and "duplicate" in payload["error"]


def test_log_append_refuses_a_cert_whose_ref_is_not_in_the_log(lab, tmp_path):
    orphan = F.make_cert(
        "fingerprint",
        subject=lab["subject"],
        recipe=F.recipe(battery=F.fake_id("nowhere")),
        body=F.fingerprint_body(),
        refs=[{"role": "battery", "id": F.fake_id("nowhere")}],
    )
    path = write_json(tmp_path / "orphan.json", orphan)
    code, payload = cli("log", "append", path, "--log", lab["log"])
    assert code == EXIT["invalid"]
    assert "does not resolve" in payload["error"]


def test_log_sth_and_prove_agree_on_the_tree_size(lab):
    assert lab["sth_body"]["tree_size"] == lab["size"]
    assert lab["sth_body"]["log_id"] == lab["log_id"]
    proof = json.loads(lab["proof"].read_text(encoding="utf-8"))
    assert proof["leaf_index"] == 0 and proof["tree_size"] == lab["size"]


def test_log_sth_needs_the_logs_own_key(lab, tmp_path):
    other = tmp_path / "other.pem"
    cli("key", "generate", "--out", other)
    code, payload = cli("log", "sth", "--log", lab["log"], "--key", other)
    assert code == EXIT["invalid"]
    assert "refused" in payload["error"]


# --------------------------------------------------------------------------- the four verify-* on a mirror


def test_verify_cert_passes_on_every_entry_of_the_mirror(lab):
    entries = sorted((lab["mirror"] / "entries").rglob("*.json"))
    certs = [p for p in entries if not p.name.endswith(".meta.json")]
    assert len(certs) == lab["size"]
    for path in certs:
        code, payload = cli("log", "verify-cert", path)
        assert code == 0, (path.name, payload)
        assert payload["ok"] is True and payload["reasons"] == []


def test_verify_sth_passes_against_the_mirrors_own_pinned_key(lab):
    code, payload = cli("log", "verify-sth", lab["sth"], "--log", lab["mirror"])
    assert code == 0, payload
    assert payload["ok"] is True and payload["tree_size"] == lab["size"]
    code, payload = cli(
        "log", "verify-sth", lab["sth"], "--pin", lab["mirror"] / "keys" / "log.pub"
    )
    assert code == 0 and payload["ok"] is True


def test_verify_inclusion_passes_on_the_mirror(lab):
    code, payload = cli(
        "log", "verify-inclusion", lab["proof"], lab["sth"], "--log", lab["mirror"]
    )
    assert code == 0, payload
    assert payload["ok"] is True and payload["leaf_index"] == 0


def test_verify_consistency_passes_between_two_heads_of_the_mirror(lab):
    # The head at tree_size 1 (the pool battery alone) and the full head, both mirrored.
    assert lab["sth_small_body"]["tree_size"] == 1
    code, payload = cli(
        "log", "verify-consistency", lab["sth_small"], lab["sth"], "--log", lab["mirror"]
    )
    assert code == 0, payload
    assert payload["ok"] is True
    assert payload["first"] == 1 and payload["second"] == lab["size"]


def test_verify_consistency_takes_a_proof_file_and_a_pinned_key_instead_of_a_log(lab, tmp_path):
    # A stranger holding the two heads, the proof and the pinned key -- and no log directory.
    proof = Log(lab["mirror"]).consistency(1, lab["size"])
    proof_path = write_json(tmp_path / "consistency.json", proof)
    code, payload = cli(
        "log", "verify-consistency", lab["sth_small"], lab["sth"],
        "--proof", proof_path, "--pin", lab["mirror"] / "keys" / "log.pub",
    )
    assert code == 0, payload
    assert payload["ok"] is True and payload["first"] == 1


def test_verify_sth_refuses_a_forged_root(lab, tmp_path):
    forged = json.loads(lab["sth"].read_text(encoding="utf-8"))
    forged["root_hash"] = "sha256:" + "0" * 64
    path = write_json(tmp_path / "forged.json", forged)
    code, payload = cli("log", "verify-sth", path, "--log", lab["mirror"])
    assert code == EXIT["invalid"]
    assert payload["ok"] is False and payload["reason"]


def test_verify_inclusion_refuses_a_proof_for_the_wrong_leaf(lab, tmp_path):
    proof = json.loads(lab["proof"].read_text(encoding="utf-8"))
    proof["leaf_index"] = 1
    path = write_json(tmp_path / "wrong_leaf.json", proof)
    code, payload = cli("log", "verify-inclusion", path, lab["sth"], "--log", lab["mirror"])
    assert code == EXIT["invalid"]
    assert payload["ok"] is False


def test_verify_consistency_needs_a_proof_or_a_log(lab):
    code, payload = cli("log", "verify-consistency", lab["sth"], lab["sth"])
    # A missing proof is a malformed request (4), checked before the pinned key (5) is looked for.
    assert code == EXIT["invalid"]
    assert "--proof" in payload["error"] and "--log" in payload["error"]


def test_verify_sth_without_a_key_to_check_it_against_is_unavailable(lab):
    code, payload = cli("log", "verify-sth", lab["sth"])
    assert code == EXIT["unavailable"]
    assert "log public key" in payload["error"]


# --------------------------------------------------------------------------- mirror


def test_the_mirror_verified_every_entry_and_every_head(lab):
    report = lab["mirror_report"]
    assert report["verified"] is True
    assert report["entries"] == lab["size"] and report["sths"] >= 1
    assert report["misbehaviour"] == [] and report["tamper"] == []


def test_a_tampered_source_mirrors_with_a_report_and_exits_invalid(lab, tmp_path):
    tampered = tmp_path / "tampered_log"
    shutil.copytree(os.fspath(lab["log"]), os.fspath(tampered))
    entry = tampered / "entries" / "000000" / "00000000.json"
    raw = entry.read_bytes()
    # One byte inside the canonical bytes: the leaf hash the tree holds no longer matches.
    entry.write_bytes(raw.replace(b'"pool-v1"', b'"pool-v2"'))
    code, payload = cli("log", "mirror", "--log", tampered, "--to", tmp_path / "out")
    assert code == EXIT["invalid"], payload
    assert payload["report"]["verified"] is False
    assert payload["report"]["tamper"] or payload["report"]["misbehaviour"]


def test_mirror_checks_the_source_against_a_pinned_key_and_a_pinned_head(lab, tmp_path):
    code, payload = cli(
        "log", "mirror", "--log", lab["log"], "--to", tmp_path / "pinned",
        "--pin", lab["mirror"] / "keys" / "log.pub", "--pinned-sth", lab["sth"],
    )
    assert code == 0, payload
    assert payload["report"]["verified"] is True
    assert payload["report"]["unpublished"] == []


def test_mirror_refuses_a_source_pinned_to_a_stranger_key(lab, tmp_path):
    other_pem = tmp_path / "other.pem"
    cli("key", "generate", "--out", other_pem)
    code, payload = cli(
        "log", "mirror", "--log", lab["log"], "--to", tmp_path / "wrong",
        "--pin", Path(str(other_pem) + ".pub"),
    )
    assert code == EXIT["invalid"], payload
    assert payload["report"]["verified"] is False


def test_log_verbs_refuse_a_missing_log_directory(tmp_path):
    code, payload = cli("log", "sth", "--log", tmp_path / "nothing", "--key", tmp_path / "k.pem")
    assert code == EXIT["unavailable"]
    assert "no such directory" in payload["error"]


def test_log_needs_a_known_subcommand(lab):
    code, payload = cli("log", "rewind", "--log", lab["log"])
    assert code == EXIT["invalid"]
    assert "unknown subcommand" in payload["error"]


# --------------------------------------------------------------------------- stdout discipline


def test_stdout_carries_nothing_but_the_object_even_on_the_longest_payload(lab):
    argv = [sys.executable, "-m", "styxx.v8", "verify", "--ref",
            os.fspath(lab["fingerprint"]), "--log", os.fspath(lab["log"])]
    proc = subprocess.run(argv, cwd=os.fspath(REPO_ROOT), env=CHILD_ENV, capture_output=True, timeout=600)
    assert proc.stderr == b""
    text = proc.stdout.decode("utf-8")
    assert text.endswith("\n") and text.count("\n") == text.rstrip("\n").count("\n") + 1
    assert b"\r\n" not in proc.stdout.replace(b"\\r\\n", b"")
    assert text.lstrip().startswith("{")
    json.loads(text)  # one value, nothing trailing


def test_the_entry_point_module_is_the_only_v8_command_surface():
    # GATED S11-01: the v8 verbs live under `python -m styxx.v8` and nowhere else.
    main_py = REPO_ROOT / "styxx" / "v8" / "__main__.py"
    assert main_py.is_file()
    text = main_py.read_text(encoding="utf-8")
    assert "from .cli import main" in text
    marker = "# GATED S11-01: recommendation implemented; operator may reverse."
    assert marker in text
    assert marker in (REPO_ROOT / "styxx" / "v8" / "cli.py").read_text(encoding="utf-8")
    seven = (REPO_ROOT / "styxx" / "cli.py").read_text(encoding="utf-8", errors="replace")
    assert "styxx.v8" not in seven
