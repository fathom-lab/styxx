"""Coverage and environment: the three places a floor's own description was still a free field.

`papers/v8/THE_BOUNDARY_2026_09_09.md` partitions every surviving defect into a class reachable
by a predicate over bytes on disk and a class reachable by nothing. These three are class one,
and they share one shape: a field the whole of section 5.4 resolves against, written by the party
the answer is about, with nothing re-deriving it.

* **PLAN-ENV, the root.** `prereg noise-plan` built no runner and copied `environment` straight
  out of the `--subject` spec file. Section 2.2's mint rule ("the environment is OBSERVED, never
  asserted") had already been applied to `styxx fingerprint`; the plan never got it. So the block
  section 5.4's whole vocabulary resolves against was a value the issuer typed, on the one cert
  that is signed BEFORE the runs and that every run of the floor names by id.
* **A-COVER.** `noise_floor.covers` and `noise_floor.not_covered` were never compared against the
  plan the floor names. The anchor re-derives `per_channel` and section 5.7's overall size from
  the run certs and compares exactly, and these two lists sat beside those numbers as assertions.
  `covers` decides whether an outside reproduction is a dispute or a coverage report (sections
  5.4, 9), so an issuer picking it after the runs picks who may contradict its result -- and an
  empty `not_covered` is section 5.4's claim to cover every environment there is, because
  `verify._coverage_diff` acts over `not_covered` alone.
* **ENV-ABSENT.** The mint-side environment repair sat behind `if "environment" in subject:`, and
  `schema/fingerprint.json` required only `kind` on `subject`. `schema/subject.json` requires the
  block on the `weights` branch and not on the `alias` one, so an alias spec with the key deleted
  minted a fingerprint that observed no environment and named none -- and carried a floor whose
  `not_covered` was therefore empty. A guard an omission turns off is not a guard.

WHAT IS NOT CLOSED HERE, and it is the same sentence three times. Every predicate below compares
one of the issuer's certs against another of the issuer's certs. A party that mints its plan and
its runs on one box and writes one environment into both is not caught by any of them, and cannot
be: that is the label class, and `Log._check_floor_environment_matches_the_plan` says so in its
own docstring. What these remove is the state where the two disagreed and nothing looked, and the
state where the coverage line was typed rather than derived.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import cli
from styxx.v8 import floor as floormod
from styxx.v8 import keys
from styxx.v8 import runner as runnermod
from styxx.v8.consts import EXIT
from styxx.v8.log import AppendRefused, Log
from tests import v8_fixtures as F
from tests.test_v8_environment_guard import CPU_ENV, GPU_ENV, DeviceRunner

ISSUER_SEED, ISSUER_PUB = F.keypair("issuer")
LOG_SEED, LOG_PUB = F.keypair("log-key")

RUNS = 5


# ----------------------------------------------------------------- helpers


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
    return path


def key_file(tmp_path: Path) -> Path:
    path = tmp_path / "issuer.pem"
    if not path.exists():
        keys.save_private_pem(ISSUER_SEED, path)
    return path


def battery_file(tmp_path: Path) -> tuple[dict, Path]:
    cert = F.make_cert("battery", body=F.battery_body("fixed-v1", n=8))
    return cert, write_json(tmp_path / "battery.json", cert)


def spec_file(tmp_path: Path, subject: dict, battery_id: str, name: str = "spec.json") -> Path:
    return write_json(
        tmp_path / name, {"subject": subject, "recipe": F.recipe(battery=battery_id)}
    )


def use_runner(monkeypatch, runner) -> None:
    """Drive the CLI with a runner that is not the mock.

    `_make_runner` knows `mock` and `hf` only, and the mock is SYNTHETIC -- section 2.2 exempts a
    runner that computes from a hash and observes no hardware, so an environment supplied beside
    one stands as written. Every environment rule below is about a runner that claims to be a
    model, which is what `DeviceRunner` is.
    """
    monkeypatch.setattr(cli, "_make_runner", lambda opts: runner)


def roster() -> list[dict]:
    return [
        {
            "name": F.ISSUER_NAME,
            "key": F.public_key("issuer"),
            "from_index": 0,
            "retired_at_index": None,
        }
    ]


def plan_cert(*factors, environment: dict, runs: int = RUNS) -> dict:
    return F.make_cert(
        "prereg",
        body={
            "kind": "noise-plan",
            "runs": runs,
            "nuisance": [{"factor": f, "values": list(v)} for f, v in factors],
            "environment": copy.deepcopy(environment),
        },
    )


SIZES = (8, 4, 8, 4, 8)


def ladder(tmp_path, *, plan=None, run_environment=None, block_overrides=None):
    """A log holding a battery, a plan fixing R = 5 over batch_size 8|4, four runs, a canonical.

    The floor's numbers are always the ones the named bodies really produce, so every refusal here
    is about the floor's DESCRIPTION -- its coverage lists, its environment -- and never about its
    arithmetic. `floor_disagreement` is asserted empty wherever it is asked.
    """
    subject_env = F.WEIGHTS_SUBJECT["environment"] if run_environment is None else run_environment
    subject = F.weights_subject(environment=copy.deepcopy(subject_env))
    log = Log.init(tmp_path / "log", LOG_PUB, roster())
    battery = F.make_cert("battery", body=F.battery_body("fixed-v1", n=8))
    log.append(battery)
    if plan is None:
        plan = plan_cert(("batch_size", ("8", "4")), environment=F.WEIGHTS_SUBJECT["environment"])
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]

    runs = []
    for k in range(1, 5):
        body = F.fingerprint_body(n=8, run_index=k, batch_size=SIZES[k])
        cert = F.make_cert(
            "fingerprint",
            subject=subject,
            recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=SIZES[k])),
            refs=[{"role": "battery", "id": battery["id"]}] + plan_ref,
            body=body,
        )
        log.append(cert)
        runs.append(cert)

    body = F.fingerprint_body(n=8, run_index=0, batch_size=SIZES[0])
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    body["noise_floor"].update(copy.deepcopy(block_overrides or {}))
    canonical = F.make_cert(
        "fingerprint",
        subject=subject,
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=SIZES[0])),
        refs=(
            [{"role": "battery", "id": battery["id"]}]
            + plan_ref
            + [{"role": "run", "id": c["id"]} for c in runs]
        ),
        body=body,
    )
    return log, plan, runs, canonical


# ================================================================== 1. PLAN-ENV


def test_the_plan_records_the_environment_the_runner_observed_not_the_one_the_spec_names(
    tmp_path, monkeypatch
):
    """PLAN-ENV. `prereg noise-plan` built no runner at all, so the spec file WAS the plan's
    environment -- and that block is what section 5.4's `not_covered` is derived from and what
    `verify` resolves every coverage name against."""
    use_runner(monkeypatch, DeviceRunner(CPU_ENV))
    battery, battery_path = battery_file(tmp_path)
    # The spec names nothing the runner did not report; the runner is on the CPU.
    spec = spec_file(tmp_path, F.weights_subject(environment=copy.deepcopy(CPU_ENV)), battery["id"])
    code, payload = cli.run(
        [
            "prereg", "noise-plan", "--runs", str(RUNS), "--nuisance", "batch_size=1|8",
            "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--created", F.CREATED,
        ]
    )
    assert code == 0, payload
    body = payload["cert"]["body"]
    assert body["environment"] == CPU_ENV
    assert body["environment"]["hardware"]["device"] == "cpu"
    # and the plan's own subject carries the observed block, not the one it was handed
    assert payload["cert"]["subject"]["environment"] == CPU_ENV


def test_a_plan_cannot_name_a_card_the_runner_minting_it_does_not_have(tmp_path, monkeypatch):
    """The S-DEVICE demonstration moved to the plan. One CPU runner, a spec naming an RTX 4070:
    before this, `prereg noise-plan` signed the card, every run of that plan inherited the
    sentence through `not_covered`, and no byte anywhere had been observed."""
    use_runner(monkeypatch, DeviceRunner(CPU_ENV))
    battery, battery_path = battery_file(tmp_path)
    spec = spec_file(tmp_path, F.weights_subject(environment=copy.deepcopy(GPU_ENV)), battery["id"])
    code, payload = cli.run(
        [
            "prereg", "noise-plan", "--runs", str(RUNS), "--nuisance", "batch_size=1|8",
            "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--created", F.CREATED,
        ]
    )
    assert code == EXIT["unavailable"], payload
    assert "contradicts the environment observed" in payload["error"]
    assert "hardware.gpu" in payload["error"]


def test_the_mint_refuses_runs_whose_observed_environment_is_not_the_plans(tmp_path, monkeypatch):
    """The half that makes an observed plan bind. A plan is written before its runs, so it names
    an INTENDED environment; without this the intention is a sentence about another process."""
    use_runner(monkeypatch, DeviceRunner(GPU_ENV))
    battery, battery_path = battery_file(tmp_path)
    gpu_spec = spec_file(
        tmp_path, F.weights_subject(environment=copy.deepcopy(GPU_ENV)), battery["id"], "gpu.json"
    )
    code, payload = cli.run(
        [
            "prereg", "noise-plan", "--runs", str(RUNS), "--nuisance", "batch_size=1|8",
            "--subject", str(gpu_spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--created", F.CREATED,
            "--out", str(tmp_path / "plan.json"),
        ]
    )
    assert code == 0, payload
    plan_path = tmp_path / "plan.json"

    # the same plan, run on the CPU of the same box
    use_runner(monkeypatch, DeviceRunner(CPU_ENV))
    cpu_spec = spec_file(
        tmp_path, F.weights_subject(environment=copy.deepcopy(CPU_ENV)), battery["id"], "cpu.json"
    )
    code, payload = cli.run(
        [
            "fingerprint", "--subject", str(cpu_spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--runs", str(RUNS), "--plan", str(plan_path),
            "--created", F.CREATED,
        ]
    )
    assert code == EXIT["invalid"], payload
    assert "the plan fixed environment hardware." in payload["error"]
    assert "would measure a floor of a different quantity" in payload["error"]


def test_the_log_refuses_a_floor_whose_runs_are_not_in_the_plans_environment(tmp_path):
    """The same predicate over bytes on disk, which is the one a stranger can re-run. The floor's
    arithmetic is honest about the bodies it names -- that is the whole point of putting it here
    rather than trusting the mint."""
    plan = plan_cert(("batch_size", ("8", "4")), environment=copy.deepcopy(GPU_ENV))
    log, _, _, canonical = ladder(tmp_path, plan=plan, run_environment=copy.deepcopy(CPU_ENV))
    assert log.floor_disagreement(canonical) == []
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert reason.startswith("floor:")
    assert "fixed environment hardware." in reason
    assert "PLAN-ENV" in reason


def test_a_run_silent_about_a_field_the_plan_fixed_is_refused_rather_than_passed(tmp_path):
    """Fail closed. A run whose `subject.environment` does not carry a leaf the plan fixed did not
    hold it fixed -- it failed to say, and a check whose quietest state is "the field was missing"
    is not a check."""
    # `hardware.device` is the leaf section 2.2 added and `schema/subject.json` does not require,
    # so a run can legally omit it while the plan fixes it -- which is the shape this refuses.
    partial = copy.deepcopy(CPU_ENV)
    partial["hardware"] = {k: v for k, v in partial["hardware"].items() if k != "device"}
    plan = plan_cert(("batch_size", ("8", "4")), environment=copy.deepcopy(CPU_ENV))
    log, _, _, canonical = ladder(tmp_path, plan=plan, run_environment=partial)
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "records no such field in subject.environment" in exc.value.reason


def test_a_run_may_carry_leaves_the_plan_never_mentioned(tmp_path):
    """One-directional on purpose: `harness` and `env_lock_sha256` are section 2.3's skew fields,
    which a runner cannot observe and a plan does not fix."""
    plan = plan_cert(("batch_size", ("8", "4")), environment=copy.deepcopy(CPU_ENV))
    extra = dict(copy.deepcopy(CPU_ENV), harness={"name": "styxx", "version": "8.0.0"})
    log, _, _, canonical = ladder(tmp_path, plan=plan, run_environment=extra)
    assert log.append(canonical) == log.size() - 1


# ================================================================== 2. A-COVER


def test_the_honest_floor_carries_the_plans_derivation_and_appends(tmp_path):
    """The control. `covers` is the plan's declared factors; `not_covered` is every environment
    leaf it did not declare."""
    log, plan, _, canonical = ladder(tmp_path)
    block = canonical["body"]["noise_floor"]
    assert block["covers"] == ["batch_size"]
    assert block["not_covered"] == [
        "hardware.count", "hardware.driver", "hardware.gpu",
        "runtime.backend", "runtime.framework", "runtime.version",
    ]
    assert floormod.plan_coverage(plan["body"]) == (block["covers"], block["not_covered"])
    assert log.append(canonical) == log.size() - 1


def test_refuses_a_floor_that_claims_to_cover_what_its_plan_did_not_vary(tmp_path):
    """A-COVER, in the form the third pass demonstrated: `covers` naming the gpu, the driver and
    everything else, `not_covered` empty, an honest per-channel floor beside them, exit 0.

    `floor_disagreement` is empty here and that is the finding -- the anchor recomputes the
    NUMBERS from the runs and had nothing to say about the two lists that decide who the numbers
    are binding on."""
    log, _, _, canonical = ladder(
        tmp_path,
        block_overrides={
            "covers": [
                "batch_size", "hardware.gpu", "hardware.driver", "hardware.count",
                "runtime.backend", "runtime.framework", "runtime.version",
            ],
            "not_covered": [],
        },
    )
    assert log.floor_disagreement(canonical) == []  # the numerals ARE the named bodies'
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert reason.startswith("floor: noise_floor.covers is signed as")
    assert "A-COVER" in reason
    assert "picks who may contradict it" in reason


def test_refuses_a_floor_whose_not_covered_alone_was_emptied(tmp_path):
    """The narrower move, and the one that actually buys the exit code: `verify` computes
    `coverage_diff` over `not_covered` alone, so an empty list is the claim that every
    environment is within this floor's coverage and no reproduction is ever
    `beyond-floor-coverage`."""
    log, _, _, canonical = ladder(tmp_path, block_overrides={"not_covered": []})
    assert log.floor_disagreement(canonical) == []
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert exc.value.reason.startswith("floor: noise_floor.not_covered is signed as")


def test_refuses_a_floor_whose_plan_fixes_no_environment(tmp_path):
    """The omission route to the same claim. With no environment the derivation gives
    `not_covered == []`, so a branch that returned quietly here would hand the whole of A-COVER
    back for the price of deleting one key from a hand-signed plan -- which is exactly how
    A-NORUNS worked."""
    plan = F.make_cert("prereg", body={"kind": "noise-plan", "runs": RUNS})
    log, _, _, canonical = ladder(tmp_path, plan=plan)
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert "fixes no environment field" in reason
    assert "covers every environment there is" in reason


def test_the_mint_derives_the_two_lists_and_the_command_line_cannot_state_them(tmp_path, monkeypatch):
    """The mint writes what the log re-derives, so an honest floor never meets these refusals."""
    use_runner(monkeypatch, DeviceRunner(CPU_ENV))
    battery, battery_path = battery_file(tmp_path)
    spec = spec_file(tmp_path, F.weights_subject(environment=copy.deepcopy(CPU_ENV)), battery["id"])
    code, payload = cli.run(
        [
            "prereg", "noise-plan", "--runs", str(RUNS), "--nuisance", "batch_size=1|8",
            "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--created", F.CREATED,
            "--out", str(tmp_path / "plan.json"),
        ]
    )
    assert code == 0, payload
    plan_body = payload["cert"]["body"]
    assert plan_body["covers"] == ["batch_size"]
    assert plan_body["not_covered"] == [
        "hardware.count", "hardware.device", "hardware.driver", "hardware.gpu",
        "runtime.backend", "runtime.framework", "runtime.version",
    ]
    assert floormod.plan_coverage(plan_body) == (
        plan_body["covers"], plan_body["not_covered"]
    )

    code, payload = cli.run(
        [
            "fingerprint", "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--runs", str(RUNS),
            "--plan", str(tmp_path / "plan.json"), "--created", F.CREATED,
        ]
    )
    assert code == 0, payload
    assert payload["covers"] == plan_body["covers"]
    assert payload["not_covered"] == plan_body["not_covered"]
    block = payload["cert"]["body"]["noise_floor"]
    assert (block["covers"], block["not_covered"]) == floormod.plan_coverage(plan_body)


def test_prereg_no_longer_takes_a_not_covered_flag(tmp_path, monkeypatch):
    """A list the anchor recomputes is not a list the command line may state. `--not-covered` was
    the one place an issuer chose by hand, before the runs, which environment fields would read as
    held fixed."""
    use_runner(monkeypatch, DeviceRunner(CPU_ENV))
    battery, battery_path = battery_file(tmp_path)
    spec = spec_file(tmp_path, F.weights_subject(environment=copy.deepcopy(CPU_ENV)), battery["id"])
    code, payload = cli.run(
        [
            "prereg", "noise-plan", "--runs", str(RUNS), "--nuisance", "batch_size=1|8",
            "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--not-covered", "hardware.gpu",
        ]
    )
    assert code == EXIT["invalid"], payload
    assert "not-covered" in payload["error"]


# ================================================================== 3. ENV-ABSENT


def test_a_fingerprint_whose_subject_names_no_environment_does_not_check_out():
    """The bytes half. `schema/subject.json` requires the block on the `weights` branch and leaves
    it optional on `alias`, so this was reachable through the alias tier alone -- and an alias
    fingerprint carrying no environment resolves every section 5.4 coverage name against nothing.

    This is the test that fails if the requirement is dropped from schema/fingerprint.json."""
    subject = {k: v for k, v in F.alias_subject().items() if k != "environment"}
    cert = F.make_cert("fingerprint", subject=subject, body=F.fingerprint_body(logprobs=False))
    outcome = certmod.check(cert)
    assert outcome.ok is False
    assert any(
        "environment" in reason and "required" in reason for reason in outcome.reasons
    ), outcome.reasons


def test_the_mint_observes_the_environment_even_when_the_spec_names_none(tmp_path, monkeypatch):
    """The guard half. It sat behind `if "environment" in subject:`, so the un-observed path was
    one deleted key away; there is no omission branch now."""
    use_runner(monkeypatch, DeviceRunner(CPU_ENV))
    battery, battery_path = battery_file(tmp_path)
    bare = {k: v for k, v in F.alias_subject().items() if k != "environment"}
    spec = spec_file(tmp_path, bare, battery["id"])
    code, payload = cli.run(
        [
            "fingerprint", "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--runs", "1", "--created", F.CREATED,
        ]
    )
    assert code == 0, payload
    assert payload["cert"]["subject"]["environment"] == CPU_ENV


def test_with_the_omission_branch_back_the_same_spec_mints_a_cert_that_observed_nothing(
    tmp_path, monkeypatch
):
    """The second half is what makes the first mean anything: restore the `if` and the same
    command produces a fingerprint whose subject names no environment at all -- so section 5.4
    has nothing to resolve against and `schema/fingerprint.json` is the only thing left refusing
    it."""
    runner = DeviceRunner(CPU_ENV)
    use_runner(monkeypatch, runner)

    def old_observe(runner_arg, subject, flag):
        """`cmd_fingerprint` as it was: the guard, behind a key that may simply be left off."""
        if "environment" in subject:
            subject["environment"] = runnermod.observed_environment(
                runner_arg, subject["environment"]
            )
        return subject.get("environment")

    minted: dict = {}

    def capture(cert_type, **kw):
        if cert_type == "fingerprint":
            minted.setdefault("subject", copy.deepcopy(kw.get("subject")))
        raise cli.CliError("stop: the subject is all this test needs")

    monkeypatch.setattr(cli, "_observe_environment", old_observe)
    monkeypatch.setattr(cli, "_sign_cert", capture)

    battery, battery_path = battery_file(tmp_path)
    bare = {k: v for k, v in F.alias_subject().items() if k != "environment"}
    spec = spec_file(tmp_path, bare, battery["id"])
    code, _ = cli.run(
        [
            "fingerprint", "--subject", str(spec), "--battery", str(battery_path),
            "--key", str(key_file(tmp_path)), "--runs", "1", "--created", F.CREATED,
        ]
    )
    assert code == EXIT["invalid"]
    assert "environment" not in minted["subject"]
