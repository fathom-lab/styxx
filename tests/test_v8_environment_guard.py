"""The environment guard: `verify --ref` must record the environment it RAN IN.

The defect this file pins is S-DEVICE, the third adversarial pass over
`papers/v8/challenge_and_attack_2026_09_09`.  Two holes met:

1. `TransformersRunner.environment()` called `probe_environment()`, which asked the BOX what
   cards it had (`torch.cuda.get_device_name(0)` whenever cuda was available) instead of asking
   the RUNNER what device it was using.  `--device cpu` therefore reported an RTX 4070, and
   `device` appeared in no signed field at all -- it is not in `S_identity` (section 2.2 keeps
   the environment out of identity) and it was not in the environment block either.
2. `--environment <file>` replaced the observed environment wholesale, in the result body AND a
   second time in the challenge body.

So, through the shipped CLI, in ninety seconds:

    verify --ref <a GPU-minted canonical> --runner hf --device cpu
           --environment env_real.json --challenge --own <id>

ran one CPU forward pass and signed a result and a challenge naming a card it never touched,
with `mismatched: []`, `identity_diff: []` and exit 2.

What is asserted here, and the second half is what makes the first mean anything:

1. the environment a `--ref` records comes from the runner; an `--environment` file may add the
   fields a runner cannot observe (`harness`, `env_lock_sha256`) and a field of it that
   contradicts an observed one is exit 5, no cert, no challenge; a CPU run against a GPU-minted
   floor is `beyond-floor-coverage`, which is section 5.4 working;
2. with the guard removed -- `runner.observed_environment` monkeypatched back to the line that
   was there -- the SAME setup signs the RTX 4070 again, in both bodies.

Nothing here makes `device` identity.  The repair is that the field is observed; where a
difference LANDS is section 5.4, and whether a difference outside `not_covered` should gate is
OPERATOR-GATED S2-02 in the spec.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.v8 import cli
from styxx.v8 import fingerprint as FP
from styxx.v8 import runner as runnermod
from styxx.v8 import verify as V
from styxx.v8.consts import EXIT
from styxx.v8.runner import MockRunner
from styxx.v8.runner_hf import hardware_block, run_device
from tests import v8_fixtures as F
from tests.test_v8_verify import reference

ROOT = Path(__file__).resolve().parent.parent

# The two environments of the operator's own run: one box, two devices.  Only the hardware
# block differs -- the runtime is the same interpreter, which is exactly why an environment
# taken from a file could pass for the one that ran.
GPU_ENV: dict = {
    "runtime": {"framework": "transformers", "version": "4.57.3", "backend": "torch 2.5.1+cu121"},
    "hardware": {
        "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
        "driver": "560.94",
        "count": 1,
        "device": "cuda",
    },
}
CPU_ENV: dict = {
    "runtime": {"framework": "transformers", "version": "4.57.3", "backend": "torch 2.5.1+cu121"},
    "hardware": {"gpu": "cpu", "driver": "none", "count": 0, "device": "cpu"},
}


class DeviceRunner(MockRunner):
    """A runner that claims to be a MODEL and observes the device it runs on.

    ``synthetic = False`` is the whole point: the exemption in ``observed_environment`` is for
    runners that compute from a hash and observe no hardware, and a test double that kept the
    mock's marker would exercise the exemption instead of the guard.

    Which is why ``run`` moves every token id off the mock's own derivation.  ``cert.check``
    re-derives ``MockRunner``'s digests from the cert's subject, recipe core and item ids and
    refuses an unmarked body that reproduces them (C-MOCK-2, ``runner.mock_derived_items``), so
    a double that claims to be a model and hands back the mock's arithmetic is refused at mint --
    correctly.  The shift is deterministic, so the same double mints and re-runs the same numbers.
    """

    synthetic = False

    def __init__(self, environment: dict, **kw) -> None:
        super().__init__(**kw)
        self._environment = copy.deepcopy(environment)

    def environment(self) -> dict:
        return copy.deepcopy(self._environment)

    def run(self, items, recipe, subject):
        rows = super().run(items, recipe, subject)
        for row in rows:
            row["token_ids"] = [(t + 1) % 32000 for t in row["token_ids"]]
            row["output_text"] = " ".join(f"w{t}" for t in row["token_ids"])
            for entry in row.get("topk") or []:
                entry["ids"] = [row["token_ids"][entry["pos"]], *entry["ids"][1:]]
        return rows


class NoEnvironmentRunner(DeviceRunner):
    """A runner from before the obligation: it runs, and it will not say where."""

    environment = None  # type: ignore[assignment]

    def __init__(self, **kw) -> None:
        super().__init__(CPU_ENV, **kw)


def _old_observed_environment(runner, annotation=None):
    """`runner.observed_environment` as it behaved before this repair.

    The line in `verify.ref` was::

        dict(environment) if isinstance(environment, Mapping) else runner.environment()

    -- a supplied environment REPLACED the observed one, and nothing compared the two.
    """
    if isinstance(annotation, dict):
        return dict(annotation)
    return dict(runner.environment())


def gpu_reference():
    """(battery, a canonical fingerprint minted on the GPU, resolver) from a non-synthetic run."""
    subject = F.weights_subject(environment=copy.deepcopy(GPU_ENV))
    return reference(subject=subject, runner=DeviceRunner(GPU_ENV))


# ------------------------------------------------------------------ the arithmetic of a device


def test_run_device_answers_what_the_run_takes_not_what_the_box_holds():
    assert run_device("cpu", True) == "cpu"          # a card is present and unused
    assert run_device("CUDA:1", True) == "cuda:1"    # the index is kept; `gpu` alone loses it
    assert run_device(None, True) == "cuda"          # nothing asked for -> what `.to()` would get
    assert run_device(None, False) == "cpu"


def test_the_hardware_block_names_the_device_the_run_computes_on():
    """A CPU run beside an idle RTX 4070 says `cpu`, and counts zero cards, because it used none."""
    assert hardware_block("cpu") == {"gpu": "cpu", "driver": "none", "count": 0, "device": "cpu"}
    assert hardware_block("cuda", "NVIDIA GeForce RTX 4070 Laptop GPU", "560.94") == {
        "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
        "driver": "560.94",
        "count": 1,
        "device": "cuda",
    }


if importlib.util.find_spec("torch") is not None:

    def test_probe_environment_on_the_cpu_names_no_card_on_a_box_that_has_one():
        """The first half of S-DEVICE, measured on this box rather than argued about.

        In a subprocess so the suite stays torch-free (`tests/test_v8_runner_hf.py` holds the
        same rule).  The assertion is not "no GPU is present" -- one is, and the previous
        implementation named it.
        """
        code = (
            "import json; from styxx.v8.runner_hf import probe_environment; "
            "print(json.dumps(probe_environment('cpu')))"
        )
        p = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), check=False
        )
        assert p.returncode == 0, p.stderr
        hardware = json.loads(p.stdout)["hardware"]
        assert hardware == {"gpu": "cpu", "driver": "none", "count": 0, "device": "cpu"}


# --------------------------------------------------------------------- the obligation itself


def test_a_runner_that_will_not_say_where_it_ran_is_unavailable():
    battery, cert, res = gpu_reference()
    out = V.ref(cert, NoEnvironmentRunner(), res)
    assert (out.verdict, out.exit_code) == ("unavailable", EXIT["unavailable"])
    assert "does not report the environment" in out.result_body["unavailable_reason"]
    assert out.result_body["per_channel"] == {}


@pytest.mark.parametrize("hole", ["runtime", "hardware"])
def test_an_environment_missing_a_block_is_unavailable(hole):
    partial = {k: v for k, v in CPU_ENV.items() if k != hole}
    with pytest.raises(runnermod.EnvironmentUnavailable) as exc:
        runnermod.reported_environment(DeviceRunner(partial))
    assert hole in str(exc.value)


def test_an_annotation_adds_the_fields_a_runner_cannot_observe():
    """`harness` and `env_lock_sha256` are the reason an annotation exists at all (section 2.3)."""
    lock = F.sha256_text("a lockfile")
    merged = runnermod.observed_environment(
        DeviceRunner(CPU_ENV),
        {"harness": {"name": "styxx", "version": "8.0.1"}, "env_lock_sha256": lock},
    )
    assert merged["hardware"] == CPU_ENV["hardware"]
    assert merged["harness"]["version"] == "8.0.1"
    assert merged["env_lock_sha256"] == lock


def test_an_annotation_that_contradicts_an_observation_names_every_conflict():
    with pytest.raises(runnermod.EnvironmentUnavailable) as exc:
        runnermod.observed_environment(DeviceRunner(CPU_ENV), GPU_ENV)
    message = str(exc.value)
    for path in ("hardware.gpu", "hardware.driver", "hardware.count", "hardware.device"):
        assert path in message
    assert "runtime.version" not in message  # it agreed, so it is not a conflict


def test_a_synthetic_runner_has_no_observation_to_contradict():
    """The exemption, stated as a test: a hash observes no hardware.

    Its certs carry `synthetic` in the signed bytes and `cert.comparable` refuses them as a
    baseline for, a diff against or a challenge to a measured cert, so the environment such a
    runner names cannot reach a stranger as a claim about a machine.
    """
    assert runnermod.is_synthetic(MockRunner()) is True
    assert runnermod.observed_environment(MockRunner(), GPU_ENV) == GPU_ENV


def test_a_mint_records_the_gpu_the_runner_reports_not_the_one_it_was_handed():
    """The mint half: `body.nuisance.gpu`/`driver` come from `runner.environment()` (section 3.1).

    `styxx fingerprint` applies the same rule one level up -- the `--subject` spec's
    `environment` block annotates what the runner observed and does not replace it -- so a
    canonical fingerprint minted on the CPU cannot carry a subject naming a card.  That refusal
    is `observed_environment`'s, pinned above; a non-synthetic runner cannot be built by the CLI
    without torch and a snapshot, so the CLI's own path is not exercised here.
    """
    battery, cert, res = gpu_reference()
    body = FP.run_fingerprint(
        DeviceRunner(CPU_ENV),
        F.weights_subject(environment=copy.deepcopy(GPU_ENV)),
        F.recipe(battery=battery["id"]),
        battery,
        run_index=0,
        nuisance={},
    )
    assert body["nuisance"]["gpu"] == "cpu"
    assert body["nuisance"]["driver"] == "none"


# ----------------------------------------------------------------------- S-DEVICE, both ways


def test_a_cpu_run_cannot_sign_a_body_naming_the_card_it_did_not_touch():
    """S-DEVICE through the library: the file is refused, and nothing is produced to sign."""
    battery, cert, res = gpu_reference()

    out = V.ref(cert, DeviceRunner(CPU_ENV), res, environment=copy.deepcopy(GPU_ENV))

    assert (out.verdict, out.exit_code) == ("unavailable", EXIT["unavailable"])
    assert "contradicts the environment observed" in out.result_body["unavailable_reason"]
    assert out.result_body["per_channel"] == {}
    assert "environment" not in out.result_body  # nothing to record: nothing was accepted
    # And the human report says which obligation was not met, not just "unavailable".
    assert "unavailable: the environment supplied contradicts" in out.printed
    with pytest.raises(ValueError):
        V.challenge_body(out, GPU_ENV)


def test_the_same_cpu_run_without_the_file_records_the_cpu_and_falls_outside_coverage():
    """Section 5.4 doing its job: the run really was made where the floor does not describe."""
    battery, cert, res = gpu_reference()

    out = V.ref(cert, DeviceRunner(CPU_ENV), res)

    assert out.result_body["environment"] == CPU_ENV
    assert out.result_body["environment"]["hardware"]["gpu"] == "cpu"
    assert out.result_body["coverage"] == "beyond-floor-coverage"
    assert out.result_body["coverage_diff"] == ["hardware.gpu"]  # the floor's own `not_covered`
    assert out.result_body["environment_diff"] == [
        "hardware.count", "hardware.device", "hardware.driver", "hardware.gpu"
    ]
    assert (out.verdict, out.exit_code) == ("beyond-floor-coverage", 2)
    assert out.result_body["identity_diff"] == []  # `device` is not identity, and does not become it
    # And the challenge a stranger would read names the CPU.
    body = V.challenge_body(out, out.result_body["environment"])
    assert body["environment"]["hardware"]["gpu"] == "cpu"


def test_a_gpu_run_against_the_gpu_floor_is_still_within_coverage():
    """The control: the repair does not make every verify beyond-coverage."""
    battery, cert, res = gpu_reference()
    out = V.ref(cert, DeviceRunner(GPU_ENV), res)
    assert out.result_body["coverage"] == "within"
    assert out.result_body["environment_diff"] == []
    assert (out.verdict, out.exit_code) == ("same", 0)


def test_the_challenge_the_cli_signs_carries_the_observed_environment(tmp_path):
    """The second half of the hole: `--challenge` re-read the file after `verify` had finished.

    A challenge is the body that travels furthest -- it is read by strangers holding neither the
    runner nor the box -- and S-DEVICE wrote its RTX 4070 into that one through this path even
    when the result body had been repaired.
    """
    battery, cert, res = gpu_reference()
    out = V.ref(cert, DeviceRunner(CPU_ENV), res)
    env_file = tmp_path / "env_real.json"
    env_file.write_text(json.dumps(GPU_ENV), encoding="utf-8", newline="\n")

    payload = cli._challenge(
        {"note": None, "own": None, "key": None, "challenge_out": None,
         "environment": str(env_file)},
        out,
        "ref",
    )

    assert payload["challenge"]["environment"] == CPU_ENV
    assert payload["challenge"]["coverage"] == "beyond-floor-coverage"


def test_without_the_guard_the_cpu_run_signs_the_rtx_4070_again(monkeypatch):
    """The mutation: put the replacement back and S-DEVICE returns, in both bodies.

    Every assertion above is worthless if this one does not fail when the guard is present, so
    it restores the exact line `verify.ref` used to run and shows what it produced.
    """
    battery, cert, res = gpu_reference()
    monkeypatch.setattr(V.runnermod, "observed_environment", _old_observed_environment)

    out = V.ref(cert, DeviceRunner(CPU_ENV), res, environment=copy.deepcopy(GPU_ENV))

    assert out.result_body["environment"]["hardware"]["gpu"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
    assert out.result_body["coverage"] == "within"
    assert out.result_body["coverage_diff"] == []
    assert out.mismatched == []
    assert out.result_body["identity_diff"] == []
    assert (out.verdict, out.exit_code) == ("same", 0)
    # And the accusation a stranger reads names the card the run never touched.
    body = V.challenge_body(out, out.result_body["environment"])
    assert body["environment"]["hardware"]["gpu"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
