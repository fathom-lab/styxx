"""The subject guard: `verify --ref` must check that it loaded the model the cert names.

The defect this file pins is C2 of `papers/v8/challenge_and_attack_2026_09_09`.  `verify --ref`
read an OPTIONAL `subject` member off the runner; `TransformersRunner` had none, so
`_observed_subject` fell back to the cert's own subject and `comparable(cert, cert)` was
trivially empty.  Running

    python -m styxx.v8 verify --ref <a bf16 cert> --runner hf --snapshot <gemma-2-2b-it>
                              --dtype float16

therefore loaded OTHER WEIGHTS, reported `identity_diff: []`, `mismatched: []`, coverage
`within`, and — with the confirmation run — signed a `drift` accusation against another party's
cert from a run that cert never named.

Two things are asserted here, and the second is what makes the first mean anything:

1. a runner that runs something other than the cert's subject produces `identity` or a refusal,
   never `drift` and never `same`;
2. with the guard removed — `_observed_subject` monkeypatched back to `dict(fallback)`, the
   exact line that was there — the SAME setup produces `drift` and `same`.  Every test in (1)
   fails if the guard is deleted, and the tests in (2) fail if the fallback ever returns.

The fixtures are the ones the section 6 exit-code table is driven with, imported from
`tests/test_v8_verify.py` so that the exploit is run against the same reference cert.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as FP
from styxx.v8 import runner as runnermod
from styxx.v8 import verify as V
from styxx.v8.consts import EXIT
from styxx.v8.runner import MockRunner, SubjectUnavailable
from tests import v8_fixtures as F
from tests.test_v8_verify import pool_cert, reference

ROOT = Path(__file__).resolve().parent.parent
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The two subjects of the operator's own run: one snapshot, two dtypes.  Only `precision`
# differs — the weights files on disk are the same bytes, which is exactly why the fallback
# could not see it.
OTHER_PRECISION = {"precision": "fp16"}
OTHER_WEIGHTS = {"weights_sha256": "e" * 64}


def _defective_observed_subject(runner, fallback):
    """`verify._observed_subject` as it behaved before this repair.

    The old body read an optional `subject` member and fell back to `dict(fallback)`.  With
    `TransformersRunner` — the runner the exploit used — there was no such member, so the
    fallback WAS the whole function, and that is what is restored here: the cert's own subject,
    handed back to be compared against the cert.
    """
    return dict(fallback)


def _echo_identity(runner, requested):
    """`runner.reported_identity` with the same hole: the request, echoed back as the report.

    `run_fingerprint` is the second line of the same guard, so a mutation that only restores
    the `verify` fallback is caught at mint time.  Restoring both is what the tree looked like
    when the exploit was run.
    """
    return runnermod.subject_identity(requested)


def _remove_both_guards(monkeypatch):
    monkeypatch.setattr(V, "_observed_subject", _defective_observed_subject)
    monkeypatch.setattr(FP.runnermod, "reported_identity", _echo_identity)


class NoSubjectRunner:
    """A runner from before the obligation: `run` and `environment`, and nothing else."""

    def run(self, items, recipe, subject):
        return MockRunner().run(items, recipe, subject)

    def environment(self):
        return MockRunner().environment()


class HoleRunner(MockRunner):
    """A runner that reports an identity with one field missing."""

    def subject(self, requested):
        reported = dict(super().subject(requested))
        reported.pop("weights_sha256")
        return reported


# --------------------------------------------------------------- the exploit, both directions


def test_a_runner_that_ran_other_weights_is_identity_and_never_drift():
    """The C2 shape with the hashes moved: a `drift` verdict is not available for it."""
    battery, cert, res = reference()
    swapped = F.weights_subject(**OTHER_WEIGHTS)
    runner = MockRunner(loaded_subject=swapped, drift_items={"i01"})

    out = V.ref(cert, runner, res)

    assert out.verdict == "identity (weights_sha256)"
    assert out.exit_code == EXIT["identity"] == 1
    assert out.result_body["identity_diff"] == ["weights_sha256"]
    assert "drift" not in out.verdict


def test_without_the_guard_that_same_run_is_a_signed_drift_accusation(monkeypatch):
    """The mutation: put the fallback back and the accusation returns."""
    battery, cert, res = reference()
    swapped = F.weights_subject(**OTHER_WEIGHTS)
    runner = MockRunner(loaded_subject=swapped, drift_items={"i01"})

    _remove_both_guards(monkeypatch)
    out = V.ref(cert, runner, res)

    assert out.verdict == "drift"
    assert out.exit_code == EXIT["drift"] == 1
    assert out.result_body["identity_diff"] == []
    assert out.mismatched == []
    assert out.result_body["coverage"] == "within"
    # And it is challengeable: a body that names another party's cert as the target.
    assert V.challenge_body(out, MockRunner().environment())["coverage"] == "within"


def test_a_precision_swap_is_refused_and_cannot_become_a_challenge():
    """The operator's own run: same snapshot, `--dtype float16`, a bf16 cert.

    `precision` is inside S_identity (section 2.2) but section 5.2's `identity` names only the
    four A.2 hashes, so a `--ref` that did not re-run the cert's subject refuses to compare —
    `cross-subject` is a `--diff` affordance (section 6).
    """
    battery, cert, res = reference()
    runner = MockRunner(loaded_subject=F.weights_subject(**OTHER_PRECISION))

    out = V.ref(cert, runner, res)

    assert out.exit_code == EXIT["mismatch"] == 3
    assert out.mismatched == ["cross-subject:precision"]
    assert out.result_body["per_channel"] == {}
    with pytest.raises(ValueError):
        V.challenge_body(out, MockRunner().environment())


def test_without_the_guard_a_precision_swap_signs_a_bare_same(monkeypatch):
    """The worse half of C2: fp16 weights, a bf16 cert, exit 0.

    The mock moves nothing under a precision change here, so the fallback does not merely lose
    a distance — it certifies agreement with a model it never ran.
    """
    battery, cert, res = reference()
    runner = MockRunner(loaded_subject=F.weights_subject(**OTHER_PRECISION))

    _remove_both_guards(monkeypatch)
    out = V.ref(cert, runner, res)

    assert (out.verdict, out.exit_code) == ("same", EXIT["same"]) == ("same", 0)
    assert out.result_body["identity_diff"] == []


def test_the_mint_guard_is_the_second_line_when_the_verify_guard_is_removed(monkeypatch):
    """Restore only the `verify` fallback: `run_fingerprint` still refuses to produce the body.

    The outcome is `unavailable` — no distances, no verdict, no challenge — which is a refusal
    and not an answer.  It is written down so that a later reader knows which of the two guards
    each mutation measures.
    """
    battery, cert, res = reference()
    runner = MockRunner(loaded_subject=F.weights_subject(**OTHER_WEIGHTS), drift_items={"i01"})

    monkeypatch.setattr(V, "_observed_subject", _defective_observed_subject)
    out = V.ref(cert, runner, res)

    assert (out.verdict, out.exit_code) == ("unavailable", 5)
    assert "runs a different subject" in out.result_body["unavailable_reason"]


# --------------------------------------------------------------- fail closed, not fail quiet


def test_a_runner_that_does_not_report_its_subject_is_unavailable():
    """An optional guard is the defect being repaired: no report, no verdict."""
    battery, cert, res = reference()

    out = V.ref(cert, NoSubjectRunner(), res)

    assert (out.verdict, out.exit_code) == ("unavailable", EXIT["unavailable"]) == ("unavailable", 5)
    assert out.result_body["per_channel"] == {}
    assert "does not report the subject it runs" in out.result_body["unavailable_reason"]
    assert out.result_body["attempted"]["subject.kind"] == "weights"


def test_a_runner_that_reports_a_hole_is_unavailable():
    battery, cert, res = reference()

    out = V.ref(cert, HoleRunner(), res)

    assert (out.verdict, out.exit_code) == ("unavailable", 5)
    assert "weights_sha256" in out.result_body["unavailable_reason"]
    assert out.result_body["per_channel"] == {}


def test_the_reported_identity_helper_refuses_every_way_of_saying_nothing():
    subject = F.weights_subject()

    with pytest.raises(SubjectUnavailable, match="does not report"):
        runnermod.reported_identity(NoSubjectRunner(), subject)

    class NotAMapping(MockRunner):
        def subject(self, requested):
            return "gemma-2-2b-it"

    with pytest.raises(SubjectUnavailable, match="not a subject mapping"):
        runnermod.reported_identity(NotAMapping(), subject)

    class WrongKind(MockRunner):
        def subject(self, requested):
            return {"kind": "checkpoint", "weights_sha256": "a" * 64}

    with pytest.raises(SubjectUnavailable, match="reported kind"):
        runnermod.reported_identity(WrongKind(), subject)

    class Raising(MockRunner):
        def subject(self, requested):
            raise FileNotFoundError("snapshot directory not found")

    with pytest.raises(SubjectUnavailable, match="FileNotFoundError"):
        runnermod.reported_identity(Raising(), subject)


def test_an_honest_runner_still_verifies_and_the_identity_comes_from_it_alone():
    """The mock really is a function of the identity it is handed, so it reports it back."""
    battery, cert, res = reference()

    out = V.ref(cert, MockRunner(), res)

    assert (out.verdict, out.exit_code) == ("same", 0)
    assert out.result_body["identity_diff"] == []
    # Advisory fields come from the cert; identity comes from the runner.
    merged = V._observed_subject(MockRunner(loaded_subject=F.weights_subject(**OTHER_WEIGHTS)),
                                 cert["subject"])
    assert merged["weights_sha256"] == "e" * 64
    assert merged["model_family"] == cert["subject"]["model_family"]
    assert merged["environment"] == cert["subject"]["environment"]


# --------------------------------------------------------------- the same guard at mint time


def test_minting_a_fingerprint_with_a_runner_that_runs_other_weights_is_refused():
    """A body records `subject`; a body may not name weights that did not produce it."""
    battery = pool_cert()
    runner = MockRunner(loaded_subject=F.weights_subject(**OTHER_PRECISION))

    with pytest.raises(ValueError, match="runs a different subject"):
        FP.run_fingerprint(
            runner, F.weights_subject(), F.recipe(), battery, run_index=0, nuisance={}
        )


def test_minting_with_a_runner_that_will_not_say_is_unavailable_not_a_cert():
    battery = pool_cert()

    with pytest.raises(SubjectUnavailable):
        FP.run_fingerprint(
            NoSubjectRunner(), F.weights_subject(), F.recipe(), battery, run_index=0, nuisance={}
        )


# --------------------------------------------------------------- the two S_identity tables


def test_the_runner_table_is_the_certs_table():
    """`runner.SUBJECT_IDENTITY` restates `cert.IDENTITY_FIELDS`; drift between them is the bug."""
    assert runnermod.SUBJECT_IDENTITY == certmod.IDENTITY_FIELDS
    assert "precision" in runnermod.SUBJECT_IDENTITY["weights"]


def test_the_mock_reports_kind_and_every_identity_field_and_nothing_else():
    reported = MockRunner().subject(F.weights_subject())
    assert set(reported) == {"kind", *certmod.IDENTITY_FIELDS["weights"]}
    reported_alias = MockRunner().subject(F.alias_subject())
    assert set(reported_alias) == {"kind", *certmod.IDENTITY_FIELDS["alias"]}


# --------------------------------------------------------------- the real runner, torch-free


def _snapshot(root: Path) -> Path:
    snap = root / "models--Acme--Tiny-Instruct" / "snapshots" / "0123abcd"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"shard bytes")
    (snap / "config.json").write_bytes(b'{"model_type": "tiny"}')
    (snap / "generation_config.json").write_bytes(b'{"do_sample": false}')
    (snap / "tokenizer.json").write_bytes(b"tok json")
    return snap


def test_the_hf_runner_reports_the_snapshot_it_holds_and_the_dtype_it_will_load(tmp_path):
    from styxx.v8.runner_hf import TransformersRunner, snapshot_hashes, subject_from_snapshot

    snap = _snapshot(tmp_path)
    env = {
        "runtime": {"framework": "transformers", "version": "0", "backend": "torch 0"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
    }
    cert_subject = subject_from_snapshot(str(snap), "tiny", "bf16", environment=env)

    honest = TransformersRunner(str(snap), dtype="bfloat16").subject(cert_subject)
    assert honest == {
        "kind": "weights",
        "hf_repo": "Acme/Tiny-Instruct",
        "revision": "0123abcd",
        **snapshot_hashes(snap),
        "precision": "bf16",
    }
    assert runnermod.reported_identity(TransformersRunner(str(snap)), cert_subject) == honest

    # The operator's exploit, at the source: --dtype float16 against a bf16 cert.
    swapped = TransformersRunner(str(snap), dtype="float16").subject(cert_subject)
    assert swapped["precision"] == "fp16"
    assert certmod.comparable({"subject": cert_subject}, {"subject": swapped}) == [
        "cross-subject:precision"
    ]

    # A cert spelling the same dtype differently keeps its spelling: bf16 and bfloat16 are one
    # dtype, and a rename is not an identity difference.
    spelled = dict(cert_subject, precision="bfloat16")
    assert TransformersRunner(str(snap), dtype="bfloat16").subject(spelled)["precision"] == "bfloat16"
    assert all(HEX64.match(honest[k]) for k in snapshot_hashes(snap))


def test_the_hf_runner_reports_without_loading_the_model(tmp_path):
    """The guard must fire on a box with no GPU and no torch: it is a claim about bytes."""
    snap = _snapshot(tmp_path)
    code = (
        "import sys, json;"
        "from styxx.v8.runner_hf import TransformersRunner;"
        "s = TransformersRunner(%r, dtype='float16').subject({'precision': 'bf16'});"
        "print(json.dumps([s['precision'], 'torch' in sys.modules, 'transformers' in sys.modules]))"
        % snap.as_posix()
    )
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip().endswith('["fp16", false, false]')
