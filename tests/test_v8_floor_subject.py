"""A floor is a statement about ONE subject, and the anchor never asked whose runs it added up.

A third adversarial pass against ``styxx.v8.log`` (scripts under the session scratchpad,
``reattack2/``) put the whole of it in one sentence: *the boundary slid from the floor's
arithmetic to the floor's provenance, and the issuer still writes the bytes.* Three findings:

* **A-SWAP** — the sharpest. ``Log._check_floor_matches_its_runs`` recomputes a floor's arithmetic
  from the bodies ``noise_floor.runs`` names and compares exactly; nothing in it, or in
  ``_check_floor_honours_its_plan``, or in ``_check_floor_names_the_plans_runs``, ever compared a
  named run's SUBJECT against the appending cert's. So the published fp16 canonical, relabelled
  ``run_index 1`` and dressed in a bf16 run's ``nuisance`` and ``recipe``, appended as a run of
  the bf16 canonical's floor: ``floor_disagreement()`` returned ``[]``, the signed floor was
  ``exact 0.0625 / seqlp 0.058564664 / topk 1.407087824`` against the honest bf16 floor's
  ``0.046875 / 0.036070694 / 2.140233900``, and a ``verify --diff`` of that cert against the fp16
  canonical read ``same`` on every channel. The cross-precision drift had been made the floor and
  the recomputation certified it to the last digit.
* **A-SPLIT** — ``_floor_body_certs`` decided whether the appending cert's own body belongs in the
  floor by looking for a named run carrying the same ``run_index`` and the same ``items``, which
  is a signature the appending party writes; and ``_check_floor_names_the_plans_runs`` counted the
  post-split list, so the completeness check moved with the signature.
* **R-EXEC-3** — relabelling ``recipe.decoding`` alongside ``body.nuisance`` defeats
  ``_check_floor_labels_match_the_recipe``, because both halves are the same signed cert written
  by the same party. **That one is not closed here and cannot be closed here**; see
  ``test_r_exec_3_is_open_and_the_module_says_so`` at the bottom, which pins the written-down
  limitation rather than a defence that does not hold.

Every test here is a refusal a demonstrated attack earned. Nothing here skips.
"""
from __future__ import annotations

import copy

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import log as logmod
from styxx.v8.log import AppendRefused, Log
from tests import v8_fixtures as F

ISSUER_SEED, ISSUER_PUB = F.keypair("issuer")
LOG_SEED, LOG_PUB = F.keypair("log-key")


# ----------------------------------------------------------------- helpers


def _roster() -> list[dict]:
    return [
        {
            "name": F.ISSUER_NAME,
            "key": F.public_key("issuer"),
            "from_index": 0,
            "retired_at_index": None,
        }
    ]


def _fresh(root) -> Log:
    return Log.init(root, LOG_PUB, _roster())


def _plan(*factors, runs: int = 5) -> dict:
    """A noise-plan prereg fixing R and declaring ``(factor, values)`` pairs (section 5.1 step 1)."""
    return F.make_cert(
        "prereg",
        body={
            "kind": "noise-plan",
            "runs": runs,
            "nuisance": [{"factor": f, "values": list(v)} for f, v in factors],
            "environment": {"hardware": {"gpu": "none", "driver": "none", "count": 0}},
        },
    )


def _resign(cert: dict) -> dict:
    core = {k: v for k, v in cert.items() if k not in ("id", "sig")}
    return certmod.sign(core, ISSUER_SEED)


BATCHES = ("8", "4")
SIZES = (8, 4, 8, 4, 8)  # run 0 .. run 4; the plan declares batch_size 8|4


def ladder(tmp_path, *, run_overrides=None, run_mutator=None, pre=(), canonical_overrides=None,
           named=None):
    """A log holding a battery, a plan fixing R = 5, four run certs, and the canonical over them.

    ``run_overrides`` is ``{run_index: {envelope key: value}}`` applied to a run cert before it is
    signed and appended — that is where a cross-subject run goes in. ``run_mutator(k, core, ctx)``
    is the same hook for a change that needs the battery or plan ids. ``pre`` are certs appended
    ahead of the battery. ``named`` overrides which certs ``noise_floor.runs`` names.

    The floor's numbers are always the ones the named bodies really produce, so every refusal
    below is about provenance and never about arithmetic: ``floor_disagreement`` is asserted empty
    wherever it can be computed.
    """
    run_overrides = run_overrides or {}
    log = _fresh(tmp_path / "log")
    for cert in pre:
        log.append(cert)
    battery = F.make_cert("battery", body=F.battery_body("fixed-v1", n=8))
    log.append(battery)
    plan = _plan(("batch_size", BATCHES))
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    ctx = {"battery": battery, "plan": plan, "log": log}

    runs = []
    for k in range(1, 5):
        body = F.fingerprint_body(n=8, run_index=k, batch_size=SIZES[k])
        core = {
            "recipe": F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=SIZES[k])),
            "refs": [{"role": "battery", "id": battery["id"]}] + plan_ref,
            "body": body,
        }
        core.update(copy.deepcopy(run_overrides.get(k, {})))
        if run_mutator is not None:
            run_mutator(k, core, ctx)
        cert = F.make_cert("fingerprint", **core)
        log.append(cert)
        runs.append(cert)

    chosen = runs if named is None else named
    body = F.fingerprint_body(n=8, run_index=0, batch_size=SIZES[0])
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in chosen],
        bodies=([body] if len(chosen) == 4 else []) + [c["body"] for c in chosen],
    )
    core = {
        "recipe": F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=SIZES[0])),
        "refs": (
            [{"role": "battery", "id": battery["id"]}]
            + plan_ref
            + [{"role": "run", "id": c["id"]} for c in chosen]
        ),
        "body": body,
    }
    core.update(copy.deepcopy(canonical_overrides or {}))
    canonical = F.make_cert("fingerprint", **core)
    return log, battery, plan, runs, canonical


# ----------------------------------------------------------------- A-SWAP


def test_the_honest_ladder_is_what_appends(tmp_path):
    """The control. Every refusal below differs from this by one field."""
    log, _, _, _, canonical = ladder(tmp_path)
    assert log.floor_disagreement(canonical) == []
    assert log.append(canonical) == log.size() - 1


def test_refuses_a_floor_resting_on_another_subjects_run(tmp_path):
    """A-SWAP, in one line: the fp16 body in the bf16 floor.

    The arithmetic is honest about the bodies named — that is the whole finding — so the anchor
    says nothing and ``floor_disagreement`` is empty. What is wrong is whose runs those are.
    """
    log, _, _, _, canonical = ladder(
        tmp_path,
        run_overrides={1: {"subject": F.weights_subject(precision="fp16")}},
    )
    assert log.floor_disagreement(canonical) == []  # the numerals ARE the named bodies'
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert reason.startswith("floor:")
    assert "a run of a different subject" in reason
    assert "'precision'" in reason and "fp16" in reason
    assert F.WEIGHTS_SUBJECT["precision"] in reason  # both sides are named, not just the offender
    assert "ONE subject" in reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights_sha256", "b" * 64),
        ("config_sha256", "c" * 64),
        ("tokenizer_sha256", "d" * 64),
        ("generation_config_sha256", "e" * 64),
    ],
)
def test_a_run_contradicting_the_snapshot_is_refused_an_append_earlier(tmp_path, field, value):
    """Four of the seven no longer reach the floor guard at all (M3).

    This was one parametrisation of seven ending at ``log.append(canonical)`` with A-SWAP's "a run
    of a different subject". A run that names the logged (hf_repo, revision) with a different
    Appendix A.2 hash now never gets that far: ``Log.append``'s M3 predicate
    (``snapshot_disagreement``) refuses the RUN itself, because one revision of one repository is
    one set of files and two certs hashing it two ways cannot both be right. The refusal moves an
    append earlier; the floor guard behind it is untouched, and the three remaining fields below
    still reach it.
    """
    with pytest.raises(AppendRefused) as exc:
        ladder(tmp_path, run_overrides={2: {"subject": F.weights_subject(**{field: value})}})
    reason = exc.value.reason
    assert reason.startswith("subject:")
    assert field in reason
    assert "(M3, section 2.2)" in reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("hf_repo", "acme/other-model"),
        ("revision", "0" * 40),
        ("precision", "fp16"),
    ],
)
def test_the_three_fields_m3_does_not_refuse_still_bind_the_floor(tmp_path, field, value):
    """Section 2.2 whole, not a chosen half — and these three are the floor guard's alone.

    ``precision`` is a load-time cast and moves no content hash, so M3 cannot see it. ``hf_repo``
    and ``revision`` moved alone leave the four A.2 hashes intact, which M3 *discloses* rather than
    refuses (an honest commit outside the A.2 list gives two revisions one quadruple —
    ``Log.snapshot_aliases``). All three therefore reach ``log.append(canonical)``, where a floor
    resting on another subject's run is refused: ``cert.comparable`` softens ``precision`` and
    ``revision`` to ``cross-subject:``, right for a *comparison* and wrong for a *run of one
    floor*, so the floor guard compares identity fields directly.
    """
    log, _, _, _, canonical = ladder(
        tmp_path, run_overrides={2: {"subject": F.weights_subject(**{field: value})}}
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert f"'{field}'" in exc.value.reason
    assert "a run of a different subject" in exc.value.reason


def test_refuses_a_floor_run_over_another_battery(tmp_path):
    """``recipe_core`` binds too (section 2.3): a floor across two batteries measures the battery.

    The second battery is a real cert in the same log, so nothing here is unresolvable — the run
    is well formed, signed, appended and refused only when the floor tries to rest on it.
    """
    other = F.make_cert("battery", body=F.battery_body("fixed-v1", n=8, source="other@v8"))

    def swap_battery(k, core, ctx):
        if k != 3:
            return
        core["recipe"] = F.recipe(battery=other["id"], decoding=F.decoding(batch_size=SIZES[3]))
        core["refs"] = [{"role": "battery", "id": other["id"]}] + [
            r for r in core["refs"] if r["role"] != "battery"
        ]

    log, _, _, _, canonical = ladder(tmp_path, pre=(other,), run_mutator=swap_battery)
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert "names recipe.battery" in reason and "ONE recipe core" in reason


def test_refuses_a_floor_run_at_a_decoding_key_no_plan_declared(tmp_path):
    """``seed`` is not a nuisance factor anybody preregistered, and a floor across two seeds is a
    measurement of the seed. ``batch_size`` and ``padding_side`` are the two the runs are handed
    by section 5.1 step 2; everything else in ``decoding`` has to be equal."""
    def another_seed(k, core, ctx):
        if k != 4:
            return
        core["recipe"] = F.recipe(
            battery=ctx["battery"]["id"], decoding=F.decoding(batch_size=SIZES[4], seed=99)
        )

    log, _, _, _, canonical = ladder(tmp_path, run_mutator=another_seed)
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "names recipe.decoding ['seed']" in exc.value.reason
    assert "the plan declared" in exc.value.reason


def test_a_run_may_vary_the_decoding_the_plan_committed_to(tmp_path):
    """The control for the rule above, and the reason it is not simple recipe_core equality: the
    lab's own published floor has runs at ``decoding.batch_size`` 8 and 32 under a canonical at 1
    (``papers/v8/first_verdict_2026_09_09``). Refusing that would refuse every real floor."""
    log, _, _, runs, canonical = ladder(tmp_path)
    decodings = {c["recipe"]["decoding"]["batch_size"] for c in runs}
    assert decodings == {8, 4}  # the runs really do differ from the canonical's recipe_core
    assert canonical["recipe"]["decoding"]["batch_size"] == 8
    assert log.append(canonical) == log.size() - 1


def test_the_check_reads_the_run_certs_the_log_holds(tmp_path):
    """Not a copy the appending cert carries: ``_floor_run_certs`` resolves every id against this
    log, and ``_check_floor_matches_its_runs`` already refuses a named run the log does not hold.
    So the subject a floor is bound to is the subject of the entry a stranger can read back."""
    log, _, _, runs, canonical = ladder(
        tmp_path, run_overrides={1: {"subject": F.weights_subject(precision="fp16")}}
    )
    at = log.find(runs[0]["id"])
    assert log.cert(at)["subject"]["precision"] == "fp16"
    with pytest.raises(AppendRefused):
        log.append(canonical)


# ----------------------------------------------------------------- A-SPLIT


def _mirror(canonical: dict, runs: list[dict], **body_over) -> dict:
    """A cert claiming to BE run 0: the canonical's ``run_index``, and by default its body."""
    m = copy.deepcopy(runs[0])
    body = dict(m["body"])
    body["run_index"] = canonical["body"]["run_index"]
    body["items"] = copy.deepcopy(canonical["body"]["items"])
    body["channels"] = copy.deepcopy(canonical["body"]["channels"])
    body["nuisance"] = copy.deepcopy(canonical["body"]["nuisance"])
    body.pop("noise_floor", None)
    body.update(copy.deepcopy(body_over))
    m["body"] = body
    m["recipe"] = copy.deepcopy(canonical["recipe"])
    return _resign(m)


def test_the_two_floor_shapes_give_one_number(tmp_path):
    """A-SPLIT's premise, made harmless. ``floor.floors`` reads ``items`` and ``channels`` and
    nothing else, so once the named run 0 is required to carry the canonical's ``items`` and
    ``channels``, it does not matter which of the two bodies the anchor counts: the R-id shape and
    the R−1 shape produce the same floor, to the digit."""
    log, battery, plan, runs, r_minus_one = ladder(tmp_path)
    mirror = _mirror(r_minus_one, runs)
    log.append(mirror)
    chosen = [mirror] + runs
    body = copy.deepcopy(r_minus_one["body"])
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in chosen],
        bodies=[c["body"] for c in chosen],  # the R-id shape: the canonical's body is NOT here
    )
    r_shape = F.make_cert(
        "fingerprint",
        recipe=copy.deepcopy(r_minus_one["recipe"]),
        refs=(
            [r for r in r_minus_one["refs"] if r["role"] != "run"]
            + [{"role": "run", "id": c["id"]} for c in chosen]
        ),
        body=body,
    )
    assert r_shape["id"] != r_minus_one["id"]
    assert (
        r_shape["body"]["noise_floor"]["per_channel"]
        == r_minus_one["body"]["noise_floor"]["per_channel"]
    )
    assert log.floor_disagreement(r_shape) == []
    assert log.append(r_shape) == log.size() - 1


def test_the_split_is_decided_by_the_plans_r_and_not_by_the_appending_certs_signature(tmp_path):
    """A-SPLIT. ``R`` is the one number in a floor committed BEFORE the runs, on a cert this floor
    names by id, so ``R`` decides which shape this is — and ``_floor_body_certs`` takes it."""
    log, _, plan, runs, canonical = ladder(tmp_path)
    named = log._floor_run_certs(canonical)[1:]
    assert log._floor_declared_runs(canonical) == 5
    # 4 named ids under R = 5: the R−1 shape, the canonical's own body prepended.
    assert len(Log._floor_body_certs(canonical, named, 5)) == 5
    assert Log._floor_body_certs(canonical, named, 5)[0] is canonical
    # 4 named ids under a plan fixing R = 4 would be the R-id shape — the number decides, and
    # nothing the appending cert says about itself takes part.
    assert Log._floor_body_certs(canonical, named, 4) == named


def test_refuses_an_r_shaped_floor_whose_run_zero_is_not_this_certs_run(tmp_path):
    """The mirror carries the canonical's ``run_index`` and a different ``channels`` block, which
    is exactly the freedom A-SPLIT named: whichever body the anchor counts changes the answer.
    Refused, because the two bodies claiming one run disagree about what the floor reads."""
    log, battery, plan, runs, r_minus_one = ladder(tmp_path)
    channels = {
        k: v for k, v in r_minus_one["body"]["channels"].items() if k != "topk"
    }
    mirror = _mirror(r_minus_one, runs, channels=channels)
    log.append(mirror)
    _, _, _, _, template = ladder(tmp_path / "b")
    # rebuild the canonical naming all five, with the numbers those five bodies really give
    body = copy.deepcopy(r_minus_one["body"])
    chosen = [mirror] + runs
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in chosen],
        bodies=[c["body"] for c in chosen],
    )
    forged = F.make_cert(
        "fingerprint",
        recipe=copy.deepcopy(r_minus_one["recipe"]),
        refs=(
            [r for r in r_minus_one["refs"] if r["role"] != "run"]
            + [{"role": "run", "id": c["id"]} for c in chosen]
        ),
        body=body,
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    reason = exc.value.reason
    assert "is named as run 0 of this floor" in reason
    assert "body.channels is not this cert's body.channels" in reason


def test_refuses_an_r_minus_one_shaped_floor_whose_named_run_claims_run_zero(tmp_path):
    """The other side: four named runs under R = 5 makes the canonical run 0 itself, so a named
    run carrying index 0 would be counted twice and the advertised pair count would be R+1's."""
    log, battery, plan, runs, canonical = ladder(tmp_path)
    mirror = _mirror(canonical, runs)
    log.append(mirror)
    chosen = [mirror] + runs[:3]
    body = copy.deepcopy(canonical["body"])
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in chosen],
        bodies=[body] + [c["body"] for c in chosen],
    )
    forged = F.make_cert(
        "fingerprint",
        recipe=copy.deepcopy(canonical["recipe"]),
        refs=(
            [r for r in canonical["refs"] if r["role"] != "run"]
            + [{"role": "run", "id": c["id"]} for c in chosen]
        ),
        body=body,
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert "claims that index too" in exc.value.reason


# ----------------------------------------------------------------- R-EXEC-3, written down


def test_r_exec_3_is_open_and_the_module_says_so():
    """R-EXEC-3 relabels ``recipe.decoding`` alongside ``body.nuisance``. Both halves are one
    cert, one key, one party; a predicate over those bytes can only ask whether the party
    contradicted itself, and a party that does not contradict itself is not caught by asking.

    There is no repair to pin, so what is pinned is the statement — that the gap is open, and that
    what detects it is a second party running the same battery and publishing its own floor, not a
    check invented here. A test that asserted a defence would be worse than the gap.
    """
    doc = logmod.Log._check_floor_labels_match_the_recipe.__doc__ or ""
    assert "R-EXEC-3" in doc
    assert "open" in logmod.__doc__ or "OPEN" in logmod.__doc__
    assert "second party" in doc
    assert "It is a tax, not a wall." in doc
