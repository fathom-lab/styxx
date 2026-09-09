"""The whole styxx.v8 ladder on the mock runner (contract section 11).

Two ladders are built end to end, one per subject kind, each into its own temporary log:

    pool-v1 battery cert (a ROOT: empty recipe, no refs -- section 4.5)
      -> study prereg (the section 4.6 sensitivity hypothesis, on record before the sweep)
      -> reference fingerprint on the pool
      -> delta sweep -> canary-v1 battery (weights) / fixed-v1 battery (alias)
      -> noise-plan prereg
      -> 5 runs at batch 1 varying only the item order
      -> the sensitivity receipt (a positive control measured against that floor)
      -> the canonical fingerprint carrying the floor and the sensitivity id
      -> a drifted fingerprint of the same subject
      -> verify --diff (exceeds_floor, exit 2) and verify --ref (drift, exit 1)
      -> the two verify result certs
      -> tree heads, the four verify commands, a mirror, and a tampered mirror

One further ladder is built at the bottom of this file through the COMMAND LINE alone, on the
same mock runner: ``battery pool``, ``prereg noise-plan``, ``fingerprint --runs 5 --plan``, and
``log append`` in the section 5.1 step 5 order -- the plan, the R-1 non-canonical run certs, the
canonical fingerprint carrying the floor.  It exists because the run recorded in
``papers/v8/first_log_2026_09_09/`` could not mint the plan at all, so no floor could be
constructed and the canonical cert came out byte-identical to run 0.

Every number asserted here is derived from the mock's construction, with the arithmetic beside
it: at batch 1 the mock moves nothing under a reordering, so every floor is 0.0 and one drifted
item out of eight is an exact distance of 1/8 = 0.125.  Nothing skips.

What the ladder had to decide, and why (each is pinned by a test below):

* **The canary selection is unavailable to an alias without log-probs.**  Appendix C needs
  ``margin_by_position`` and ``stay``; ``battery.score`` refuses without them.  The alias ladder
  therefore carries a ``fixed-v1`` battery, which is the section 4.5 kind for a public,
  model-agnostic item set, and the two ladders are otherwise identical.
* **The sensitivity control differs by kind.**  Section 5.3 wants the delta-1 precision variants
  and one committed perturbation.  The weights ladder measures the precision variant (the mock
  moves ``precision_items`` when ``subject.precision`` changes); an alias subject has no
  precision at all, so its receipt rests on the committed perturbation alone.
* **Both sides absent is not "skipped".**  ``verify`` lists a channel under
  ``skipped_channels`` when it is present on one side only (section 5.2).  An alias fingerprint
  compared against another run with no log-probs has ``seqlp``/``topk`` absent on both sides:
  they are neither compared nor skipped, and the ``skipped_channels`` list is empty.  The
  contract's "listed under skipped_channels" is the asymmetric case, which is exercised
  separately by re-running the alias cert against a runner that does carry log-probs.
* **The drifted fingerprint is the same subject, not a second one.**  Two certs whose subject
  identity fields differ are not comparable and a ``--diff`` between them exits 3 (section 2.3),
  so the drift leg re-runs the logged subject on a runner whose outputs moved.
* **Nothing in either ladder reaches the log's genesis exemption.**  The pool battery is a root:
  ``recipe`` is ``{}``, it embeds no cert id, and it carries no refs, so an empty log accepts it
  at index 0 on the ordinary rule that every ref resolves.  ``log.GENESIS_UNRESOLVED`` is asserted
  to be untouched by both ladders below, which is the evidence its own comment asks for.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from styxx.v8 import battery as B
from styxx.v8 import cert as certmod
from styxx.v8 import cli
from styxx.v8 import keys
from styxx.v8 import distances as D
from styxx.v8 import fingerprint as FP
from styxx.v8 import floor as floormod
from styxx.v8 import sweep as SW
from styxx.v8 import verify as V
from styxx.v8.consts import EXIT
from styxx.v8.jcs import canonical_bytes
from styxx.v8.log import (
    GENESIS_UNRESOLVED,
    AppendRefused,
    Log,
    mirror,
    verify_consistency,
    verify_entry,
    verify_inclusion,
    verify_sth,
)
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F

# --------------------------------------------------------------------------- material

ISSUER_SEED, ISSUER_PUB = F.keypair("issuer")
LOG_SEED, LOG_PUB = F.keypair("log-key")

TS_MID = "2026-09-08T18:30:00Z"
TS_END = "2026-09-08T19:00:00Z"

MOCK_ENVIRONMENT = MockRunner().environment()

POOL_FAMILIES = ("recall", "short-reasoning", "instruction-following", "format")
POOL_SIZE = 12

# Items the mock moves when ``batch_size != 1`` (section 4.2 delta-2): they carry flip2 > 0 and
# section 4.4 step 1 excludes every one of them from the canary battery.
NUISANCE_ITEMS = frozenset({"p00", "p05"})

# The delta-1 family.  Every item of one pool family responds to the precision change, so
# whichever two of that family the selection takes, the battery retains precision sensitivity.
DELTA1_PRECISION = "fp16"
PRECISION_FAMILY = "format"

N_CANARIES = 8
K_ANCHORS = 1
PERM_SEED = 11

# The order the log ends up in; the ladder builder appends in exactly this order.
LADDER_ORDER = (
    "pool",
    "study_prereg",
    "reference",
    "battery",
    "noise_plan",
    "run0",
    "run1",
    "run2",
    "run3",
    "run4",
    "sensitivity",
    "canonical",
    "drifted",
    "diff_result",
    "ref_result",
)


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


PRECISION_ITEMS = frozenset(
    item["item_id"] for item in pool_items() if item["family"] == PRECISION_FAMILY
)


def make_runner(*, drift=(), logprobs: bool = True) -> MockRunner:
    return MockRunner(
        nuisance_items=set(NUISANCE_ITEMS),
        drift_items=set(drift),
        precision_items={DELTA1_PRECISION: set(PRECISION_ITEMS)},
        logprobs=logprobs,
    )


def plan_orders(item_ids) -> list:
    """The five item orders of the nuisance plan; ``None`` is A.3 order (section 5.1 step 2)."""
    ids = sorted(item_ids)
    return [
        None,
        list(reversed(ids)),
        ids[1:] + ids[:1],
        ids[2:] + ids[:2],
        ids[::2] + ids[1::2],
    ]


def roles_of(body: dict) -> dict:
    return {item["item_id"]: item.get("role", "item") for item in body["items"]}


def channel_distance(a_body: dict, b_body: dict, channel: str, roles: dict) -> float:
    """Appendix B distance between two fingerprint bodies, through the public API."""
    if channel == "exact":
        return D.rounded(float(D.exact(a_body["items"], b_body["items"], roles)[0]))
    if channel == "seqlp":
        return D.rounded(float(D.seqlp(a_body["items"], b_body["items"])))
    if channel == "topk":
        return D.rounded(float(D.topk(a_body["items"], b_body["items"])))
    raise AssertionError(f"the mock ladder carries no {channel} channel")


def item_map(body: dict) -> dict:
    return {item["item_id"]: item for item in body["items"]}


# --------------------------------------------------------------------------- the ladder


def build_ladder(root: Path, kind: str) -> dict:
    """Build one whole ladder into a fresh log at ``root`` and return every piece of it."""
    logprobs = kind == "weights"
    subject = (
        F.weights_subject(environment=copy.deepcopy(MOCK_ENVIRONMENT))
        if kind == "weights"
        else F.alias_subject(environment=copy.deepcopy(MOCK_ENVIRONMENT))
    )
    runner = make_runner(logprobs=logprobs)
    log = Log.init(
        root,
        LOG_PUB,
        [{"name": F.ISSUER_NAME, "key": F.public_key("issuer"), "from_index": 0, "retired_at_index": None}],
    )
    certs: dict[str, dict] = {}

    def add(name: str, cert: dict) -> dict:
        log.append(cert)
        certs[name] = cert
        return cert

    # 1. the pool battery: a root (section 4.5). It names no battery, embeds no cert id and
    # carries no ref, so an empty log takes it at index 0 without any exemption.
    pool_body = B.pool_v1(pool_items())
    pool = add(
        "pool",
        F.make_cert("battery", subject=subject, recipe={}, body=pool_body, refs=[]),
    )
    pool_recipe = F.recipe(battery=pool["id"])

    # 2. the sensitivity hypothesis, on record before the sweep (section 4.6)
    add(
        "study_prereg",
        F.make_cert(
            "prereg",
            body={
                "kind": "study",
                "sealed": False,
                "hypotheses": [
                    {"id": "H-sensitivity", "direction": "greater", "endpoint": "exact"}
                ],
                "grader": {"kind": "gold-labels", "id": "styxx-e2e@v8"},
                "held_out": sorted(item["item_id"] for item in pool_items())[N_CANARIES:],
            },
        ),
    )

    # 3. the reference fingerprint on the pool (what the canary battery is selected against)
    reference_body = FP.run_fingerprint(
        runner, subject, pool_recipe, pool, run_index=0, nuisance={}
    )
    reference = add(
        "reference",
        F.make_cert(
            "fingerprint",
            subject=subject,
            recipe=pool_recipe,
            body=reference_body,
            refs=[{"role": "battery", "id": pool["id"]}],
        ),
    )

    # 4. the delta sweep (section 4.2) and the battery it selects
    delta2 = [
        {"batch_size": 8},
        {"batch_size": 8, "order": "perm", "perm_seed": PERM_SEED},
    ]
    sweep_record = SW.run_sweep(
        lambda precision: runner,
        pool_items(),
        pool_recipe,
        subject,
        delta1=[DELTA1_PRECISION] if kind == "weights" else [],
        delta2=delta2,
    )
    scores = None
    if kind == "weights":
        # canary-v1 is the one battery kind that is NOT a root: it was selected against a
        # fingerprint that ran on a pool, so it names that pool in its recipe (section 4.5).
        scores = B.score(sweep_record)
        battery_body = B.select(
            pool_items(), scores, n=N_CANARIES, k=K_ANCHORS, perm_seed=PERM_SEED
        )
        battery_recipe = pool_recipe
        battery_refs = F.canary_battery_refs(pool["id"], reference["id"])
    else:
        # Appendix C needs log-probs; an alias without them carries a public fixed set instead.
        # fixed-v1 is a root kind: no recipe, and the pool ref is provenance, not a dependency
        # the schema forces (nothing in this body embeds a cert id).
        chosen = [item for item in pool_items() if item["item_id"] not in NUISANCE_ITEMS][:N_CANARIES]
        battery_body = B.fixed_v1(chosen, source="styxx-e2e@v8")
        battery_recipe = {}
        battery_refs = [{"role": "pool", "id": pool["id"]}]
    battery_cert = add(
        "battery",
        F.make_cert(
            "battery",
            subject=subject,
            recipe=battery_recipe,
            body=battery_body,
            refs=battery_refs,
        ),
    )
    battery_ids = sorted(item["item_id"] for item in battery_body["items"])
    recipe = F.recipe(battery=battery_cert["id"])

    # 5. the nuisance plan, logged before the runs (section 5.1)
    orders = plan_orders(battery_ids)
    plan = add(
        "noise_plan",
        F.make_cert(
            "prereg",
            body={
                "kind": "noise-plan",
                "runs": len(orders),
                "batch_size": 1,
                "factor": "item order",
                # Section 5.4's two lists are DERIVED from what the plan fixed, and the log
                # refuses a floor whose lists are not that derivation (A-COVER). This plan
                # enumerates no `nuisance` block, so it covers nothing and holds every leaf of
                # the environment it names fixed -- and it has to name one, because an empty
                # `not_covered` is the claim to cover every environment there is.
                "environment": copy.deepcopy(subject["environment"]),
            },
        ),
    )

    # 6. the five runs of the plan
    run_bodies = [
        FP.run_fingerprint(
            runner, subject, recipe, battery_cert, run_index=k, nuisance={}, order=order
        )
        for k, order in enumerate(orders)
    ]
    run_certs = []
    for k, body in enumerate(run_bodies):
        run_certs.append(
            add(
                f"run{k}",
                F.make_cert(
                    "fingerprint",
                    subject=subject,
                    recipe=recipe,
                    body=body,
                    refs=[
                        {"role": "battery", "id": battery_cert["id"]},
                        {"role": "noise_plan", "id": plan["id"]},
                    ],
                ),
            )
        )

    floored = FP.attach_floor(
        run_bodies[0],
        run_bodies,
        plan["id"],
        [c["id"] for c in run_certs],
        *floormod.plan_coverage(plan["body"]),
    )
    floor_block = floored["noise_floor"]

    # 7. the sensitivity receipt: a positive control measured against that floor (section 5.3)
    roles = roles_of(run_bodies[0])
    if kind == "weights":
        control_subject = dict(subject)
        control_subject["precision"] = DELTA1_PRECISION
        control_runner = runner
        control_note = {
            "kind": "delta1-precision",
            "precision": DELTA1_PRECISION,
            "moved_items": sorted(set(battery_ids) & PRECISION_ITEMS),
        }
    else:
        control_subject = subject
        control_runner = make_runner(drift=set(battery_ids[:2]), logprobs=logprobs)
        control_note = {
            "kind": "committed-perturbation",
            "moved_items": sorted(battery_ids[:2]),
        }
    control_body = FP.run_fingerprint(
        control_runner, control_subject, recipe, battery_cert, run_index=0, nuisance={}
    )
    sensitivity_per_channel = {}
    for channel, block in floor_block["per_channel"].items():
        distance = channel_distance(run_bodies[0], control_body, channel, roles)
        sensitivity_per_channel[channel] = {
            "distance": distance,
            "floor": block["floor"],
            "exceeded": distance > block["floor"],
        }
    sensitivity = add(
        "sensitivity",
        F.make_cert(
            "result",
            body={
                "kind": "sensitivity",
                "deviations": [],
                "control": control_note,
                "per_channel": sensitivity_per_channel,
                "rounding": D.ROUND_PLACES,
            },
            refs=[
                {"role": "prereg", "id": certs["study_prereg"]["id"]},
                {"role": "battery", "id": battery_cert["id"]},
            ],
        ),
    )

    # 8. the canonical fingerprint: the floor and the sensitivity receipt it rests on
    canonical_body = copy.deepcopy(floored)
    canonical_body["sensitivity"] = sensitivity["id"]
    canonical = add(
        "canonical",
        F.make_cert(
            "fingerprint",
            subject=subject,
            recipe=recipe,
            body=canonical_body,
            refs=[
                {"role": "battery", "id": battery_cert["id"]},
                {"role": "noise_plan", "id": plan["id"]},
                {"role": "sensitivity", "id": sensitivity["id"]},
            ]
            + [{"role": "run", "id": c["id"]} for c in run_certs],
        ),
    )

    # 9. the same subject on a runner whose outputs moved
    drift_item = battery_ids[0]
    drift_runner = make_runner(drift={drift_item}, logprobs=logprobs)
    drifted_body = FP.run_fingerprint(
        drift_runner, subject, recipe, battery_cert, run_index=0, nuisance={}
    )
    probe = F.make_cert(
        "fingerprint",
        subject=subject,
        recipe=recipe,
        body=drifted_body,
        refs=[{"role": "battery", "id": battery_cert["id"]}],
    )
    previous_index = log.previous_comparable(probe)
    drifted = add(
        "drifted",
        F.make_cert(
            "fingerprint",
            subject=subject,
            recipe=recipe,
            body=drifted_body,
            refs=[
                {"role": "battery", "id": battery_cert["id"]},
                {"role": "previous", "id": log.cert(previous_index)["id"]},
            ],
        ),
    )

    # 10. the two verifications and their result certs
    diff_outcome = V.diff(canonical, drifted, log)
    ref_outcome = V.ref(canonical, drift_runner, log)
    clean_outcome = V.ref(canonical, runner, log)
    add(
        "diff_result",
        V.make_result_cert(
            diff_outcome,
            F.issuer("issuer"),
            ISSUER_SEED,
            diff_outcome.result_body["refs_suggested"],
            {},
            {},
            created=F.CREATED,
        ),
    )
    add(
        "ref_result",
        V.make_result_cert(
            ref_outcome,
            F.issuer("issuer"),
            ISSUER_SEED,
            ref_outcome.result_body["refs_suggested"],
            {},
            {},
            created=F.CREATED,
        ),
    )

    # 11. tree heads: one over the ladder's spine, one over the whole log
    sth_end = log.sth(LOG_SEED, TS_END)

    # 12. the two contrast plans on the pool battery (section 5.6 and the probe)
    pool_orders = plan_orders([item["item_id"] for item in pool_items()])
    pool_runs_batch1 = [
        FP.run_fingerprint(
            runner, subject, pool_recipe, pool, run_index=k, nuisance={}, order=order
        )
        for k, order in enumerate(pool_orders)
    ]
    batch8_recipe = F.recipe(battery=pool["id"], decoding=F.decoding(batch_size=8))
    pool_runs_batch8 = [
        FP.run_fingerprint(
            runner, subject, batch8_recipe, pool, run_index=k, nuisance={}, order=order
        )
        for k, order in enumerate(pool_orders)
    ]

    return {
        "kind": kind,
        "logprobs": logprobs,
        "subject": subject,
        "recipe": recipe,
        "pool_recipe": pool_recipe,
        "log": log,
        "certs": certs,
        "battery_ids": battery_ids,
        "battery_body": battery_body,
        "sweep_record": sweep_record,
        "scores": scores,
        "run_bodies": run_bodies,
        "run_certs": run_certs,
        "floor_block": floor_block,
        "roles": roles,
        "control_body": control_body,
        "sensitivity_per_channel": sensitivity_per_channel,
        "drift_item": drift_item,
        "drifted_body": drifted_body,
        "previous_index": previous_index,
        "diff_outcome": diff_outcome,
        "ref_outcome": ref_outcome,
        "clean_outcome": clean_outcome,
        "sth_end": sth_end,
        "runner": runner,
        "drift_runner": drift_runner,
        "pool_runs_batch1": pool_runs_batch1,
        "pool_runs_batch8": pool_runs_batch8,
    }


@pytest.fixture(scope="module")
def weights_ladder(tmp_path_factory):
    return build_ladder(tmp_path_factory.mktemp("weights") / "log", "weights")


@pytest.fixture(scope="module")
def alias_ladder(tmp_path_factory):
    return build_ladder(tmp_path_factory.mktemp("alias") / "log", "alias")


@pytest.fixture(params=["weights", "alias"])
def ladder(request, weights_ladder, alias_ladder):
    return weights_ladder if request.param == "weights" else alias_ladder


# --------------------------------------------------------------------------- the log


def test_the_log_holds_the_whole_ladder_in_dependency_order(ladder):
    log = ladder["log"]
    assert log.size() == len(LADDER_ORDER)
    for index, name in enumerate(LADDER_ORDER):
        assert log.cert(index)["id"] == ladder["certs"][name]["id"], name
    # dependency order: every ref of every entry resolves to a strictly earlier entry. There is
    # no exception -- the ladder's root is the pool battery, which carries no ref at all.
    for index in range(log.size()):
        cert = log.cert(index)
        for role, rid in certmod.refs(cert):
            at = log.find(rid)
            assert at is not None, f"entry {index}: {role} ref {rid} resolves to nothing"
            assert at < index, f"entry {index}: {role} ref resolves forward to {at}"


def test_every_entry_checks_out_and_verifies_from_its_own_bytes(ladder):
    log = ladder["log"]
    for index in range(log.size()):
        outcome = certmod.check(log.cert(index))
        assert outcome.ok, (index, outcome.reasons)
        ok, reason = verify_entry(log, index)
        assert ok, (index, reason)


def test_the_ladder_never_reaches_the_genesis_exemption(ladder):
    """The root of the ladder carries no ref, so the exemption is not what starts the log.

    ``log.GENESIS_UNRESOLVED`` permits one unresolvable ref, ``(battery, pool-v1, battery)``.
    This ladder's pool battery has an empty recipe and an empty ref list: nothing about it is a
    ref that fails to resolve, so the exemption is never consulted.  A battery that DOES name an
    absent battery is still refused, whatever its kind.
    """
    log = ladder["log"]
    assert GENESIS_UNRESOLVED == frozenset({("battery", "pool-v1", "battery")})

    pool = ladder["certs"]["pool"]
    assert pool["recipe"] == {}
    assert pool["refs"] == []
    assert certmod.embedded_ids(pool) == set()
    assert log.find(pool["id"]) == 0

    # A fixed-v1 battery naming an absent battery is not in the exempted set and is refused.
    # (pool-v1 with that same ref still IS in the set -- that is the whole remaining reach of
    # GENESIS_UNRESOLVED, and neither ladder builds one, so nothing here depends on it.)
    absent = F.fake_id("not in this log")
    stray = F.make_cert(
        "battery",
        subject=ladder["subject"],
        recipe=F.recipe(battery=absent),
        body=B.fixed_v1(pool_items()[:3], source="styxx-e2e@v8"),
        refs=[{"role": "battery", "id": absent}],
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(stray)
    assert exc.value.reason == f"refs: battery ref {absent} does not resolve in this log"
    assert log.size() == len(LADDER_ORDER)


def test_only_the_canary_battery_of_the_ladder_names_a_pool_in_its_recipe(ladder):
    """Section 4.5: two of the three battery kinds are roots and the third is not.

    ``pool-v1`` and ``fixed-v1`` carry no ``recipe.battery``; ``canary-v1`` does, because it was
    selected against a fingerprint that ran on that pool, and the schema requires it there.
    """
    battery = ladder["certs"]["battery"]
    pool_id = ladder["certs"]["pool"]["id"]
    roles = {role: rid for role, rid in certmod.refs(battery)}
    assert certmod.check(battery).ok
    if ladder["kind"] == "weights":
        assert battery["body"]["kind"] == "canary-v1"
        assert battery["recipe"]["battery"] == pool_id
        assert roles == {
            "battery": pool_id,
            "pool": pool_id,
            "selected_against": ladder["certs"]["reference"]["id"],
        }
        assert certmod.embedded_ids(battery) == {pool_id}
    else:
        assert battery["body"]["kind"] == "fixed-v1"
        assert battery["recipe"] == {}
        assert roles == {"pool": pool_id}
        # The pool ref is provenance the schema does not force: nothing in a root battery is an
        # embedded id, so section 2.1 asks for no ref at all here.
        assert certmod.embedded_ids(battery) == set()


def test_the_baseline_rule_holds_the_drifted_fingerprint_to_a_previous_ref(ladder):
    """Section 5.5: a comparable fingerprint is on record, so a new baseline names its parent."""
    log = ladder["log"]
    assert ladder["previous_index"] is not None
    naked = F.make_cert(
        "fingerprint",
        subject=ladder["subject"],
        recipe=ladder["recipe"],
        body=ladder["drifted_body"],
        refs=[{"role": "battery", "id": ladder["certs"]["battery"]["id"]}],
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(naked)
    assert exc.value.reason.startswith("baseline:")
    assert log.size() == len(LADDER_ORDER)


def test_the_five_runs_share_a_noise_plan_instead_of_a_previous_ref(ladder):
    """Runs of one floor are comparable by construction and do not each start a baseline."""
    for cert in ladder["run_certs"][1:]:
        roles = {role for role, _ in certmod.refs(cert)}
        assert "previous" not in roles
        assert "noise_plan" in roles
    canonical_roles = {role for role, _ in certmod.refs(ladder["certs"]["canonical"])}
    assert "previous" not in canonical_roles
    assert "noise_plan" in canonical_roles


# --------------------------------------------------------------------------- the floor


def test_the_batch_one_plan_has_floor_zero_on_every_present_channel(ladder):
    """Five runs varying only the item order: the mock does not move, so max(D_c) = 0.0."""
    block = ladder["floor_block"]
    expected = {"exact", "seqlp", "topk"} if ladder["logprobs"] else {"exact"}
    assert set(block["per_channel"]) == expected
    for channel, per in block["per_channel"].items():
        assert per["floor"] == 0.0, channel
        assert per["runs"] == 5
        assert per["pairs"] == 10  # 5 runs -> 5*4/2
        assert per["alpha_single"] == 1 / 11
    assert block["plan"] == ladder["certs"]["noise_plan"]["id"]
    assert block["runs"] == [c["id"] for c in ladder["run_certs"]]
    # Section 5.4's two lists are the plan's own derivation (A-COVER,
    # `Log._check_floor_covers_match_the_plan`), and this ladder is the case that shows why the
    # derivation had to replace a hand-written pair. This plan enumerates no nuisance factor and
    # CANNOT: its five runs vary the item order at batch size 1, where nothing in the execution
    # can tell one order from another (`Log._check_floor_varied_an_execution` refuses exactly
    # that), which is also why every channel's floor above is 0.0. The hand-written
    # `covers: ["order"]` this used to carry was therefore a claim to cover a factor the floor
    # measured nothing about, sitting beside a floor of zero -- in the lab's own reference
    # ladder. What it covers is nothing, and it now says so.
    assert block["covers"] == []
    assert block["not_covered"] == [
        "hardware.count", "hardware.driver", "hardware.gpu",
        "runtime.backend", "runtime.framework", "runtime.version",
    ]


def test_batch_one_reruns_leave_the_nuisance_items_untouched(ladder):
    """The nuisance items are in the pool; at batch 1 no reordering moves any of them."""
    runs = ladder["pool_runs_batch1"]
    baseline = item_map(runs[0])
    assert NUISANCE_ITEMS <= set(baseline)
    for other in runs[1:]:
        seen = item_map(other)
        assert set(seen) == set(baseline)
        for item_id in sorted(baseline):
            assert canonical_bytes(seen[item_id]) == canonical_bytes(baseline[item_id]), item_id
    per = floormod.floors(runs)
    assert per["exact"]["floor"] == 0.0


def test_the_same_plan_at_batch_eight_measures_a_different_quantity(ladder):
    """Section 5.6: a floor taken across batch sizes is not a floor a batch-1 run can use."""
    per = floormod.floors(ladder["pool_runs_batch8"])
    assert per["exact"]["floor"] > 0.0
    moved = set()
    baseline = item_map(ladder["pool_runs_batch8"][0])
    for other in ladder["pool_runs_batch8"][1:]:
        seen = item_map(other)
        moved |= {i for i in baseline if seen[i]["token_ids"] != baseline[i]["token_ids"]}
    assert moved <= NUISANCE_ITEMS and moved


# --------------------------------------------------------------------------- sensitivity


def test_the_sensitivity_control_exceeds_the_floor_on_every_channel(ladder):
    """Section 5.3: a `same` verdict rests on a receipt that the instrument can detect a change."""
    per = ladder["sensitivity_per_channel"]
    assert set(per) == set(ladder["floor_block"]["per_channel"])
    for channel, block in per.items():
        assert block["floor"] == 0.0, channel
        assert block["distance"] > block["floor"], channel
        assert block["exceeded"] is True, channel
    # two of eight items moved under the control -> 2/8 on the exact channel
    assert per["exact"]["distance"] == 0.25


def test_the_canonical_fingerprint_names_its_floor_and_its_receipt(ladder):
    cert = ladder["certs"]["canonical"]
    assert certmod.check(cert).ok
    assert cert["body"]["sensitivity"] == ladder["certs"]["sensitivity"]["id"]
    carried = {rid for _, rid in certmod.refs(cert)}
    assert certmod.embedded_ids(cert) <= carried
    assert certmod.embedded_ids(cert) == {
        ladder["certs"]["battery"]["id"],
        ladder["certs"]["noise_plan"]["id"],
        ladder["certs"]["sensitivity"]["id"],
        *[c["id"] for c in ladder["run_certs"]],
    }


# --------------------------------------------------------------------------- the verdicts


def test_diff_against_the_drifted_cert_is_exceeds_floor_and_exits_two(ladder):
    """A --diff has no confirmation run, so an exceedance stays `exceeds_floor` (GATED S5-02)."""
    outcome = ladder["diff_outcome"]
    assert outcome.verdict == "exceeds_floor"
    assert outcome.exit_code == EXIT["inconclusive"] == 2
    assert outcome.mismatched == []
    body = outcome.result_body
    assert body["floor_owner"] == "A"
    assert body["coverage"] == "within"
    # one drifted item out of eight, scored on the exact channel
    assert body["per_channel"]["exact"]["distance"] == 0.125
    assert body["per_channel"]["exact"]["floor"] == 0.0
    assert body["per_channel"]["exact"]["ratio"] is None  # a zero floor has no ratio
    assert body["per_channel"]["exact"]["verdict"] == "exceeds_floor"
    assert body["anchor_flips"] == 0


def test_ref_with_the_drifted_runner_is_drift_and_records_its_confirmation(ladder):
    outcome = ladder["ref_outcome"]
    assert outcome.verdict == "drift"
    assert outcome.exit_code == EXIT["drift"] == 1
    body = outcome.result_body
    assert body["confirmation_run"] is not None
    assert body["per_channel"]["exact"]["distance"] == 0.125
    assert body["per_channel"]["exact"]["confirmation_distance"] == 0.125
    assert body["per_channel"]["exact"]["verdict"] == "drift"
    assert body["coverage"] == "within"
    assert body["skew_fields"] == []
    assert body["identity_diff"] == []


def test_ref_with_the_logged_runner_is_same_and_exits_zero(ladder):
    outcome = ladder["clean_outcome"]
    assert outcome.verdict == "same"
    assert outcome.exit_code == EXIT["same"] == 0
    assert outcome.result_body["confirmation_run"] is None
    assert outcome.result_body["sensitivity"] == ladder["certs"]["sensitivity"]["id"]
    for channel in ladder["floor_block"]["per_channel"]:
        assert outcome.result_body["per_channel"][channel]["distance"] == 0.0


def test_only_the_drifted_item_moved(ladder):
    baseline = item_map(ladder["run_bodies"][0])
    seen = item_map(ladder["drifted_body"])
    moved = {i for i in baseline if seen[i]["token_ids"] != baseline[i]["token_ids"]}
    assert moved == {ladder["drift_item"]}
    assert len(baseline) == N_CANARIES


def test_a_genuinely_second_subject_is_a_mismatch_and_carries_no_distances(ladder):
    """Why the drift leg re-runs one subject instead of building a second one (section 2.3).

    The contract's ladder says "a second subject with drift_items". Two subjects whose identity
    fields differ are not comparable at all: ``--diff`` between them exits 3 and prints the
    mismatched field, so a second subject could never produce the drift verdict the same line
    asks for. This pins that, and the drift leg uses one subject on a moved runner instead.
    """
    # One field out of S_identity (section 2.2), the kind's own: a different set of weights, or
    # a different alias behind the same endpoint. `model_family` is NOT in S_identity.
    field = "weights_sha256" if ladder["kind"] == "weights" else "alias"
    other = copy.deepcopy(ladder["subject"])
    other[field] = "e" * 64 if field == "weights_sha256" else "acme-small"
    second = F.make_cert(
        "fingerprint",
        subject=other,
        recipe=ladder["recipe"],
        body=ladder["drifted_body"],
        refs=[{"role": "battery", "id": ladder["certs"]["battery"]["id"]}],
    )
    outcome = V.diff(ladder["certs"]["canonical"], second, ladder["log"])
    assert outcome.verdict == "mismatch"
    assert outcome.exit_code == EXIT["mismatch"] == 3
    assert outcome.mismatched == [f"subject.{field}"]
    assert outcome.result_body["per_channel"] == {}  # section 6 exit 3: no distances at all
    assert "distance" not in outcome.printed
    assert f"subject.{field}" in outcome.printed


def test_both_verify_result_certs_are_logged_and_check_out(ladder):
    log = ladder["log"]
    for name, mode in (("diff_result", "diff"), ("ref_result", "ref")):
        cert = ladder["certs"][name]
        assert certmod.check(cert).ok
        assert cert["type"] == "result"
        assert cert["body"]["kind"] == "verify"
        assert cert["body"]["mode"] == mode
        assert cert["body"]["ref"] == ladder["certs"]["canonical"]["id"]
        assert log.find(cert["id"]) is not None
    roles = {role for role, _ in certmod.refs(ladder["certs"]["diff_result"])}
    assert {"target", "own", "sensitivity"} <= roles


# --------------------------------------------------------------------------- heads and mirrors


def test_the_four_verify_commands_pass_on_the_finished_log(ladder):
    log = ladder["log"]
    head = ladder["sth_end"]
    assert head["tree_size"] == log.size()

    ok, reason = verify_sth(head, LOG_PUB)
    assert ok, reason

    for index in range(log.size()):
        ok, reason = verify_entry(log, index)
        assert ok, (index, reason)
        ok, reason = verify_inclusion(log.inclusion(index, head["tree_size"]), head, LOG_PUB)
        assert ok, (index, reason)

    assert log.latest_sth()["root_hash"] == head["root_hash"]


def test_a_consistency_proof_carries_an_earlier_head_into_the_finished_one(ladder, tmp_path):
    """Two heads over the same entries: the earlier one is signed on a truncated copy."""
    log = ladder["log"]
    head = ladder["sth_end"]
    small = tmp_path / "earlier"
    shutil.copytree(log.path, small)
    keep = 5
    for index in range(keep, log.size()):
        Log(small).entry_path(index).unlink()
        Log(small).meta_path(index).unlink()
    truncated = Log(small)
    for path in truncated.sth_dir.glob("*.json"):
        path.unlink()
    assert truncated.size() == keep
    earlier = truncated.sth(LOG_SEED, TS_MID)

    ok, reason = verify_sth(earlier, LOG_PUB)
    assert ok, reason
    proof = log.consistency(keep, head["tree_size"])
    ok, reason = verify_consistency(earlier, head, proof, LOG_PUB)
    assert ok, reason


def test_a_mirror_of_the_finished_log_verifies(ladder, tmp_path):
    report = mirror(ladder["log"].path, tmp_path / "mirror", LOG_PUB, pinned_sth=ladder["sth_end"])
    assert report["tamper"] == []
    assert report["misbehaviour"] == []
    assert report["unpublished"] == []
    assert report["entries"] == len(LADDER_ORDER)
    assert report["sths"] == 1
    assert report["verified"] is True


def test_a_mirror_of_an_edited_log_is_reported(ladder, tmp_path):
    source = tmp_path / "edited"
    shutil.copytree(ladder["log"].path, source)
    entry = Log(source).entry_path(0)
    raw = entry.read_bytes()
    edited = raw.replace(b"pool prompt 0", b"pool prompt X", 1)
    assert edited != raw and len(edited) == len(raw)
    entry.write_bytes(edited)

    report = mirror(source, tmp_path / "mirror-edited", LOG_PUB, pinned_sth=ladder["sth_end"])
    assert report["verified"] is False
    assert report["tamper"]
    assert any("entry 0" in line for line in report["tamper"])


# --------------------------------------------------------------------------- weights only


def test_the_selection_excludes_the_nuisance_items_and_keeps_precision_sensitivity(weights_ladder):
    """Section 4.4 step 1 strips every batch-sensitive item; what remains is measured, not assumed."""
    body = weights_ladder["battery_body"]
    assert B.validate_body(body) == []
    assert body["kind"] == "canary-v1"
    assert {entry["item_id"] for entry in body["excluded"]} == set(NUISANCE_ITEMS)
    for entry in body["excluded"]:
        assert entry["flip2"] == 1.0
    assert len(body["items"]) == N_CANARIES
    assert not set(weights_ladder["battery_ids"]) & NUISANCE_ITEMS
    kept = set(weights_ladder["battery_ids"]) & PRECISION_ITEMS
    assert kept, "the ladder needs a battery that still moves under a precision change"
    assert body["params"]["sensitivity_after_exclusion"] == round(len(kept) / N_CANARIES, 9)
    # The mock's log-prob gaps are never wide enough for exp(stay) == 1, so zero(i) is
    # unsatisfiable on mock data and no anchor is selected.
    assert body["params"]["k_anchors"] == K_ANCHORS
    assert body["params"]["k_anchors_actual"] == 0
    assert {item["role"] for item in body["items"]} == {"canary"}


def test_the_sweep_record_is_keyed_by_item_and_round_trips(weights_ladder, tmp_path):
    record = weights_ladder["sweep_record"]
    assert set(record["reference"]) == {item["item_id"] for item in pool_items()}
    assert set(record["delta1"]) == {DELTA1_PRECISION}
    assert len(record["delta2"]) == 2
    assert record["params"]["pool_size"] == POOL_SIZE
    path = SW.write_record(record, tmp_path / "sweep.json")
    assert SW.read_record(path) == record


def test_the_precision_variant_moved_exactly_the_precision_items(weights_ladder):
    scores = weights_ladder["scores"]
    moved = {item_id for item_id, block in scores.items() if block["flip1"] > 0.0}
    assert moved == set(PRECISION_ITEMS)
    batched = {item_id for item_id, block in scores.items() if block["flip2"] > 0.0}
    assert batched == set(NUISANCE_ITEMS)


# --------------------------------------------------------------------------- alias only


def test_an_alias_without_logprobs_cannot_carry_a_canary_selection(alias_ladder):
    """Appendix C has no fallback without margin_by_position and stay; the refusal is the finding."""
    with pytest.raises(ValueError, match="margin_by_position"):
        B.score(alias_ladder["sweep_record"])
    assert alias_ladder["battery_body"]["kind"] == "fixed-v1"
    assert B.validate_body(alias_ladder["battery_body"]) == []
    assert {item["role"] for item in alias_ladder["battery_body"]["items"]} == {"item"}


def test_the_alias_channels_are_absent_rather_than_zero(alias_ladder):
    body = alias_ladder["certs"]["canonical"]["body"]
    assert body["tier"] == "black-box"
    assert body["channels"]["seqlp"] == {"present": False}
    assert body["channels"]["topk"] == {"present": False}
    assert set(body["noise_floor"]["per_channel"]) == {"exact"}
    for outcome in (alias_ladder["diff_outcome"], alias_ladder["ref_outcome"]):
        assert set(outcome.result_body["per_channel"]) == {"exact"}
        # absent on BOTH sides is neither compared nor skipped (section 5.2)
        assert outcome.result_body["skipped_channels"] == []
    assert alias_ladder["diff_outcome"].result_body["alias_note"] == V.ALIAS_NOTE
    assert alias_ladder["ref_outcome"].result_body["alias_note"] == V.ALIAS_NOTE
    assert V.ALIAS_NOTE in alias_ladder["ref_outcome"].printed


@pytest.mark.parametrize(
    "drift, verdict, code",
    [(False, "same", EXIT["same"]), (True, "drift", EXIT["drift"])],
)
def test_a_logprob_runner_against_the_alias_cert_lists_the_two_channels_as_skipped(
    alias_ladder, drift, verdict, code
):
    """One side carrying log-probs is the asymmetric case: listed, and never an exit code."""
    runner = make_runner(
        drift={alias_ladder["drift_item"]} if drift else (), logprobs=True
    )
    outcome = V.ref(alias_ladder["certs"]["canonical"], runner, alias_ladder["log"])
    assert outcome.result_body["skipped_channels"] == ["seqlp", "topk"]
    assert set(outcome.result_body["per_channel"]) == {"exact"}
    assert outcome.verdict == verdict
    assert outcome.exit_code == code


def test_the_alias_exit_codes_match_the_weights_ladder(weights_ladder, alias_ladder):
    """Section 5.2: an absent channel never changes the code."""
    for key in ("diff_outcome", "ref_outcome", "clean_outcome"):
        assert alias_ladder[key].exit_code == weights_ladder[key].exit_code, key
        assert alias_ladder[key].verdict == weights_ladder[key].verdict, key


# ----------------------------------------------------------- section 5.1 step 5, through the CLI


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return path


def test_the_floor_append_order_of_section_5_1_step_5_works_end_to_end(tmp_path):
    """A floor built and logged by nothing but the command line, on the mock runner.

    This is the leg the run in `papers/v8/first_log_2026_09_09/` could not reach: the
    verbs were battery, fingerprint, key, log, verify, so the `prereg` cert section 5.1 step 1
    requires before the runs could not be minted, section 5.5 refused the second comparable
    fingerprint, and `--runs 5` emitted five independent fingerprints with a canonical cert
    byte-identical to run 0 and no floor.

    The order appended here is exactly section 5.1 step 5: (i) the noise-plan prereg, (ii) the
    R-1 non-canonical run certs, each under the plan, (iii) the canonical fingerprint carrying
    the floor.  Every append is asserted to be accepted at a consecutive index, and the floor on
    the canonical cert is asserted to be non-empty.
    """
    root = tmp_path / "ladder"
    key_pem = root / "issuer.pem"
    root.mkdir(parents=True, exist_ok=True)
    keys.save_private_pem(ISSUER_SEED, key_pem)

    def run(*args) -> dict:
        code, payload = cli.run([str(a) for a in args])
        assert code == 0, (args[:2], payload)
        return payload

    subject = F.weights_subject(environment=copy.deepcopy(MOCK_ENVIRONMENT))
    items = _write_json(root / "pool_items.json", pool_items())

    # 0. the root battery: empty recipe, no refs, index 0 of an empty log (section 4.5)
    root_spec = _write_json(root / "spec_root.json", {"subject": subject, "recipe": {}})
    pool_cert = run(
        "battery", "pool", "--source", items, "--key", key_pem, "--subject", root_spec,
        "--created", F.CREATED, "--out", root / "certs" / "pool.json",
    )
    pool_id = pool_cert["id"]
    spec = _write_json(
        root / "spec.json", {"subject": subject, "recipe": F.recipe(battery=pool_id)}
    )

    # i. the plan, minted before a single run happens
    plan = run(
        "prereg", "noise-plan", "--runs", 5,
        "--nuisance", "order=a3|permuted", "--nuisance", "batch_size=1|4",
        "--subject", spec, "--battery", root / "certs" / "pool.json", "--key", key_pem,
        "--created", F.CREATED, "--out", root / "certs" / "plan.json",
    )
    assert plan["runs"] == 5 and plan["covers"] == ["batch_size", "order"]

    # ii + iii. the R-1 run certs and the canonical, in the order the CLI reports them
    fp = run(
        "fingerprint", "--subject", spec, "--battery", root / "certs" / "pool.json",
        "--key", key_pem, "--runs", 5, "--plan", root / "certs" / "plan.json",
        "--created", F.CREATED, "--out", root / "fp",
    )
    assert fp["append_order"] == [*fp["run_ids"], fp["id"]]
    assert len(fp["run_ids"]) == 4 and fp["id"] not in fp["run_ids"]

    # the log takes the whole sequence, in that order, at consecutive indexes
    log_dir = root / "log"
    run("log", "init", "--log", log_dir, "--key", key_pem,
        "--issuer", "lab=" + F.public_key("issuer"))
    order = [root / "certs" / "pool.json", root / "certs" / "plan.json"]
    order += [Path(p) for p in fp["written"]]
    indexes = [run("log", "append", path, "--log", log_dir)["index"] for path in order]
    assert indexes == list(range(len(order)))

    log = Log(log_dir)
    assert [log.cert(i)["type"] for i in indexes] == [
        "battery", "prereg", "fingerprint", "fingerprint", "fingerprint", "fingerprint",
        "fingerprint",
    ]
    assert log.cert(1)["body"]["kind"] == "noise-plan"
    assert [log.cert(i)["body"]["run_index"] for i in indexes[2:6]] == [1, 2, 3, 4]

    canonical = log.cert(indexes[-1])
    assert canonical["id"] == fp["id"]
    block = canonical["body"]["noise_floor"]
    assert canonical["body"]["run_index"] == 0
    assert block["per_channel"], "the canonical fingerprint carries no floor"
    assert set(block["per_channel"]) == {"exact", "seqlp", "topk"}
    assert block["plan"] == plan["id"]
    assert block["runs"] == fp["run_ids"]
    assert block["covers"] == ["batch_size", "order"]
    assert block["not_covered"] and "hardware.gpu" in block["not_covered"]
    # section 5.7: the overall size is measured over the R runs, and a floor of 0 is governed by
    # the exceedance rule rather than by a ratio nothing can compute
    assert block["alpha_overall"] == 0.0
    assert len(block["standardized_max"]) == 5

    # every id the floor names resolves in this log, which is what makes the block re-derivable
    for cert_id in [block["plan"], *block["runs"]]:
        assert log.find(cert_id) is not None
    assert verify_entry(log, indexes[-1]) == (True, "ok")

    # and the canonical is a NEW baseline the rule accepts, because it shares the runs' plan
    assert log.previous_comparable(canonical) == indexes[-2]
