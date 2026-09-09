"""styxx.v8 section 5.5 -- the baseline gap: what a replaced baseline announces.

Section 5.5 lets a second canonical fingerprint for one subject append when it carries a
``previous`` ref, or when it is another run under the same logged noise plan. Choosing a new
baseline is legitimate. Choosing one *silently* is the BASELINE-CHOICE attack, and the quantity
that ends the silence -- the distance between the baseline already on record and the one
replacing it -- was computable from two certs the log already held and was computed nowhere.

What is pinned here:

* ``floor.baseline_gap`` as arithmetic over two run bodies: the distance, the previous
  baseline's floor as the yardstick, the ratio, the exceedance, the skipped channels;
* ``Log.baseline_gap`` and the entry metadata ``Log.append`` writes, in both the announced case
  (a ``previous`` ref) and the unannounced one (another run under the same noise plan);
* ``verify`` printing it, from a log through the resolver and from the cert's own ``previous``
  ref, and printing nothing when it has neither;
* **that it changes no verdict.** Every disclosure test below also asserts the exit code and the
  overall verdict are what they were without it. This is disclosure, not prevention: the gap
  says the baseline moved and by how much, and cannot say which of the two baselines measured
  the subject, because a relabelled computation is byte-indistinguishable from a computation
  (``papers/v8/THE_BOUNDARY_2026_09_09.md``, class two). BASELINE-CHOICE stays in the label
  class where the relabel itself is concerned; only the announcement leaves it.

**The announcement itself was then found announcing things that did not happen**, and the last
section of this file is the repair. ``THE_BOUNDARY`` credited member 2 with this field as a
mitigation and withdrew the credit the same day, on three defects:

* **G1** — the previous-comparable lookup never excluded the appending cert's OWN
  ``noise_floor.runs``. Verified directly on the lab's published verdict: the lookup for the
  canonical returns entry 5, which is that canonical's own run 4, so a fresh append measured a
  canonical against a run of its own floor and called the result a baseline gap. Nothing had been
  replaced. EXTERNAL-1 is what an accuser that fires on honest artifacts costs: 0.23 precision,
  class disabled.
* **G2** — the honest case (two runs of one preregistered plan) and the dishonest one (a second
  canonical quietly replacing the first under that plan) produced the same three values:
  ``announced: false``, ``same_noise_plan: true``, a distance. An announcement that reads the same
  either way discriminates nothing. It now carries ``relation`` / ``replaced_a_baseline``, which
  are decided by a fact the log holds — whether either cert carries a ``noise_floor`` block, i.e.
  whether either is a baseline at all.
* **G3** — the announcement is pairwise, so a baseline staged from A to B to C publishes nothing
  cumulative. The decision is that a cumulative *distance* does not belong in this field, and
  ``test_the_gap_publishes_no_cumulative_distance_and_says_why`` pins the reason rather than a
  defence: none of the three channels composes along a chain, so a sum of pairwise gaps is not the
  A-to-C distance and would be a second number that announces something that did not happen.

This file is not one of ``conformance/v8/__init__.py``'s SOURCES, so nothing here is recorded
into the conformance set. Adding a source changes what the set is a function of and is a
separate decision; ``floor.baseline_gap`` is therefore unvectored, and the README says so.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import floor as FLOOR
from styxx.v8 import verify as V
from styxx.v8.log import Log
from tests import v8_fixtures as F
# The five-run ladder the floor tests build: a battery, a plan fixing R = 5, four run certs and
# the canonical over them. G1 and G2 are statements about a cert carrying a REAL floor block, and
# a real floor block needs a plan and four resolvable runs in the same log.
from tests.test_v8_floor_subject import ladder as floor_ladder

PUBLISHED = Path(__file__).resolve().parents[1] / "papers/v8/first_verdict_2026_09_09/log"

_LOG_SEED, LOG_PUB = F.keypair("log-key")


# ----------------------------------------------------------------- helpers

def roster(label: str = "issuer") -> list[dict]:
    return [{
        "name": F.ISSUER_NAME,
        "key": F.public_key(label),
        "from_index": 0,
        "retired_at_index": None,
    }]


def fresh(root) -> Log:
    return Log.init(root, LOG_PUB, roster())


def pool_cert() -> dict:
    return F.make_cert("battery", body=F.battery_body("pool-v1"))


def fp_cert(battery_id: str, *, extra_refs=(), body=None, **body_kw) -> dict:
    body = F.fingerprint_body(**body_kw) if body is None else body
    refs = [{"role": "battery", "id": battery_id}] + [dict(r) for r in extra_refs]
    return F.make_cert(
        "fingerprint", recipe=F.recipe(battery=battery_id), refs=refs, body=body
    )


def moved(body: dict, *, item: int = 0, seqlp_delta: float = 0.5) -> dict:
    """``body`` with one item's tokens and one item's logprob moved: a body that is genuinely a
    different measurement, so ``exact`` and ``seqlp`` both carry a non-zero distance."""
    out = copy.deepcopy(body)
    it = out["items"][item]
    it["token_ids"] = [9000 + n for n in range(len(it["token_ids"]))]
    it["token_ids_sha256"] = F.sha256_text(str(it["token_ids"]))
    it["output_sha256"] = F.sha256_text("moved")
    it["output_text"] = "moved"
    it["seq_logprob"] = round(it["seq_logprob"] - seqlp_delta, 9)
    return out


def with_floor(body: dict, *, exact: float, seqlp: float | None = None) -> dict:
    """``body`` carrying a hand-set ``noise_floor`` -- the yardstick, not a measurement."""
    out = copy.deepcopy(body)
    per: dict = {"exact": {"floor": exact, "runs": 5, "pairs": 10, "alpha_single": 1 / 11}}
    if seqlp is not None:
        per["seqlp"] = {"floor": seqlp, "runs": 5, "pairs": 10, "alpha_single": 1 / 11}
    out["noise_floor"] = {
        "plan": F.NOISE_PLAN_ID,
        "runs": list(F.RUN_IDS),
        "covers": ["runtime.version"],
        "not_covered": ["hardware.gpu"],
        "per_channel": per,
    }
    return out


# ----------------------------------------------------------------- floor.baseline_gap


def test_two_identical_baselines_are_zero_apart_on_every_shared_channel():
    body = F.fingerprint_body()
    gap = FLOOR.baseline_gap(body, copy.deepcopy(body))
    assert gap["floor_owner"] == "previous"
    assert set(gap["per_channel"]) == {"exact", "seqlp", "topk"}
    for block in gap["per_channel"].values():
        assert block["distance"] == 0.0
    assert gap["channels_exceeding_floor"] == []
    assert gap["skipped_channels"] == []


def test_a_moved_baseline_carries_the_distance_the_floor_and_the_ratio():
    previous = with_floor(F.fingerprint_body(), exact=0.1, seqlp=0.25)
    new = moved(previous)
    gap = FLOOR.baseline_gap(previous, new)

    exact = gap["per_channel"]["exact"]
    assert exact["distance"] == pytest.approx(1 / 3)   # one of three items moved
    assert exact["floor"] == 0.1                        # the PREVIOUS baseline's floor
    assert exact["ratio"] == pytest.approx((1 / 3) / 0.1)
    assert exact["exceeds_floor"] is True

    seqlp = gap["per_channel"]["seqlp"]
    assert seqlp["distance"] == pytest.approx(0.5 / 3)
    assert seqlp["exceeds_floor"] is False              # 0.1667 under a floor of 0.25

    assert gap["channels_exceeding_floor"] == ["exact"]
    assert gap["max_ratio"] == pytest.approx((1 / 3) / 0.1)


def test_the_yardstick_is_the_previous_baselines_floor_not_the_new_ones():
    """The act that moves the baseline must not also move the ruler."""
    previous = with_floor(F.fingerprint_body(), exact=0.1)
    new = with_floor(moved(F.fingerprint_body()), exact=0.9)
    gap = FLOOR.baseline_gap(previous, new)
    assert gap["per_channel"]["exact"]["floor"] == 0.1
    assert gap["per_channel"]["exact"]["exceeds_floor"] is True
    assert gap["floor_new"]["exact"] == 0.9


def test_a_channel_present_on_one_side_only_is_skipped_never_zero_filled():
    previous = F.fingerprint_body()
    new = F.fingerprint_body(logprobs=False)  # seqlp and topk absent on the new baseline
    gap = FLOOR.baseline_gap(previous, new)
    assert sorted(gap["skipped_channels"]) == ["seqlp", "topk"]
    assert set(gap["per_channel"]) == {"exact"}


def test_a_channel_whose_distance_will_not_compute_carries_a_note_and_no_number():
    previous = F.fingerprint_body()
    new = copy.deepcopy(previous)
    new["items"][0]["seq_logprob"] = None   # present per `channels`, absent per the item
    gap = FLOOR.baseline_gap(previous, new)
    block = gap["per_channel"]["seqlp"]
    assert block["distance"] is None
    assert block["exceeds_floor"] is None
    assert "seq_logprob absent" in block["note"]


def test_baseline_gap_refuses_something_that_is_not_a_run_body():
    with pytest.raises(TypeError):
        FLOOR.baseline_gap([], {})
    with pytest.raises(TypeError):
        FLOOR.baseline_gap({}, "body")


# ----------------------------------------------------------------- Log.baseline_gap


def test_the_first_fingerprint_of_a_subject_has_no_baseline_to_be_apart_from(tmp_path):
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    first = fp_cert(pool["id"])
    index = log.append(first)
    assert log.baseline_gap(first) is None
    assert "baseline_gap" not in log.meta(index)


def test_a_second_baseline_with_a_previous_ref_is_announced_in_the_entry_metadata(tmp_path):
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    first = fp_cert(pool["id"], body=F.fingerprint_body())
    at_first = log.append(first)

    second = fp_cert(
        pool["id"],
        body=moved(F.fingerprint_body()),
        extra_refs=[{"role": "previous", "id": first["id"]}],
    )
    at_second = log.append(second)

    gap = log.meta(at_second)["baseline_gap"]
    assert gap["previous_index"] == at_first
    assert gap["previous_id"] == first["id"]
    assert gap["declared_previous"] == [first["id"]]
    assert gap["announced"] is True
    assert gap["same_noise_plan"] is False
    assert gap["per_channel"]["exact"]["distance"] == pytest.approx(1 / 3)
    # These two carry no `noise_floor` (a full one needs a plan and five run certs in the log),
    # so there is no yardstick and the gap says so rather than inventing one.
    assert gap["per_channel"]["exact"]["floor"] is None
    assert gap["per_channel"]["exact"]["exceeds_floor"] is None
    assert gap["channels_exceeding_floor"] == []


def test_the_unannounced_case_is_the_one_the_gap_exists_for(tmp_path):
    """Two runs under one noise plan append with no ``previous`` ref at all (section 5.5).

    That is legitimate and stays legitimate; what changes is that the entry now carries how far
    the second sits from the first, so a reader who verifies against the second is not the last
    to know that a first existed.
    """
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    plan = F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5})
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]

    run0 = fp_cert(pool["id"], extra_refs=plan_ref, body=F.fingerprint_body())
    at0 = log.append(run0)
    run1 = fp_cert(
        pool["id"], extra_refs=plan_ref, body=moved(F.fingerprint_body())
    )
    at1 = log.append(run1)

    gap = log.meta(at1)["baseline_gap"]
    assert gap["previous_index"] == at0
    assert gap["declared_previous"] == []
    assert gap["announced"] is False
    assert gap["same_noise_plan"] is True
    assert gap["per_channel"]["exact"]["distance"] == pytest.approx(1 / 3)


def test_the_gap_in_the_metadata_is_what_a_reader_recomputes_from_the_bytes(tmp_path):
    """The metadata is unsigned and outside the tree; every number in it is re-derivable."""
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    first = fp_cert(pool["id"])
    log.append(first)
    second = fp_cert(
        pool["id"],
        body=moved(F.fingerprint_body()),
        extra_refs=[{"role": "previous", "id": first["id"]}],
    )
    at_second = log.append(second)

    reopened = Log(log.path)
    assert reopened.baseline_gap(reopened.cert(at_second)) == reopened.meta(at_second)["baseline_gap"]


def test_the_gap_refuses_no_append_the_section_55_rule_accepted(tmp_path):
    """Disclosure, not prevention: a baseline a mile from its predecessor still appends."""
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    first = fp_cert(pool["id"], body=F.fingerprint_body())
    log.append(first)
    far = fp_cert(
        pool["id"],
        body=moved(moved(F.fingerprint_body(), item=0), item=1),
        extra_refs=[{"role": "previous", "id": first["id"]}],
    )
    at = log.append(far)   # accepted
    gap = log.meta(at)["baseline_gap"]
    assert gap["per_channel"]["exact"]["distance"] == pytest.approx(2 / 3)
    assert log.cert(at)["id"] == far["id"]


# ----------------------------------------------------------------- verify prints it


def _resolver(*certs) -> dict:
    return {c["id"]: c for c in certs}


def test_verify_diff_prints_the_gap_from_the_certs_own_previous_ref():
    battery = pool_cert()
    first = fp_cert(battery["id"], body=F.fingerprint_body())
    a = fp_cert(
        battery["id"],
        body=moved(F.fingerprint_body()),
        extra_refs=[{"role": "previous", "id": first["id"]}],
    )
    b = fp_cert(battery["id"], body=moved(F.fingerprint_body()))

    out = V.diff(a, b, _resolver(battery, first, a, b))
    gap = out.result_body["baseline_gap"]
    assert gap["source"] == "previous_ref"
    assert gap["previous_id"] == first["id"]
    assert gap["announced"] is True
    assert gap["per_channel"]["exact"]["distance"] == pytest.approx(1 / 3)
    assert "baseline replaced" in out.printed
    assert "does not say which baseline measured the subject" in out.printed


def test_a_result_cert_over_a_gap_carries_the_previous_ref_section_2_1(tmp_path):
    """The disclosure embeds a cert id, so the result that signs it must carry it as a ref.

    Section 2.1 makes every embedded cert id a ref. `refs_suggested` therefore names the previous
    baseline under the `previous` role, and `cert.role_type("result", "previous")` is
    `fingerprint` -- ROLE_TYPES alone reads `previous` as "a cert of my own type", which is right
    on a fingerprint and wrong on a result whose previous baseline is a fingerprint. Without the
    override the log refused such a result at append.
    """
    assert certmod.role_type("result", "previous") == "fingerprint"
    assert certmod.role_type("fingerprint", "previous") is None

    log = fresh(tmp_path / "log")
    battery = pool_cert()
    log.append(battery)
    first = fp_cert(battery["id"], body=F.fingerprint_body())
    log.append(first)
    a = fp_cert(
        battery["id"],
        body=moved(F.fingerprint_body()),
        extra_refs=[{"role": "previous", "id": first["id"]}],
    )
    log.append(a)
    b = fp_cert(
        battery["id"],
        body=moved(F.fingerprint_body(), item=1),
        extra_refs=[{"role": "previous", "id": a["id"]}],
    )
    log.append(b)

    out = V.diff(a, b, log)
    # `a` is in the log with `b` appended after it: the baseline `a` replaced is `first`, not the
    # later fingerprint. A reader re-deriving the gap gets what the append wrote.
    assert out.result_body["baseline_gap"]["previous_id"] == first["id"]
    suggested = out.result_body["refs_suggested"]
    assert {"role": "previous", "id": first["id"]} in suggested

    seed, _public = F.keypair("issuer")
    result = V.make_result_cert(
        out, F.issuer(), seed, suggested, {}, {}, created=F.CREATED
    )
    assert certmod.check(result).ok is True
    assert log.append(result) == 4      # the ref resolves, and to a fingerprint


def test_verify_diff_prints_no_gap_when_it_has_neither_a_log_nor_a_previous_ref():
    """The absence of a gap is not evidence that no baseline was replaced, and this is the
    limit: with the cert alone there is nothing to be apart from."""
    battery = pool_cert()
    a = fp_cert(battery["id"], body=F.fingerprint_body())
    b = fp_cert(battery["id"], body=moved(F.fingerprint_body()))
    out = V.diff(a, b, _resolver(battery, a, b))
    assert "baseline_gap" not in out.result_body
    assert "baseline replaced" not in out.printed


def test_a_log_as_the_resolver_reaches_the_unannounced_case(tmp_path):
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    plan = F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5})
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    run0 = fp_cert(pool["id"], extra_refs=plan_ref, body=F.fingerprint_body())
    log.append(run0)
    run1 = fp_cert(
        pool["id"], extra_refs=plan_ref, body=moved(F.fingerprint_body())
    )
    log.append(run1)

    out = V.diff(run1, run0, log)
    gap = out.result_body["baseline_gap"]
    assert gap["source"] == "log"
    assert gap["announced"] is False
    assert gap["same_noise_plan"] is True
    assert gap["previous_id"] == run0["id"]
    assert "NOT named by this cert" in out.printed


def test_the_gap_moves_no_verdict_and_no_exit_code():
    """One comparison, run twice: once by a cert that replaced a baseline, once by the same
    cert that did not. The announcement is the only difference in the result."""
    battery = pool_cert()
    first = fp_cert(battery["id"], body=moved(F.fingerprint_body(), item=2))
    body = F.fingerprint_body()
    a = fp_cert(
        battery["id"], body=body, extra_refs=[{"role": "previous", "id": first["id"]}]
    )
    a_alone = fp_cert(battery["id"], body=body)
    b = fp_cert(battery["id"], body=moved(F.fingerprint_body()))

    seen = V.diff(a, b, _resolver(battery, first, a, b))
    blind = V.diff(a_alone, b, _resolver(battery, a_alone, b))
    assert "baseline_gap" in seen.result_body
    assert "baseline_gap" not in blind.result_body
    assert seen.verdict == blind.verdict
    assert seen.exit_code == blind.exit_code
    assert seen.result_body["per_channel"] == blind.result_body["per_channel"]


# ------------------------------------------- G1: an announcement of something that did not happen


def test_the_published_canonical_announces_no_baseline():
    """THE PIN. ``papers/v8/first_verdict_2026_09_09/log`` is this lab's own published verdict.

    Its canonical (entry 6) is the first canonical fingerprint of its subject and it replaced
    nothing; the four comparable fingerprints below it are the four runs its own ``noise_floor``
    block names. The lookup used to return entry 5 — run 4 — so an append computing this field
    today would have announced a baseline gap on the lab's own published bytes, with the honest
    answer being that no baseline was ever replaced.

    The stored entries carry no ``baseline_gap`` at all (the log predates the field), so this
    pins the DERIVATION, which is what a reader re-running it gets: None, and no announcement.
    """
    log = Log(PUBLISHED)
    canonical = log.cert(6)
    assert canonical["type"] == "fingerprint"
    floor_runs = canonical["body"]["noise_floor"]["runs"]
    assert len(floor_runs) == 4

    # The state that produced the false announcement is still there to be seen: every comparable
    # fingerprint below the canonical IS one of its own floor runs.
    assert log.previous_comparable(canonical) == 5
    assert log.cert(5)["id"] in floor_runs

    assert log.baseline_gap(canonical) is None
    assert "baseline_gap" not in log.derived_meta(6)
    for index in log.indices():
        gap = log.baseline_gap(log.cert(index))
        assert gap is None or gap["replaced_a_baseline"] is False


def test_a_canonical_is_not_measured_against_a_run_of_its_own_floor(tmp_path):
    """G1 on a log built from scratch: the same shape as the published one, freshly appended."""
    log, _battery, _plan, runs, canonical = floor_ladder(tmp_path)
    at = log.append(canonical)
    named = set(canonical["body"]["noise_floor"]["runs"])
    assert named == {c["id"] for c in runs}
    assert log.previous_comparable(canonical) is not None   # a comparable run IS on record
    assert log.cert(log.previous_comparable(canonical))["id"] in named
    assert log.baseline_gap(canonical) is None
    assert "baseline_gap" not in log.meta(at)


def test_section_55_still_sees_the_runs_the_gap_excludes(tmp_path):
    """The exclusion is in the disclosure, not in the rule, and this is why that matters.

    Section 5.5 refuses a second canonical fingerprint that carries no ``previous`` ref unless it
    is another run under the same logged noise plan. That rule reads ``previous_comparable``,
    which still counts a cert's own floor runs — otherwise a canonical could dodge the rule by
    naming the fingerprint it replaces as one of its own runs.
    """
    log, battery, _plan, runs, canonical = floor_ladder(tmp_path)
    log.append(canonical)
    assert log.previous_comparable(canonical) is not None

    # A comparable fingerprint under NO plan and with no `previous` ref is still refused.
    orphan = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=8)),
        refs=[{"role": "battery", "id": battery["id"]}],
        body=F.fingerprint_body(n=8, run_index=0, batch_size=8),
    )
    with pytest.raises(Exception) as exc:
        log.append(orphan)
    assert "needs a previous ref" in str(exc.value)


# ------------------------------------- G2: the honest and the dishonest case are not one shape


def _second_canonical(battery, plan, runs, *, item: int = 0) -> dict:
    """A SECOND canonical over the same four runs, one item moved: the baseline quietly replaced.

    Section 5.5 admits it with no ``previous`` ref because it is another run under the same logged
    noise plan — which is exactly the branch the honest sibling-run case appends under.
    """
    body = moved(F.fingerprint_body(n=8, run_index=0, batch_size=8), item=item)
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    return F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=8)),
        refs=(
            [{"role": "battery", "id": battery["id"]}]
            + [{"role": "noise_plan", "id": plan["id"]}]
            + [{"role": "run", "id": c["id"]} for c in runs]
        ),
        body=body,
    )


def test_a_second_canonical_under_one_plan_is_named_a_silent_replacement(tmp_path):
    """The dishonest side of G2: a baseline really moved, and nobody was told by a ref."""
    log, battery, plan, runs, canonical = floor_ladder(tmp_path)
    at_first = log.append(canonical)
    at_second = log.append(_second_canonical(battery, plan, runs))

    gap = log.meta(at_second)["baseline_gap"]
    assert gap["previous_index"] == at_first
    assert gap["previous_is_canonical"] is True and gap["own_is_canonical"] is True
    assert gap["relation"] == "baseline-replaced"
    assert gap["replaced_a_baseline"] is True
    assert gap["silent_replacement"] is True          # no `previous` ref named it
    assert gap["announced"] is False and gap["same_noise_plan"] is True
    assert gap["per_channel"]["exact"]["distance"] > 0.0


def test_the_honest_sibling_runs_are_not_a_replacement_and_no_longer_read_like_one(tmp_path):
    """The honest side of G2, and the discrimination the field did not have.

    Two runs of one preregistered plan and one canonical replacing another appended with the same
    three values before this: ``announced: false``, ``same_noise_plan: true``, a distance. They
    now differ on ``relation``, and ``relation`` is decided by whether either cert carries a
    ``noise_floor`` block — a fact in the log's bytes, not a label either party chose for it.
    """
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    plan = F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5})
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    log.append(fp_cert(pool["id"], extra_refs=plan_ref, body=F.fingerprint_body()))
    at1 = log.append(
        fp_cert(pool["id"], extra_refs=plan_ref, body=moved(F.fingerprint_body()))
    )

    honest = log.meta(at1)["baseline_gap"]
    assert honest["announced"] is False and honest["same_noise_plan"] is True
    assert honest["relation"] == "runs-of-one-plan"
    assert honest["replaced_a_baseline"] is False
    assert honest["silent_replacement"] is False
    assert honest["previous_is_canonical"] is False and honest["own_is_canonical"] is False

    # The three values that used to be the whole announcement are still identical between the two
    # cases; the fields that discriminate are the new ones.
    log2, battery, floor_plan, runs, canonical = floor_ladder(tmp_path / "second")
    log2.append(canonical)
    at = log2.append(_second_canonical(battery, floor_plan, runs))
    dishonest = log2.meta(at)["baseline_gap"]
    old_shape = ("announced", "same_noise_plan")
    assert [honest[k] for k in old_shape] == [dishonest[k] for k in old_shape]
    assert honest["relation"] != dishonest["relation"]
    assert honest["replaced_a_baseline"] != dishonest["replaced_a_baseline"]


def test_verify_says_no_baseline_moved_where_none_did(tmp_path):
    """The reader's half of G2: the printed headline was "baseline replaced" in both cases."""
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    plan = F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5})
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    run0 = fp_cert(pool["id"], extra_refs=plan_ref, body=F.fingerprint_body())
    log.append(run0)
    run1 = fp_cert(pool["id"], extra_refs=plan_ref, body=moved(F.fingerprint_body()))
    log.append(run1)

    out = V.diff(run1, run0, log)
    assert out.result_body["baseline_gap"]["relation"] == "runs-of-one-plan"
    assert "no baseline moved [runs-of-one-plan]" in out.printed
    assert "baseline replaced:" not in out.printed
    assert "NOT named by this cert" in out.printed      # the disclosure itself is unchanged
    # And it still moves no verdict: the same comparison with a resolver that offers no gap.
    blind = V.diff(run1, run0, _resolver(pool, plan, run0, run1))
    assert "baseline_gap" not in blind.result_body
    assert (out.exit_code, out.verdict) == (blind.exit_code, blind.verdict)


def test_a_declared_previous_ref_is_a_replacement_whoever_carries_a_floor():
    """The cert that NAMES a baseline is declaring a replacement, log or no log.

    Both sources must agree on the word, or a reader's classification depends on whether they ran
    with ``--log``.
    """
    battery = pool_cert()
    first = fp_cert(battery["id"], body=F.fingerprint_body())
    a = fp_cert(
        battery["id"],
        body=moved(F.fingerprint_body()),
        extra_refs=[{"role": "previous", "id": first["id"]}],
    )
    b = fp_cert(battery["id"], body=moved(F.fingerprint_body(), item=1))
    gap = V.diff(a, b, _resolver(battery, first, a, b)).result_body["baseline_gap"]
    assert gap["source"] == "previous_ref"
    assert gap["relation"] == "baseline-replaced"
    assert gap["replaced_a_baseline"] is True
    assert gap["silent_replacement"] is False


# --------------------------------------------------- G3: the decision about a cumulative quantity


def test_the_gap_publishes_no_cumulative_distance_and_says_why(tmp_path):
    """G3, decided: a cumulative DISTANCE does not belong in this field, and the module says why.

    The announcement is pairwise, so a baseline walked A -> B -> C in three appends publishes three
    small gaps and never A-to-C. The obvious repair is to sum along the chain, and the sum is not a
    distance: ``exact`` is the fraction of items whose outputs differ and ``seqlp``/``topk`` are
    means of per-item magnitudes, so moving one item away and back sums to a non-zero number
    between endpoints that are byte-identical. Publishing that in the very field that exists to
    stop a false announcement repeats the defect.

    What a reader needs is ``floor.baseline_gap(A_body, C_body)`` over the two ENDPOINTS, which
    this file already pins as a public function and which a reader can call on any two logged
    certs. This pins the decision as an absence plus a stated reason, in the shape
    ``test_r_exec_3_is_open_and_the_module_says_so`` uses.
    """
    log, battery, plan, runs, canonical = floor_ladder(tmp_path)
    log.append(canonical)
    at = log.append(_second_canonical(battery, plan, runs))
    gap = log.meta(at)["baseline_gap"]
    assert not [k for k in gap if "cumulative" in k or "chain" in k]

    doc = Log.baseline_gap.__doc__
    assert "G3" in doc and "does NOT belong here" in doc
    assert "none of them composes along a chain" in doc

    # And the endpoint quantity is reachable without this field: two logged bodies, one public
    # function, the same arithmetic.
    a_body = log.cert(log.find(canonical["id"]))["body"]
    c_body = log.cert(at)["body"]
    direct = FLOOR.baseline_gap(a_body, c_body)
    assert direct["per_channel"]["exact"]["distance"] == pytest.approx(
        gap["per_channel"]["exact"]["distance"]
    )
