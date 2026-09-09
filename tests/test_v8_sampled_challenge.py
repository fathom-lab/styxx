"""Tests for ``styxx.v8.sampled_challenge``.

Every test that needs a certificate uses the PUBLISHED one --
``papers/v8/first_verdict_2026_09_09/fp_bf16/fingerprint-canonical-73a09ffa3f1e.json``, the
64-item canonical fingerprint of ``google/gemma-2-2b-it`` -- rather than a fixture built to
pass.  The two statistical tests (the empirical detection rate, and the grinding cost) run
against that cert's own 64 item ids.

Nothing here signs or appends.  Signature checking and log order are `cert`/`log` predicates and
are named in ``sampled_challenge.LOG_SIDE_OWED``, not simulated here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from styxx.v8 import jcs
from styxx.v8 import sampled_challenge as sc

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "papers" / "v8" / "first_verdict_2026_09_09"
)
CANONICAL = ARTIFACT / "fp_bf16" / "fingerprint-canonical-73a09ffa3f1e.json"
STH = ARTIFACT / "log" / "sth" / "000000000007.json"

KEY_A = "ed25519:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
KEY_B = "ed25519:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
NONCE = "00112233445566778899aabbccddeeff"


# ---------------------------------------------------------------- fixtures off the artifact

@pytest.fixture(scope="module")
def target():
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def head():
    return json.loads(STH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def item_ids(target):
    return [i["item_id"] for i in target["body"]["items"]]


def make_selector(head, target, *, k=20, nonce=NONCE, key=KEY_A, n=64, mode="head", commit_index=None):
    return sc.selector(
        log_id=head["log_id"],
        tree_size=head["tree_size"],
        root_hash=head["root_hash"],
        target=target["id"],
        challenger_key=key,
        nonce=nonce,
        k=k,
        n=n,
        mode=mode,
        commit_index=commit_index,
    )


def honest_challenge(head, target, item_ids, *, k=20, nonce=NONCE, key=KEY_A, fabricated=()):
    """A challenge from a challenger whose own run reproduced the target exactly.

    ``fabricated`` names item ids where the CHALLENGER observes something different -- which is
    what a fabricated target item looks like from the challenger's side.
    """
    sel = make_selector(head, target, k=k, nonce=nonce, key=key)
    drawn = sc.select_items(item_ids, sel)
    tgt = sc.target_digests(target)
    own = {}
    for iid in drawn:
        rec = tgt[iid]
        if iid in fabricated:
            own[iid] = {
                "output_sha256": hashlib.sha256(("real:" + iid).encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(("realtok:" + iid).encode()).hexdigest(),
            }
        else:
            own[iid] = {"output_sha256": rec["output_sha256"], "token_ids_sha256": rec["token_ids_sha256"]}
    rows = sc.compare(drawn, target, own)
    body = sc.challenge_body(
        sc.sample_block(sel, rows),
        environment={"hardware": {"gpu": "other"}, "runtime": {"framework": "transformers"}},
        subject={"hf_repo": target["subject"]["hf_repo"]},
        recipe_core={"battery": target["recipe"]["battery"]},
        synthetic=False,
    )
    return {
        "body": body,
        "created": "2026-09-09T20:00:00Z",
        "id": "sha256:" + "c" * 64,
        "issuer": {"key": key, "name": "challenger"},
        "recipe": {},
        "refs": [{"id": target["id"], "role": "target"}, {"id": "sha256:" + "0" * 64, "role": "own"}],
        "sig": "ed25519:unsigned-in-tests",
        "styxx": "8.0",
        "subject": {},
        "type": "challenge",
    }


def ok(challenge, target, head, item_ids, **kw):
    kw.setdefault("target_index", 4)
    return sc.verify_sample(challenge, target, sth=head, battery_item_ids=item_ids, **kw)


# ---------------------------------------------------------------- the artifact is what we think

def test_artifact_shape(target, item_ids, head):
    assert target["type"] == "fingerprint"
    assert len(item_ids) == 64 == len(set(item_ids))
    assert head["tree_size"] == 7
    digests = sc.target_digests(target)
    assert len(digests) == 64
    assert all(len(d["output_sha256"]) == 64 and len(d["token_ids_sha256"]) == 64 for d in digests.values())


# ---------------------------------------------------------------- the selection function

def test_selector_is_a_commitment_to_every_input(head, target):
    base = make_selector(head, target)
    assert base["method"] == sc.SAMPLE_TAG
    assert sc.selector_digest(base) == "sha256:" + jcs.digest(base)
    # every member moves the digest
    for key, other in (
        ("nonce", "ffffffffffffffffffffffffffffffff"),
        ("challenger_key", KEY_B),
        ("root_hash", "sha256:" + "a" * 64),
        ("target", "sha256:" + "b" * 64),
    ):
        moved = dict(base, **{key: other})
        assert sc.selector_digest(moved) != sc.selector_digest(base), key
    assert sc.selector_digest(dict(base, tree_size=8)) != sc.selector_digest(base)
    assert sc.selector_digest(dict(base, k=21)) != sc.selector_digest(base)


def test_k_is_committed_but_does_not_move_the_ranking(head, target, item_ids):
    """k is inside selector_id (a commitment) and outside the seed (so samples nest).

    If k moved the ranking, a colluding challenger would get n independent permutations to
    choose a clean prefix from. This test is the receipt for that decision.
    """
    a = sc.select_items(item_ids, make_selector(head, target, k=64))
    for k in (1, 2, 20, 63):
        assert sc.select_items(item_ids, make_selector(head, target, k=k)) == a[:k]


def test_selector_refuses_bad_inputs(head, target):
    with pytest.raises(ValueError, match="nonce must be at least"):
        make_selector(head, target, nonce="00" * 8)
    with pytest.raises(ValueError, match="even-length lowercase hex"):
        make_selector(head, target, nonce="ZZ" * 16)
    with pytest.raises(ValueError, match="1 <= k <= n"):
        make_selector(head, target, k=65)
    with pytest.raises(ValueError, match="1 <= k <= n"):
        make_selector(head, target, k=0)
    with pytest.raises(ValueError, match="mode must be one of"):
        make_selector(head, target, mode="whatever")
    with pytest.raises(ValueError, match="commit_index belongs only"):
        make_selector(head, target, commit_index=3)
    with pytest.raises(ValueError, match="deciding head must come after"):
        make_selector(head, target, mode="commit-then-head", commit_index=7)
    assert make_selector(head, target, mode="commit-then-head", commit_index=6)["commit_index"] == 6


def test_selection_is_deterministic_sized_and_a_subset(head, target, item_ids):
    for k in (1, 2, 20, 64):
        sel = make_selector(head, target, k=k)
        a, b = sc.select_items(item_ids, sel), sc.select_items(list(reversed(item_ids)), sel)
        assert a == b, "the input order must not move the sample"
        assert len(a) == k == len(set(a))
        assert set(a) <= set(item_ids)


def test_samples_are_nested_in_k(head, target, item_ids):
    """k and k+1 share a prefix: raising k adds items, it does not redraw."""
    prev = sc.select_items(item_ids, make_selector(head, target, k=1))
    for k in range(2, 33):
        cur = sc.select_items(item_ids, make_selector(head, target, k=k))
        assert cur[: k - 1] == prev
        prev = cur


def test_a_different_root_gives_a_different_sample(head, target, item_ids):
    """The issuer's defence: they signed before the root existed, and the root moves the draw."""
    base = sc.select_items(item_ids, make_selector(head, target, k=20))
    moved = 0
    for i in range(40):
        alt = dict(head, root_hash="sha256:" + hashlib.sha256(str(i).encode()).hexdigest())
        if sc.select_items(item_ids, make_selector(alt, target, k=20)) != base:
            moved += 1
    assert moved == 40


def test_selection_is_not_lopsided(head, target, item_ids):
    """Every item is drawn near k/n of the time over 2000 nonces: no item is quietly safe."""
    counts = {i: 0 for i in item_ids}
    trials, k = 2000, 16
    for t in range(trials):
        sel = make_selector(head, target, k=k, nonce=f"{t:032x}")
        for iid in sc.select_items(item_ids, sel):
            counts[iid] += 1
    p = k / len(item_ids)
    mean, sd = trials * p, math.sqrt(trials * p * (1 - p))
    worst = max(abs(c - mean) for c in counts.values())
    assert worst < 5 * sd, f"worst deviation {worst:.1f} exceeds 5 sd ({5 * sd:.1f})"


# ---------------------------------------------------------------- body construction

def test_body_omits_floor_relative_members(head, target, item_ids):
    body = honest_challenge(head, target, item_ids)["body"]
    assert body["sample_of"] == "exact"
    assert "per_channel" not in body and "coverage" not in body
    assert body["sample"]["disagreements"] == 0
    assert body["sample"]["sampled_exact_distance"] == 0.0
    assert len(body["sample"]["results"]) == 20
    assert jcs.canonical_bytes(body)  # the body is JCS-serializable as it stands


def test_distance_is_disagreements_over_k(head, target, item_ids):
    sel = make_selector(head, target, k=8)
    drawn = sc.select_items(item_ids, sel)
    block = sc.sample_block(sel, sc.compare(drawn, target, {
        i: {"output_sha256": "0" * 64, "token_ids_sha256": "0" * 64} if n < 3 else
           {"output_sha256": sc.target_digests(target)[i]["output_sha256"],
            "token_ids_sha256": sc.target_digests(target)[i]["token_ids_sha256"]}
        for n, i in enumerate(drawn)
    }))
    assert block["disagreements"] == 3
    assert block["sampled_exact_distance"] == 0.375


def test_both_digests_must_match(head, target, item_ids):
    """Text agreement with token disagreement is a disagreement, and the reverse too."""
    sel = make_selector(head, target, k=4)
    drawn = sc.select_items(item_ids, sel)
    tgt = sc.target_digests(target)
    for field in ("output_sha256", "token_ids_sha256"):
        own = {i: dict(tgt[i]) for i in drawn}
        own[drawn[0]] = {k: v for k, v in own[drawn[0]].items() if k != "n_generated"}
        own[drawn[0]][field] = "9" * 64
        rows = sc.compare(drawn, target, own)
        assert rows[0]["agree"] is False, field
        assert all(r["agree"] for r in rows[1:])


# ---------------------------------------------------------------- the verification predicate

def test_honest_challenge_verifies(head, target, item_ids):
    assert ok(honest_challenge(head, target, item_ids), target, head, item_ids) == []


def test_challenger_cannot_pick_its_own_sample(head, target, item_ids):
    """The attack the selection exists to stop: a challenger drawing k items it likes."""
    ch = honest_challenge(head, target, item_ids)
    chosen = sorted(item_ids)[:20]
    tgt = sc.target_digests(target)
    ch["body"]["sample"]["item_ids"] = chosen
    ch["body"]["sample"]["results"] = sc.compare(chosen, target, {i: dict(tgt[i]) for i in chosen})
    reasons = ok(ch, target, head, item_ids)
    assert any("not the derived sample" in r for r in reasons)


def test_misquoting_the_target_is_refused(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    row = ch["body"]["sample"]["results"][7]
    row["target_output_sha256"] = row["own_output_sha256"] = "1" * 64
    reasons = ok(ch, target, head, item_ids)
    assert any("misquotes the target's output_sha256" in r for r in reasons)


def test_a_flipped_agree_flag_is_refused(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids, fabricated=())
    ch["body"]["sample"]["results"][0]["own_token_ids_sha256"] = "2" * 64
    reasons = ok(ch, target, head, item_ids)
    assert any("claims agree=True but the digests say False" in r for r in reasons)
    assert any("disagreements is 0, the rows give 1" in r for r in reasons)
    assert any("sampled_exact_distance is 0.0, the rows give 0.05" in r for r in reasons)


def test_counts_must_follow_from_the_rows(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    ch["body"]["sample"]["disagreements"] = 3
    ch["body"]["sample"]["sampled_exact_distance"] = 0.15
    reasons = ok(ch, target, head, item_ids)
    assert any("disagreements is 3, the rows give 0" in r for r in reasons)
    assert any("sampled_exact_distance is 0.15" in r for r in reasons)


def test_selector_id_must_be_the_selector(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    ch["body"]["sample"]["selector_id"] = "sha256:" + "e" * 64
    assert any("selector_id" in r for r in ok(ch, target, head, item_ids))


def test_the_key_in_the_selector_is_the_signing_key(head, target, item_ids):
    """A challenger cannot draw with one key and sign with another."""
    ch = honest_challenge(head, target, item_ids, key=KEY_A)
    ch["issuer"]["key"] = KEY_B
    assert any("is not the cert's issuer key" in r for r in ok(ch, target, head, item_ids))


def test_the_head_must_be_the_head(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    other = dict(head, root_hash="sha256:" + "d" * 64)
    reasons = sc.verify_sample(ch, target, sth=other, battery_item_ids=item_ids, target_index=4)
    assert any("selector.root_hash" in r and "is not the head's" in r for r in reasons)


def test_the_head_must_already_commit_to_the_target(head, target, item_ids):
    """The whole issuer-side defence: a head that does not contain the target fixes nothing."""
    ch = honest_challenge(head, target, item_ids)
    reasons = sc.verify_sample(ch, target, sth=head, battery_item_ids=item_ids, target_index=7)
    assert any("does not commit to the target" in r for r in reasons)
    reasons = sc.verify_sample(ch, target, sth=head, battery_item_ids=item_ids, target_index=None)
    assert any(r.startswith("UNCHECKED:") for r in reasons), "a missing index must be reported, not assumed"


def test_the_target_ref_is_named_once(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    ch["refs"].append({"id": target["id"], "role": "target"})
    assert any("target refs are" in r for r in ok(ch, target, head, item_ids))


def test_a_synthetic_challenge_does_not_challenge_a_measured_target(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    ch["body"]["synthetic"] = True
    assert any("synthetic mismatch" in r for r in ok(ch, target, head, item_ids))


def test_the_battery_must_be_the_targets(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    reasons = sc.verify_sample(ch, target, sth=head, battery_item_ids=item_ids[:-1] + ["deadbeef"], target_index=4)
    assert any("not the battery's" in r for r in reasons)


def test_non_challenge_and_empty_bodies(head, target, item_ids):
    ch = honest_challenge(head, target, item_ids)
    assert any("not 'challenge'" in r for r in ok(dict(ch, type="result"), target, head, item_ids))
    assert any("body.sample is absent" in r for r in ok(dict(ch, body={}), target, head, item_ids))


def test_a_redacted_target_is_not_sampleable(head, target, item_ids):
    stripped = copy.deepcopy(target)
    stripped["body"].pop("items")
    ch = honest_challenge(head, target, item_ids)
    assert any("not sampleable" in r for r in ok(ch, stripped, head, item_ids))


# ---------------------------------------------------------------- binding, stated honestly

def test_binding_report_does_not_overclaim(head, target):
    h = sc.binding_report(make_selector(head, target))
    assert h["removes_issuer_choice"] is True
    assert h["removes_challenger_choice"] is False
    assert any("regrinds" in r for r in h["residue"])
    c = sc.binding_report(make_selector(head, target, mode="commit-then-head", commit_index=6))
    assert c["removes_challenger_choice"] is True
    assert any("log operator" in r for r in c["residue"])
    for report in (h, c):
        assert any("report digests it never computed" in r for r in report["residue"])


def test_grinding_is_cheap_and_the_module_says_so(head, target, item_ids):
    """Mode ``head`` measured, not argued: a colluding challenger regrinds the nonce.

    Six of the 64 items are fabricated.  A challenger who wants a clean k=20 sample tries
    nonces until one misses all six; ``expected_grinding_trials(6/64, 20)`` predicts the cost.
    """
    fabricated = set(sorted(item_ids)[:6])
    trials = []
    for run in range(20):
        for attempt in range(2000):
            sel = make_selector(head, target, k=20, nonce=f"{run:016x}{attempt:016x}")
            if not (set(sc.select_items(item_ids, sel)) & fabricated):
                trials.append(attempt + 1)
                break
        else:  # pragma: no cover
            pytest.fail("no clean nonce found in 2000 tries")
    predicted = sc.expected_grinding_trials(6 / 64, 20)
    mean = sum(trials) / len(trials)
    assert 1 <= mean <= 10 * predicted
    assert predicted < 20, f"grinding cost {predicted:.1f} draws is not a defence"


# ---------------------------------------------------------------- detection power

def test_power_pinned_values():
    assert sc.detection_with_replacement(0.1, 20) == pytest.approx(0.8784233454, abs=1e-9)
    assert sc.detection_hypergeometric(64, 6, 20) == pytest.approx(0.9058471290, abs=1e-9)
    assert sc.detection_hypergeometric(64, 0, 64) == 0.0
    assert sc.detection_hypergeometric(64, 1, 64) == 1.0
    assert sc.detection_hypergeometric(64, 1, 32) == pytest.approx(0.5, abs=1e-12)
    assert sc.detection_hypergeometric(64, 6, 0) == 0.0


def test_power_is_monotone_and_without_replacement_wins():
    for m in (1, 3, 6, 16, 32):
        f = m / 64
        prev = -1.0
        for k in range(0, 65):
            h = sc.detection_hypergeometric(64, m, k)
            assert h >= prev
            prev = h
            assert h >= sc.detection_with_replacement(f, k) - 1e-12


def test_k_for_power_is_the_smallest_k():
    for m in (1, 3, 6, 16, 32):
        k = sc.k_for_power(64, m, 0.95)
        assert k is not None
        assert sc.detection_hypergeometric(64, m, k) >= 0.95
        assert k == 0 or sc.detection_hypergeometric(64, m, k - 1) < 0.95
    assert sc.k_for_power(64, 0, 0.95) is None


def test_power_table_row_shape():
    t = sc.power_table()
    assert t["ks"] == [1, 2, 4, 8, 16, 32, 64] and t["n"] == 64
    fs = [r["f"] for r in t["rows"]]
    assert fs == [0.01, 0.05, 0.1, 0.25, 0.5]
    assert [r["m"] for r in t["rows"]] == [1, 3, 6, 16, 32]
    assert [r["k_for_power"] for r in t["rows"]] == [61, 40, 25, 10, 5]


def test_empirical_detection_matches_the_formula(head, target, item_ids):
    """The claimed power, measured against a real forged cert instead of asserted.

    Six of the published cert's 64 items are replaced with fabricated digests.  2000 honest
    challengers each draw k=20 with their own nonce and report what they observed (the ORIGINAL
    digests, since they ran the real model).  The rate at which the challenge lands a
    disagreement must track ``detection_hypergeometric(64, 6, 20)`` = 0.9058.
    """
    forged = copy.deepcopy(target)
    fabricated = set(sorted(item_ids)[:6])
    truth = sc.target_digests(target)
    for item in forged["body"]["items"]:
        if item["item_id"] in fabricated:
            item["output_sha256"] = hashlib.sha256(("forged:" + item["item_id"]).encode()).hexdigest()
            item["token_ids_sha256"] = hashlib.sha256(("forgedtok:" + item["item_id"]).encode()).hexdigest()

    rng = random.Random(20260909)
    trials, caught = 2000, 0
    for _ in range(trials):
        sel = make_selector(head, forged, k=20, nonce=f"{rng.getrandbits(128):032x}")
        drawn = sc.select_items(item_ids, sel)
        rows = sc.compare(drawn, forged, {i: dict(truth[i]) for i in drawn})
        block = sc.sample_block(sel, rows)
        if block["disagreements"] > 0:
            caught += 1
        # the challenge is well-formed whatever it found
        assert block["sampled_exact_distance"] == round(block["disagreements"] / 20, 9)

    rate = caught / trials
    predicted = sc.detection_hypergeometric(64, 6, 20)
    sd = math.sqrt(predicted * (1 - predicted) / trials)
    assert abs(rate - predicted) < 4 * sd, f"empirical {rate:.4f} vs predicted {predicted:.4f} (sd {sd:.4f})"


def test_a_challenge_that_catches_the_forgery_verifies_as_a_disagreement(head, target, item_ids):
    """A caught forgery is a valid challenge reporting distance > 0 -- not an error."""
    forged = copy.deepcopy(target)
    victim = sorted(item_ids)[0]
    truth = sc.target_digests(target)
    for item in forged["body"]["items"]:
        if item["item_id"] == victim:
            item["output_sha256"] = "7" * 64
            item["token_ids_sha256"] = "8" * 64

    for attempt in range(500):
        sel = make_selector(head, forged, k=20, nonce=f"{attempt:032x}")
        drawn = sc.select_items(item_ids, sel)
        if victim in drawn:
            break
    else:  # pragma: no cover
        pytest.fail("victim never drawn")

    rows = sc.compare(drawn, forged, {i: dict(truth[i]) for i in drawn})
    body = sc.challenge_body(
        sc.sample_block(sel, rows),
        environment={}, subject={}, recipe_core={}, synthetic=False,
    )
    ch = {
        "body": body, "created": "2026-09-09T20:00:00Z", "id": "sha256:" + "c" * 64,
        "issuer": {"key": KEY_A, "name": "challenger"}, "recipe": {},
        "refs": [{"id": forged["id"], "role": "target"}], "sig": "ed25519:unsigned-in-tests",
        "styxx": "8.0", "subject": {}, "type": "challenge",
    }
    assert sc.verify_sample(ch, forged, sth=head, battery_item_ids=item_ids, target_index=4) == []
    assert body["sample"]["disagreements"] >= 1
    assert body["sample"]["sampled_exact_distance"] > 0


# ---------------------------------------------------------------- scope, kept in the module

def test_scope_strings_are_present_and_specific():
    assert "noise floor" in sc.DOES_NOT_ESTABLISH
    assert "seqlp" in sc.DOES_NOT_ESTABLISH
    assert "colluded" in sc.ESTABLISHES
    assert "partial fingerprint" in sc.LOG_SIDE_OWED
    for text in (sc.ESTABLISHES, sc.DOES_NOT_ESTABLISH, sc.LOG_SIDE_OWED):
        low = text.lower()
        for word in ("immutable", "tamper-proof", "unbreakable", "first-ever", "proves"):
            assert word not in low, word
