# -*- coding: utf-8 -*-
"""styxx.absence — characterized against the defects it was built from.

The corpus below is GROUND TRUTH: each snippet is the real shape of a defect
confirmed and fixed in 7.36.0-7.38.0. They are inlined rather than read from git
history on purpose — CI clones shallow, and a detector's characterization must
not depend on how deep the clone was.

`test_recall_against_the_known_corpus` is the load-bearing test: it pins the
measured recall so a future edit cannot quietly make the screen blinder. The two
documented misses are asserted AS misses — a detector that silently starts
"passing" them would mean the corpus drifted, not that the screen improved.
"""
import pytest

from styxx.absence import LIMITS, scan_path, scan_source

# ── ground truth: the real shapes, from this repo's own history ─────────────

WITNESS_TRUTHY = '''
def substrate_divergence(self, *a, **k):
    flags = self._mount.read(*a, **k)
    v = WitnessVerdict(status="FLAG" if flags else "OK", capability=cap.name)
    return v
'''

MIDDLEWARE_DISJUNCT = '''
def choose(result, ceiling_only):
    passed = (not result.needs_revision) or ceiling_only
    return passed
'''

GATE_CRASH_TRUST = '''
def gate(client, model, prompt):
    try:
        return _gate_anthropic(client, model, prompt)
    except Exception as e:
        return GateVerdict(prompt=prompt, model=model, method="error",
                           will_refuse=0.0, will_confabulate=0.0, trust_score=1.0)
'''

COHERENCE_ABSENT_COMPOSITE = '''
def load(entry):
    return PulseSample(composite=float(entry.get("cogn_composite", 0.0)))
'''

PEARSON_UNDEFINED = '''
def _pearson_r(a, b):
    denom = math.sqrt(var_a * var_b)
    if denom == 0.0:
        return 0.0
    return cov / denom
'''

DYNAMICS_R2 = '''
def fit(self, obs):
    ss_tot = float(np.sum((S_next - S_next.mean(axis=0)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    return r2
'''

SLA_FABRICATED_MEAN = '''
def check_health(entries):
    confs = [e["phase4_conf"] for e in entries if e.get("phase4_conf") is not None]
    mean_conf = sum(confs) / len(confs) if confs else 0.5
    return mean_conf
'''

# documented misses — a syntactic screen cannot reach these
WEATHER_SENTINEL_DATAFLOW = '''
def tool_weather_report(args):
    try:
        report = styxx.weather(window=args["window"])
    except Exception:
        report = None
    if report is None:
        return {"summary": "no recent vitals", "gate": "pass"}
    return report
'''

FORECAST_ABSENT_GUARD = '''
def forecast(self, trajectories, n_tokens=None):
    feats = extract_features_v2(trajectories, n_tokens or self._horizon)
    z = (feats - self._mu) / self._sigma
    return ForecastResult(predicted_category=nearest, confidence=conf)
'''

CAUGHT_CORPUS = [
    ("witness truthy verdict", WITNESS_TRUTHY, "TRUTHY_GATE"),
    ("middleware dead disjunct", MIDDLEWARE_DISJUNCT, "TRUTHY_GATE"),
    ("gate() crash -> trust 1.0", GATE_CRASH_TRUST, "HEALTHY_ON_CRASH"),
    ("absent composite -> 0.0", COHERENCE_ABSENT_COMPOSITE, "SENTINEL_DEFAULT"),
    ("pearson denom 0 -> 0.0", PEARSON_UNDEFINED, "UNDEFINED_AS_NUMBER"),
    ("r2 1.0 on zero variance", DYNAMICS_R2, "UNDEFINED_AS_NUMBER"),
    ("mean_conf fabricated 0.5", SLA_FABRICATED_MEAN, "UNDEFINED_AS_NUMBER"),
    ("weather crash -> gate:pass", WEATHER_SENTINEL_DATAFLOW, "CRASH_TO_HEALTHY_SENTINEL"),
]

MISSED_CORPUS = [
    # The one class a syntactic screen genuinely cannot reach: forecast() had NO
    # validation at all. No pass over source can flag code that was never
    # written -- only a test or a reviewer catches an absent guard.
    ("forecast absent guard", FORECAST_ABSENT_GUARD),
]


@pytest.mark.parametrize("name,src,rule", CAUGHT_CORPUS,
                         ids=[c[0] for c in CAUGHT_CORPUS])
def test_each_known_defect_shape_is_flagged(name, src, rule):
    findings = scan_source(src, f"<{name}>")
    assert findings, f"{name}: the screen went blind to a shape it was built from"
    assert rule in {f.rule for f in findings}, \
        f"{name}: expected {rule}, got {[f.rule for f in findings]}"


@pytest.mark.parametrize("name,src", MISSED_CORPUS, ids=[c[0] for c in MISSED_CORPUS])
def test_documented_blind_spots_stay_documented(name, src):
    """The honest ceiling of a syntactic pass. If this starts passing, the
    docstring's measured recall is stale -- update the characterization rather
    than deleting this test."""
    assert not scan_source(src, f"<{name}>"), \
        f"{name} is now caught — re-measure recall and update the module docstring"


def test_recall_against_the_known_corpus():
    """The headline number in the module docstring, pinned."""
    caught = sum(bool(scan_source(src, name)) for name, src, _ in CAUGHT_CORPUS)
    total = len(CAUGHT_CORPUS) + len(MISSED_CORPUS)
    assert caught == 8 and total == 9, f"recall drifted: {caught}/{total}"


# ── the screen's own honesty rules ─────────────────────────────────────────

def test_a_cli_failure_return_is_not_a_healthy_value():
    """`1 in {True}` is True in Python, so every `return 1` — which in a CLI
    means FAILURE — used to be flagged as a healthy-on-crash return. The screen
    had the type confusion it exists to find."""
    src = '''
def main():
    try:
        run()
    except Exception:
        return 1
'''
    assert not [f for f in scan_source(src) if f.rule == "HEALTHY_ON_CRASH"]


def test_an_ordinary_string_default_is_not_a_gate_bypass():
    """`x = a.get("gate") or "pending"` is a default, not two live signals
    competing — flagging it buried the real dead-disjunct cases in noise."""
    src = 'gate = entry.get("gate") or "pending"\n'
    assert not [f for f in scan_source(src) if f.rule == "TRUTHY_GATE"]


def test_denominator_vocabulary_matches_whole_names_only():
    """`n` as a SUBSTRING matched lines/runs/entries — half the repo."""
    src = '''
def f(lines):
    if not lines:
        return 0
    return len(lines)
'''
    assert not [f for f in scan_source(src) if f.rule == "UNDEFINED_AS_NUMBER"]


def test_report_always_carries_its_limits():
    rep = scan_path(__file__)
    assert rep.files_scanned == 1
    assert LIMITS in rep.render()
    assert rep.as_dict()["limits"] == LIMITS


def test_clean_source_is_not_a_certificate():
    rep = scan_path(__file__)
    rep.findings = []
    assert "not a certificate" in rep.render()


def test_scan_path_survives_unparseable_files(tmp_path):
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("x = 1\n", encoding="utf-8")
    rep = scan_path(tmp_path)
    assert rep.files_scanned == 1
    assert len(rep.files_unparsed) == 1


def test_cli_exits_zero_even_with_findings(tmp_path, capsys):
    """A screen must never fail a build on its own, or it gets silenced."""
    from styxx.absence import main
    (tmp_path / "m.py").write_text(PEARSON_UNDEFINED, encoding="utf-8")
    assert main([str(tmp_path)]) == 0
    assert "UNDEFINED_AS_NUMBER" in capsys.readouterr().out
