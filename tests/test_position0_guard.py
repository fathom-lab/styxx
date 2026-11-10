# -*- coding: utf-8 -*-
"""Tests for the position-0 guard (styxx._position0) and the functions
that use it.

A one-token prefill whose tokenizer added no start token puts that token
alone at position 0. Reading the hidden state there reads the large
activation models carry at position 0, not the prompt. These tests use
small stand-in tokenizers and models only: no real model, tokenizer or
weights are loaded, and nothing is downloaded. They check where the guard
fires and that it fires before any forward pass or generation; they do
not measure any activation size.
"""
from __future__ import annotations

import types
import warnings

import pytest

torch = pytest.importorskip("torch")

from styxx._position0 import (  # noqa: E402
    ChatTemplateFallbackWarning,
    Position0Error,
    Position0Warning,
    check_position0,
    reads_position0,
)
from styxx.residual_probe.probe import ProbeVerdict, StyxxProbe  # noqa: E402
from styxx.residual_probe.intervene import (  # noqa: E402
    InterveneProbe,
    InterventionResult,
)
from styxx import hallucination as hall  # noqa: E402
from styxx import cogvm  # noqa: E402
from styxx.cogvm import GENERATE, HALT, WATCH, Program, ProgramResult  # noqa: E402
from styxx.generate_safe import SafeResponse, generate_safe  # noqa: E402
from styxx.d_axis import DAxisScorer  # noqa: E402

H = 8          # hidden size of the stand-in model
V = 64         # vocabulary size of the stand-in model
EOS = 2
BOS = 1
TOKEN = 42     # an ordinary token id


# ══════════════════════════════════════════════════════════════════
# Stand-ins
# ══════════════════════════════════════════════════════════════════

class _Tok:
    """Stand-in tokenizer. Raw tokenization returns `raw` (a list of ids);
    the chat template returns `template` or raises `template_raises`."""

    def __init__(self, raw, *, bos_token_id=None, template=None,
                 template_raises=None):
        self.raw = list(raw)
        self.bos_token_id = bos_token_id
        self.eos_token_id = EOS
        self.template = template if template is not None else [5, 6, TOKEN, 7]
        self.template_raises = template_raises
        self.raw_calls = 0

    def __call__(self, text, return_tensors="pt"):
        self.raw_calls += 1
        return types.SimpleNamespace(
            input_ids=torch.tensor([self.raw], dtype=torch.long))

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            return_tensors="pt"):
        if self.template_raises is not None:
            raise self.template_raises
        return torch.tensor([self.template], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class _Handle:
    def __init__(self, hooks, fn):
        self._hooks, self._fn = hooks, fn

    def remove(self):
        if self._fn in self._hooks:
            self._hooks.remove(self._fn)


class _Layer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return _Handle(self.hooks, fn)

    def run(self, hs):
        for fn in list(self.hooks):
            r = fn(self, (), hs)
            if r is not None:
                hs = r[0] if isinstance(r, tuple) else r
        return hs


class _Model:
    """Stand-in causal LM with two decoder layers at model.model.layers.
    Every forward pass and generate call is recorded; the next token is
    always EOS so loops end after one step."""

    def __init__(self):
        self._param = torch.zeros(1, dtype=torch.float32)
        self.model = types.SimpleNamespace(layers=[_Layer(), _Layer()])
        self.config = types.SimpleNamespace(_name_or_path="stub/model")
        self.forward_calls = 0
        self.generate_calls = 0

    def parameters(self):
        yield self._param

    def _forward(self, input_ids):
        seq = int(input_ids.shape[1])
        hs = torch.full((1, seq, H), 0.5)
        states = [hs]
        for layer in self.model.layers:
            hs = layer.run(hs.clone())
            states.append(hs)
        logits = torch.zeros((1, seq, V))
        logits[:, :, EOS] = 1.0
        return types.SimpleNamespace(hidden_states=states, logits=logits,
                                     past_key_values=object())

    def __call__(self, input_ids=None, output_hidden_states=False,
                 use_cache=False, past_key_values=None):
        self.forward_calls += 1
        return self._forward(input_ids)

    def generate(self, input_ids, **kwargs):
        self.generate_calls += 1
        self._forward(input_ids)                       # prefill
        step = torch.tensor([[EOS]], dtype=torch.long)
        self._forward(step)                            # one decode step
        return torch.cat([input_ids, step], dim=1)


def _probe(cls=InterveneProbe):
    return cls(
        model="stub/model", task="confab_prompt", layer=0, total_layers=2,
        weight=torch.arange(H, dtype=torch.float32) / H, bias=0.0,
        positive_class="confab", negative_class="factual",
    )


def _pos0_warnings(record):
    return [w for w in record if issubclass(w.category, Position0Warning)]


ONE = dict(raw=[TOKEN])                          # one token, no start token
TWO = dict(raw=[TOKEN, TOKEN + 1])               # two tokens
WITH_BOS = dict(raw=[BOS, TOKEN], bos_token_id=BOS)   # start token prepended
LONE_BOS = dict(raw=[BOS], bos_token_id=BOS)          # the start token alone


# ══════════════════════════════════════════════════════════════════
# Helper unit tests
# ══════════════════════════════════════════════════════════════════

def test_error_is_value_error():
    assert issubclass(Position0Error, ValueError)
    assert issubclass(Position0Warning, UserWarning)
    assert issubclass(ChatTemplateFallbackWarning, UserWarning)


def test_helper_length_zero_is_false():
    assert reads_position0([]) is False
    assert reads_position0([[]]) is False
    assert reads_position0(torch.zeros((1, 0), dtype=torch.long)) is False
    assert check_position0(torch.zeros((1, 0), dtype=torch.long)) is False


def test_helper_tokenizer_without_bos_counts_as_none():
    tok = types.SimpleNamespace()                 # no bos_token_id attribute
    assert reads_position0([TOKEN], tok) is True
    tok2 = types.SimpleNamespace(bos_token_id=None)
    assert reads_position0([[TOKEN]], tok2) is True
    assert reads_position0([TOKEN]) is True


def test_helper_bos_only_single_token_is_false():
    tok = types.SimpleNamespace(bos_token_id=BOS)
    assert reads_position0([BOS], tok) is False
    assert reads_position0(torch.tensor([[BOS]]), tok) is False
    assert reads_position0([TOKEN], tok) is True
    # the bos_token_id argument wins over the tokenizer
    assert reads_position0([TOKEN], tok, bos_token_id=TOKEN) is False


def test_helper_list_vs_tensor_and_batch_shape():
    for ids in ([TOKEN], [[TOKEN]], torch.tensor([TOKEN]),
                torch.tensor([[TOKEN]]), (TOKEN,)):
        assert reads_position0(ids) is True, ids
    for ids in ([TOKEN, 3], [[TOKEN, 3]], torch.tensor([[TOKEN, 3]]),
                torch.tensor([[BOS, TOKEN]])):
        assert reads_position0(ids) is False, ids
    assert torch.tensor([[TOKEN]]).shape == (1, 1)
    assert reads_position0({"input_ids": torch.tensor([[TOKEN]])}) is True


def test_check_helper_raises_warns_or_passes():
    with pytest.raises(Position0Error) as ei:
        check_position0([TOKEN], where="f")
    msg = str(ei.value)
    assert "f was about to read" in msg
    assert "apply_chat_template=True" in msg
    assert "allow_position0=True" in msg
    assert "two or more tokens" in msg
    assert "ERRATUM_real_convergence_2026_11_10.md" in msg
    with pytest.warns(Position0Warning):
        assert check_position0([TOKEN], allow_position0=True, where="f") is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_position0([TOKEN, 3]) is False


# ══════════════════════════════════════════════════════════════════
# StyxxProbe.predict_before_generation and gate
# ══════════════════════════════════════════════════════════════════

def _gate_raw(probe, model, tok, **kw):
    # gate always uses the chat template; make the template one token long
    tok.template = [TOKEN]
    return probe.gate(model, tok, "hi", **kw)


def test_probe_predict_refuses_one_token_before_forward():
    probe, model = _probe(StyxxProbe), _Model()
    with pytest.raises(Position0Error):
        probe.predict_before_generation(model, _Tok(**ONE), "hi",
                                        apply_chat_template=False)
    assert model.forward_calls == 0


def test_probe_gate_refuses_one_token_before_forward():
    probe, model = _probe(StyxxProbe), _Model()
    with pytest.raises(Position0Error):
        _gate_raw(probe, model, _Tok(**ONE))
    assert model.forward_calls == 0


@pytest.mark.parametrize("raw", [TWO, WITH_BOS, LONE_BOS])
def test_probe_passes_two_tokens_or_start_token(raw):
    probe, model = _probe(StyxxProbe), _Model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = probe.predict_before_generation(model, _Tok(**raw), "hi there",
                                            apply_chat_template=False)
    assert model.forward_calls == 1
    assert v.position0 is False
    assert "position0" not in v.as_dict()
    assert "position0" not in repr(v)
    tok = _Tok(**raw)
    tok.template = raw["raw"]
    g = probe.gate(_Model(), tok, "hi there")
    assert g.n_tokens_in_prefill == len(raw["raw"])
    assert "position0" not in g.as_dict()


def test_probe_allow_position0_marks():
    probe, model = _probe(StyxxProbe), _Model()
    with pytest.warns(Position0Warning):
        v = probe.predict_before_generation(model, _Tok(**ONE), "hi",
                                            apply_chat_template=False,
                                            allow_position0=True)
    assert model.forward_calls == 1
    assert v.position0 is True and v.as_dict()["position0"] is True
    with pytest.warns(Position0Warning):
        g = _gate_raw(probe, _Model(), _Tok(**ONE), allow_position0=True)
    assert g.position0 is True


def test_probe_chat_template_path_unchanged():
    probe, model = _probe(StyxxProbe), _Model()
    v = probe.predict_before_generation(model, _Tok(**ONE), "hi")
    assert v.n_tokens_in_prefill == 4 and v.position0 is False


# ══════════════════════════════════════════════════════════════════
# hallucination: stream_with_risk, detect_hallucination, verdict
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def stub_probe(monkeypatch):
    p = _probe()
    monkeypatch.setattr(hall, "_resolve_probe", lambda model, task: p)
    return p


def test_stream_with_risk_refuses_before_forward(stub_probe):
    model = _Model()
    gen = hall.stream_with_risk(model, _Tok(**ONE), "hi",
                                apply_chat_template=False, max_new_tokens=3)
    with pytest.raises(Position0Error):
        next(gen)
    assert model.forward_calls == 0
    assert model.model.layers[0].hooks == []      # hook removed


@pytest.mark.parametrize("raw", [TWO, WITH_BOS, LONE_BOS])
def test_stream_with_risk_passes(stub_probe, raw):
    model = _Model()
    out = list(hall.stream_with_risk(model, _Tok(**raw), "hi there",
                                     apply_chat_template=False,
                                     max_new_tokens=3))
    assert len(out) == 1 and model.forward_calls == 1
    assert out[0].position0 is False and "position0" not in repr(out[0])


def test_stream_with_risk_allow_marks(stub_probe):
    model = _Model()
    with pytest.warns(Position0Warning):
        out = list(hall.stream_with_risk(model, _Tok(**ONE), "hi",
                                         apply_chat_template=False,
                                         max_new_tokens=3,
                                         allow_position0=True))
    assert len(out) == 1 and out[0].position0 is True


def test_detect_hallucination_refuses_before_forward(stub_probe):
    model = _Model()
    with pytest.raises(Position0Error):
        hall.detect_hallucination(model, _Tok(**ONE), "hi",
                                  apply_chat_template=False, max_new_tokens=3)
    assert model.forward_calls == 0
    assert model.model.layers[0].hooks == []


@pytest.mark.parametrize("raw", [TWO, WITH_BOS, LONE_BOS])
def test_detect_hallucination_passes(stub_probe, raw):
    model = _Model()
    v = hall.detect_hallucination(model, _Tok(**raw), "hi there",
                                  apply_chat_template=False, max_new_tokens=3)
    assert v.output_tokens == 1 and model.forward_calls == 1
    assert v.position0 is False
    assert "position0" not in v.as_dict() and "position0" not in repr(v)


def test_detect_hallucination_allow_marks(stub_probe):
    model = _Model()
    with pytest.warns(Position0Warning):
        v = hall.detect_hallucination(model, _Tok(**ONE), "hi",
                                      apply_chat_template=False,
                                      max_new_tokens=3, allow_position0=True)
    assert v.position0 is True and v.as_dict()["position0"] is True


def test_hallucination_verdict_refuses_before_forward(stub_probe):
    model = _Model()
    with pytest.raises(Position0Error):
        hall.hallucination_verdict(model, _Tok(**ONE), "hi", "",
                                   apply_chat_template=False)
    assert model.forward_calls == 0
    # the chat-template path fires only when the template gives one token
    tok = _Tok(**TWO, template=[TOKEN])
    with pytest.raises(Position0Error):
        hall.hallucination_verdict(model, tok, "hi", "there")
    assert model.forward_calls == 0


@pytest.mark.parametrize("raw", [TWO, WITH_BOS, LONE_BOS])
def test_hallucination_verdict_passes(stub_probe, raw):
    model = _Model()
    v = hall.hallucination_verdict(model, _Tok(**raw), "hi", "there",
                                   apply_chat_template=False)
    assert model.forward_calls == 1 and v.output_tokens == len(raw["raw"])
    assert "position0" not in v.as_dict()


def test_hallucination_verdict_allow_marks(stub_probe):
    model = _Model()
    with pytest.warns(Position0Warning):
        v = hall.hallucination_verdict(model, _Tok(**ONE), "hi", "",
                                       apply_chat_template=False,
                                       allow_position0=True)
    assert model.forward_calls == 1 and v.position0 is True


# ══════════════════════════════════════════════════════════════════
# cogvm Program.run
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def stub_cogvm(monkeypatch):
    p = _probe()
    monkeypatch.setattr(cogvm, "_load_probes_for_profile_and_watches",
                        lambda name, profile, watches: {"confab_prompt": p})
    return Program([GENERATE(max_new_tokens=3,
                             watches=[WATCH("confab_prompt > 0.999",
                                            HALT())])])


def test_cogvm_refuses_before_forward(stub_cogvm):
    model = _Model()
    with pytest.raises(Position0Error):
        stub_cogvm.run(model=model, tokenizer=_Tok(**ONE), prompt="hi",
                       apply_chat_template=False)
    assert model.forward_calls == 0
    assert all(layer.hooks == [] for layer in model.model.layers)


@pytest.mark.parametrize("raw", [TWO, WITH_BOS, LONE_BOS])
def test_cogvm_passes(stub_cogvm, raw):
    model = _Model()
    r = stub_cogvm.run(model=model, tokenizer=_Tok(**raw), prompt="hi there",
                       apply_chat_template=False)
    assert model.forward_calls == 1 and r.output_tokens == 1
    assert r.position0 is False
    assert "position0" not in r.as_dict() and "position0" not in repr(r)


def test_cogvm_allow_marks(stub_cogvm):
    model = _Model()
    with pytest.warns(Position0Warning):
        r = stub_cogvm.run(model=model, tokenizer=_Tok(**ONE), prompt="hi",
                           apply_chat_template=False, allow_position0=True)
    assert model.forward_calls == 1
    assert r.position0 is True and r.as_dict()["position0"] is True


# ══════════════════════════════════════════════════════════════════
# generate_safe
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def stub_gs(monkeypatch):
    pytest.importorskip("transformers")
    p = _probe()
    monkeypatch.setattr(InterveneProbe, "from_pretrained",
                        classmethod(lambda cls, model, task: p))
    return p


def test_generate_safe_refuses_before_forward(stub_gs):
    model = _Model()
    with pytest.raises(Position0Error):
        generate_safe(model, "hi", tokenizer=_Tok(**ONE),
                      apply_chat_template=False, max_new_tokens=3)
    assert model.forward_calls == 0
    assert model.model.layers[0].hooks == []


@pytest.mark.parametrize("raw", [TWO, WITH_BOS, LONE_BOS])
def test_generate_safe_passes(stub_gs, raw):
    model = _Model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = generate_safe(model, "hi there", tokenizer=_Tok(**raw),
                          apply_chat_template=False, max_new_tokens=3)
    assert model.forward_calls == 1 and r.tokens_generated == 1
    assert "position0" not in r.as_dict() and "position0" not in repr(r)


def test_generate_safe_allow_marks(stub_gs):
    model = _Model()
    with pytest.warns(Position0Warning):
        r = generate_safe(model, "hi", tokenizer=_Tok(**ONE),
                          apply_chat_template=False, max_new_tokens=3,
                          allow_position0=True)
    assert model.forward_calls == 1
    assert r.position0 is True and r.as_dict()["position0"] is True


def test_generate_safe_template_fallback_warns_then_guards(stub_gs):
    model = _Model()
    tok = _Tok(**ONE, template_raises=RuntimeError("no template"))
    with pytest.warns(ChatTemplateFallbackWarning, match="RuntimeError"):
        with pytest.raises(Position0Error):
            generate_safe(model, "hi", tokenizer=tok, max_new_tokens=3)
    assert tok.raw_calls == 1 and model.forward_calls == 0

    model2 = _Model()
    tok2 = _Tok(**TWO, template_raises=KeyError("x"))
    with pytest.warns(ChatTemplateFallbackWarning, match="KeyError"):
        r = generate_safe(model2, "hi there", tokenizer=tok2,
                          max_new_tokens=3)
    assert model2.forward_calls == 1 and r.position0 is False


def test_generate_safe_template_ok_no_fallback_warning(stub_gs):
    model = _Model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = generate_safe(model, "hi", tokenizer=_Tok(**ONE),
                          max_new_tokens=3)
    assert r.tokens_generated == 1


# ══════════════════════════════════════════════════════════════════
# d_axis DAxisScorer.score_single
# ══════════════════════════════════════════════════════════════════

class _TLModel:
    def __init__(self, bos_token_id):
        self.tokenizer = types.SimpleNamespace(bos_token_id=bos_token_id)
        self.calls = 0

    def run_with_hooks(self, input_ids, fwd_hooks):
        self.calls += 1
        seq = int(input_ids.shape[-1])
        for _name, fn in fwd_hooks:
            fn(torch.full((1, seq, H), 0.5), None)
        return torch.zeros((1, seq, V))


def _scorer(bos_token_id):
    s = DAxisScorer(model_name="stub/model", device="cpu")
    s._model = _TLModel(bos_token_id)       # _ensure_loaded then loads nothing
    s._W_U = torch.ones((H, V))
    s._hook_name = "blocks.0.hook_resid_post"
    return s


def test_d_axis_refuses_before_forward():
    for bos in (None, BOS):
        s = _scorer(bos)
        with pytest.raises(Position0Error):
            s.score_single(torch.tensor([[TOKEN]]), TOKEN)
        assert s._model.calls == 0


@pytest.mark.parametrize("ids,bos", [([[TOKEN, TOKEN + 1]], None),
                                     ([[BOS, TOKEN]], BOS),
                                     ([[BOS]], BOS)])
def test_d_axis_passes(ids, bos):
    s = _scorer(bos)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = s.score_single(torch.tensor(ids), TOKEN)
    assert isinstance(d, float) and s._model.calls == 1


def test_d_axis_allow_warns():
    s = _scorer(None)
    with pytest.warns(Position0Warning):
        d = s.score_single(torch.tensor([[TOKEN]]), TOKEN,
                           allow_position0=True)
    assert isinstance(d, float) and s._model.calls == 1


# ══════════════════════════════════════════════════════════════════
# intervene: marks unmeasured scores, does not refuse
# ══════════════════════════════════════════════════════════════════

def test_intervene_one_token_marks_unmeasured():
    probe, model = _probe(), _Model()
    with pytest.warns(Position0Warning, match="not measured"):
        r = probe.intervene_and_generate(model, _Tok(**ONE), "hi",
                                         apply_chat_template=False,
                                         max_new_tokens=2)
    assert model.generate_calls == 1
    assert r.scores_measured is False
    assert r.original_score == 0.5 and r.patched_score == 0.5
    assert "not measured" in r.note and "default of 0.5" in r.note
    assert "flipped is False by default" in r.note
    assert r.as_dict()["scores_measured"] is False
    assert "scores_measured" not in repr(r)


@pytest.mark.parametrize("raw", [TWO, WITH_BOS])
def test_intervene_two_tokens_measured(raw):
    probe, model = _probe(), _Model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = probe.intervene_and_generate(model, _Tok(**raw), "hi there",
                                         apply_chat_template=False,
                                         max_new_tokens=2)
    assert r.scores_measured is True and r.note == ""
    assert r.original_score != 0.5
    assert "scores_measured" not in r.as_dict()


# ══════════════════════════════════════════════════════════════════
# Normal-path shape of every result type is unchanged
# ══════════════════════════════════════════════════════════════════

def test_result_types_unchanged_on_normal_path():
    v = ProbeVerdict(model="m", task="t", layer=0, total_layers=1,
                     residual_score=0.0, p_positive=0.5,
                     positive_class="+", negative_class="-", confidence=0.0,
                     n_tokens_in_prefill=3, probe_version="v0",
                     atlas_version="v0")
    assert set(v.as_dict()) == {
        "model", "task", "layer", "total_layers", "residual_score",
        "p_positive", "positive_class", "negative_class", "confidence",
        "n_tokens_in_prefill", "probe_version", "atlas_version"}
    assert "position0" not in repr(v)
    marked = ProbeVerdict(**{**v.as_dict(), "position0": True})
    assert marked == v                     # excluded from equality

    hv = hall.HallucinationVerdict(
        prompt="p", output_text="o", output_tokens=1, risk_score=0.1,
        flagged_tokens=[], max_risk=0.1, halt_reason="", retries_used=0,
        risk_timeline=[0.1], probe_task="t", probe_layer=0, probe_auc=None,
        threshold=0.7)
    assert set(hv.as_dict()) == {
        "prompt", "output_text", "output_tokens", "risk_score",
        "flagged_tokens", "max_risk", "halt_reason", "retries_used",
        "risk_timeline", "probe_task", "probe_layer", "probe_auc",
        "threshold"}
    assert "position0" not in repr(hv)

    pr = ProgramResult(output_text="o", output_tokens=1)
    assert set(pr.as_dict()) == {
        "output_text", "output_tokens", "halt_reason", "retries_used",
        "trace", "final_profile", "probe_readings_last"}
    assert "position0" not in repr(pr)

    sr = SafeResponse(text="t", halted=False, halt_reason="",
                      tokens_generated=1)
    assert set(sr.as_dict()) == {
        "text", "halted", "halt_reason", "tokens_generated",
        "probe_trajectory"}
    assert "position0" not in repr(sr)

    ir = InterventionResult(prompt="p", target_class="c", alpha_used=1.0,
                            layer_patched=0, original_score=0.3,
                            patched_score=0.7, output_text="o",
                            output_tokens=1, flipped=True,
                            generation_completed=True)
    assert set(ir.as_dict()) == {
        "prompt", "target_class", "alpha_used", "layer_patched",
        "original_score", "patched_score", "output_text", "output_tokens",
        "flipped", "generation_completed", "note"}
    assert "scores_measured" not in repr(ir)

    tr = hall.TokenReading(token_id=1, token_text="a", risk=0.1,
                           will_flag=False)
    assert repr(tr) == ("TokenReading(token_id=1, token_text='a', risk=0.1, "
                        "will_flag=False)")


# ==================================================================
# Fixes from the review: hook removal under warnings-as-errors, warning
# location, the zero-token verdict mark, the intervene note separator
# ==================================================================

def test_hooks_removed_when_warning_is_an_error(stub_probe):
    for call in (
        lambda m: next(hall.stream_with_risk(
            m, _Tok(**ONE), "hi", apply_chat_template=False,
            max_new_tokens=3, allow_position0=True)),
        lambda m: hall.detect_hallucination(
            m, _Tok(**ONE), "hi", apply_chat_template=False,
            max_new_tokens=3, allow_position0=True),
    ):
        model = _Model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Position0Warning)
            with pytest.raises(Position0Warning):
                call(model)
        assert model.forward_calls == 0
        assert all(layer.hooks == [] for layer in model.model.layers)


def _warned_here(record):
    ws = _pos0_warnings(record)
    assert len(ws) >= 1
    return all(w.filename == __file__ for w in ws)


def test_warning_points_at_the_caller_through_wrappers(stub_probe,
                                                       stub_cogvm):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _gate_raw(_probe(StyxxProbe), _Model(), _Tok(**ONE),
                  allow_position0=True)
    assert _warned_here(rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hall.detect_hallucination(_Model(), _Tok(**ONE), "hi",
                                  apply_chat_template=False,
                                  max_new_tokens=3, allow_position0=True)
    assert _warned_here(rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        list(hall.stream_with_risk(_Model(), _Tok(**ONE), "hi",
                                   apply_chat_template=False,
                                   max_new_tokens=3, allow_position0=True))
    assert _warned_here(rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        stub_cogvm.run(model=_Model(), tokenizer=_Tok(**ONE), prompt="hi",
                       apply_chat_template=False, allow_position0=True)
    assert _warned_here(rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _scorer(None).score_single(torch.tensor([[TOKEN]]), TOKEN,
                                   allow_position0=True)
    assert _warned_here(rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        check_position0([TOKEN], allow_position0=True)
    assert _warned_here(rec)


def test_detect_hallucination_zero_tokens_still_marked(stub_probe):
    model = _Model()
    with pytest.warns(Position0Warning):
        v = hall.detect_hallucination(model, _Tok(**ONE), "hi",
                                      apply_chat_template=False,
                                      max_new_tokens=0, allow_position0=True)
    assert v.output_tokens == 0 and v.position0 is True


def test_intervene_failed_generation_note_has_separator():
    probe, model = _probe(), _Model()

    def boom(input_ids, **kwargs):
        raise RuntimeError("gen boom")

    model.generate = boom
    with pytest.warns(Position0Warning, match="not measured"):
        r = probe.intervene_and_generate(model, _Tok(**TWO), "hi there",
                                         apply_chat_template=False,
                                         max_new_tokens=2)
    assert r.scores_measured is False
    assert "gen boom. The probe scores were not measured" in r.note


def test_helper_accepts_range():
    assert reads_position0(range(TOKEN, TOKEN + 1)) is True
    assert reads_position0([range(TOKEN, TOKEN + 2)]) is False
    assert reads_position0(range(BOS, BOS + 1), bos_token_id=BOS) is False
