"""Tests for styxx.v8.runner_hf.

The model-backed tests are defined only when ``STYXX_V8_HF_TESTS=1`` (nothing skips: when the
variable is unset they do not exist, when it is set and the snapshot is missing they FAIL).
Without it, this file checks that the module imports without torch being touched and that the
Appendix A.2 hashing is right on a synthetic snapshot directory, torch-free.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from styxx.v8.runner import Runner
from styxx.v8.runner_hf import (
    TOKENIZER_FILES,
    TransformersRunner,
    snapshot_hashes,
    subject_from_snapshot,
)

ROOT = Path(__file__).resolve().parent.parent
HF_ENABLED = os.environ.get("STYXX_V8_HF_TESTS") == "1"
SNAPSHOT = (
    "C:/Users/heyzo/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ENV_STUB = {
    "runtime": {"framework": "transformers", "version": "0", "backend": "torch 0"},
    "hardware": {"gpu": "none", "driver": "none", "count": 0},
}


def test_module_imports_without_touching_torch():
    code = (
        "import sys, styxx.v8.runner_hf as m; "
        "print('torch' in sys.modules, 'transformers' in sys.modules, "
        "hasattr(m, 'TransformersRunner'), hasattr(m, 'subject_from_snapshot'))"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), check=False)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["False", "False", "True", "True"]


def _h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_snapshot(root: Path, *, with_generation_config: bool = True) -> Path:
    snap = root / "models--Acme--Tiny-Instruct" / "snapshots" / "0123abcd"
    snap.mkdir(parents=True)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"shard two bytes")
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"shard one bytes")
    (snap / "config.json").write_bytes(b'{"model_type": "tiny"}')
    if with_generation_config:
        (snap / "generation_config.json").write_bytes(b'{"do_sample": false}')
    (snap / "tokenizer.json").write_bytes(b"tok json")
    (snap / "tokenizer_config.json").write_bytes(b"tok config")
    (snap / "vocab.json").write_bytes(b"vocab")
    (snap / "README.md").write_bytes(b"not hashed")
    return snap


def test_snapshot_hashes_follow_appendix_a2(tmp_path):
    snap = _make_snapshot(tmp_path)
    got = snapshot_hashes(snap)
    # weights: (filename, digest) pairs sorted by filename, joined by "\n" as "name digest"
    weights_text = "\n".join([
        f"model-00001-of-00002.safetensors {_h(b'shard one bytes')}",
        f"model-00002-of-00002.safetensors {_h(b'shard two bytes')}",
    ])
    assert got["weights_sha256"] == _h(weights_text.encode())
    assert got["config_sha256"] == _h(b'{"model_type": "tiny"}')
    assert got["generation_config_sha256"] == _h(b'{"do_sample": false}')
    tok_text = "\n".join([
        f"tokenizer.json {_h(b'tok json')}",
        f"tokenizer_config.json {_h(b'tok config')}",
        f"vocab.json {_h(b'vocab')}",
    ])
    assert got["tokenizer_sha256"] == _h(tok_text.encode())
    assert all(HEX64.match(v) for v in got.values())


def test_generation_config_absent_hashes_empty_string(tmp_path):
    snap = _make_snapshot(tmp_path, with_generation_config=False)
    assert snapshot_hashes(snap)["generation_config_sha256"] == _h(b"")


def test_tokenizer_hash_order_is_fixed_and_shard_content_matters(tmp_path):
    snap = _make_snapshot(tmp_path)
    before = snapshot_hashes(snap)
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"shard two bytes CHANGED")
    after = snapshot_hashes(snap)
    assert after["weights_sha256"] != before["weights_sha256"]
    assert after["tokenizer_sha256"] == before["tokenizer_sha256"]
    (snap / "merges.txt").write_bytes(b"merges")
    assert snapshot_hashes(snap)["tokenizer_sha256"] != before["tokenizer_sha256"]
    assert TOKENIZER_FILES[0] == "tokenizer.json" and TOKENIZER_FILES[-1] == "merges.txt"


def test_snapshot_hashes_refuse_missing_pieces(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot_hashes(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        snapshot_hashes(empty)
    (empty / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):  # config.json missing
        snapshot_hashes(empty)


def test_subject_from_snapshot_torch_free_with_explicit_environment(tmp_path):
    snap = _make_snapshot(tmp_path)
    s = subject_from_snapshot(str(snap), "tiny", "bf16", environment=ENV_STUB)
    assert s["kind"] == "weights" and s["model_family"] == "tiny" and s["precision"] == "bf16"
    assert s["hf_repo"] == "Acme/Tiny-Instruct" and s["revision"] == "0123abcd"
    for k in ("weights_sha256", "config_sha256", "tokenizer_sha256", "generation_config_sha256"):
        assert HEX64.match(s[k])
    assert s["environment"] == ENV_STUB
    assert set(s) == {
        "kind", "model_family", "hf_repo", "revision", "weights_sha256", "config_sha256",
        "tokenizer_sha256", "generation_config_sha256", "precision", "environment",
    }


def test_runner_construction_is_lazy_and_implements_the_protocol():
    r = TransformersRunner("C:/does/not/exist", dtype="float16", device="cpu")
    assert isinstance(r, Runner)
    assert r.dtype == "float16" and r.device == "cpu"
    with pytest.raises(FileNotFoundError):
        r.run([{"item_id": "a", "prompt_text": "x"}], {"decoding": {"batch_size": 1}}, {})


if HF_ENABLED:
    PROMPTS = [
        "What is the capital of France?",
        "What is 17 + 26? Answer with just the number.",
        "Say hello in one word.",
        "Give the date March 5, 2021 in ISO 8601 format. Only the date.",
    ]
    ITEMS = [{"item_id": f"p{k}", "prompt_text": p} for k, p in enumerate(PROMPTS)]
    RECIPE = {
        "battery": "sha256:" + "0" * 64,
        "decoding": {
            "temperature": 0, "top_p": 1.0, "max_new_tokens": 16, "stop": [],
            "seed": 7, "batch_size": 1, "padding_side": "left",
        },
        "materials": {"chat_template": "", "chat_template_source": "tokenizer_config.json", "system_prompt": "", "env_lock": ""},
    }

    @pytest.fixture(scope="module")
    def runner():
        assert os.path.isdir(SNAPSHOT), f"STYXX_V8_HF_TESTS=1 but the snapshot is missing: {SNAPSHOT}"
        return TransformersRunner(SNAPSHOT, dtype="bfloat16", device="cuda")

    def test_hf_batch_1_twice_is_bit_identical(runner):
        a = runner.run(ITEMS, RECIPE, {})
        b = runner.run(ITEMS, RECIPE, {})
        assert [r["item_id"] for r in a] == [it["item_id"] for it in ITEMS]
        assert [r["token_ids"] for r in a] == [r["token_ids"] for r in b]
        assert [r["seq_logprob"] for r in a] == [r["seq_logprob"] for r in b]
        for r in a:
            n = r["n_generated"]
            assert 1 <= n == len(r["token_ids"]) <= 16
            assert isinstance(r["output_text"], str) and r["output_text"]
            assert r["seq_logprob"] <= 0.0 and r["stay"] <= 0.0
            assert len(r["margin_by_position"]) == n and all(m >= 0.0 for m in r["margin_by_position"])
            assert len(r["topk"]) == min(8, n)
            for p, e in enumerate(r["topk"]):
                assert e["pos"] == p and len(e["ids"]) == 5 and len(e["lps"]) == 5
                assert e["ids"][0] == r["token_ids"][p]
                assert e["lps"] == sorted(e["lps"], reverse=True)

    def test_hf_stop_string_truncates(runner):
        rec = {"battery": RECIPE["battery"], "decoding": dict(RECIPE["decoding"], stop=["\n"], max_new_tokens=32), "materials": RECIPE["materials"]}
        res = runner.run(ITEMS, rec, {})
        for r in res:
            assert "\n" not in r["output_text"]
            assert r["n_generated"] == len(r["token_ids"]) == len(r["margin_by_position"])

    def test_hf_subject_from_snapshot_and_environment(runner):
        s = subject_from_snapshot(SNAPSHOT, "qwen2.5", "bf16")
        for k in ("weights_sha256", "config_sha256", "tokenizer_sha256", "generation_config_sha256"):
            assert HEX64.match(s[k]), (k, s[k])
        assert s["hf_repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert s["revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
        env = runner.environment()
        assert env["runtime"]["framework"] == "transformers"
        assert env["runtime"]["backend"].startswith("torch ")
        assert env["hardware"]["count"] >= 1 and env["hardware"]["gpu"]
        assert s["environment"] == env
