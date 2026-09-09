"""transformers-backed ``Runner`` for local weights (spec section 3, Appendix A.2).

torch and transformers are imported lazily inside functions; ``import styxx.v8.runner_hf``
touches neither.  The model is loaded from a local snapshot directory with
``HF_HUB_OFFLINE=1`` (never a hub id) -- the loading pattern that works on this box is
``papers/v8/probe_batch_invariance_2026_09_08/probe_batch_invariance.py``.

What the runner records per item (all from the model's OWN greedy prefix):

* ``token_ids``      greedy ids up to and excluding the earliest eos/pad token, then truncated at
                     the earliest stop string: the shortest prefix whose decoded text contains a
                     stop string is cut before the token that completed it (token-granular; a
                     token that carries both content and the stop is dropped with the stop).
* ``seq_logprob``    sum over kept positions of log_softmax(scores_t)[id_t].
* ``topk``           top-5 ids/log-probs at positions 0..min(7, n_generated-1).  Section 3.2
                     asks for teacher-forcing on the REFERENCE prefix for weights subjects; that
                     is a later item -- this runner scores the model's own prefix, and a
                     verifier comparing two runs whose prefixes diverge is comparing different
                     contexts from the point of divergence on.
* ``margin_by_position``  top-1 minus top-2 log-prob at every kept position (Appendix C margin).
* ``stay``           sum over kept positions of log_softmax(scores_t / 0.2)[id_t] (Appendix C).

Scores are raw logits: the checkpoint's generation_config is neutralised on the call
(repetition_penalty=1.0, temperature/top_p/top_k=None) exactly as the probe does.

``TransformersRunner.subject(requested)`` reports what this process will actually run: the four
Appendix A.2 hashes over the snapshot's own bytes, ``hf_repo``/``revision`` from the snapshot
path, and ``precision`` from the dtype -- read off the loaded model when one is loaded, from
``self.dtype`` before that.  Nothing is copied from ``requested`` except the SPELLING of a
precision that denotes the dtype actually in use (``bfloat16`` and ``bf16`` are one dtype and a
cert may use either), so ``--dtype float16`` against a bf16 cert reports ``fp16`` and
``verify --ref`` refuses.  Before this method existed the runner had no ``subject`` member at
all, ``verify --ref`` fell back to the cert's own subject, and the guard could not fail: the
defect recorded as C2 in ``papers/v8/challenge_and_attack_2026_09_09``.

The A.2 hashes are computed once per runner and cached: a snapshot swapped mid-run is not
caught by this member.  The hashes are of the files this runner loads from, which is the claim
being made; whether the process re-read them is a different claim and is not made here.

``TransformersRunner.environment()`` reports the device THIS RUNNER USES
-----------------------------------------------------------------------
``probe_environment(device)`` builds the section 2.2 environment for a run on ``device``, and the
runner passes its own ``self.device``.  Before this argument existed the function asked the box
what cards it had -- ``torch.cuda.get_device_name(0)`` whenever cuda was available -- so a
``--device cpu`` run reported the RTX 4070 sitting idle beside it, ``device`` was in no signed
field at all, and ``--environment <file>`` overwrote even that.  One CPU forward pass through the
shipped CLI produced a signed result naming a card it never touched (S-DEVICE).  ``device`` is
not and does not become an S_identity field -- section 2.2 says the environment is never identity
and section 5.4 spends it on coverage -- so the repair is not a new identity check; it is that
the field is now OBSERVED, and a difference lands where section 5.4 puts it.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from styxx.v8.runner import ItemResult, _batch_size, _check_items, _max_new_tokens

__all__ = [
    "TransformersRunner",
    "hardware_block",
    "precision_label",
    "run_device",
    "subject_from_snapshot",
    "probe_environment",
    "snapshot_hashes",
]

# Appendix A.2: the tokenizer files, in this fixed order, that are hashed when present.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)

_STAY_TEMPERATURE = 0.2
_TOPK_K = 5
_TOPK_POSITIONS = 8

# dtype spelling -> the section 2.2 ``precision`` label.  Both columns are dtype NAMES: this
# table renames, it never converts.  A dtype not listed (a quantised load) keeps its own name,
# which is still an honest report of what was loaded.
_PRECISION_LABELS = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "fp16": "fp16", "half": "fp16",
    "float32": "fp32", "fp32": "fp32", "float": "fp32",
    "float64": "fp64", "fp64": "fp64", "double": "fp64",
}


def precision_label(dtype: str) -> str:
    """The ``subject.precision`` label for a dtype spelling (``"bfloat16"`` -> ``"bf16"``)."""
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name[len("torch."):]
    return _PRECISION_LABELS.get(name, name)


_UNKNOWN_ENV = {
    "runtime": {"framework": "transformers", "version": "unavailable", "backend": "unavailable"},
    "hardware": {"gpu": "unavailable", "driver": "unavailable", "count": 0},
}


def run_device(device: Any, cuda_available: bool) -> str:
    """The device string a run takes: what was asked for, or what this process would pick.

    ``None`` means nothing was asked for, and then the answer is what ``model.to(...)`` would
    have got: ``"cuda"`` when a card is available, ``"cpu"`` when none is.  A ``torch.device``
    or an index-qualified string (``"cuda:1"``) is kept as it is written, lowercased.
    """
    if device is None:
        return "cuda" if cuda_available else "cpu"
    text = str(device).strip().lower()
    return text or ("cuda" if cuda_available else "cpu")


def hardware_block(device: str, gpu_name: str | None = None, driver: str | None = None) -> dict:
    """The ``environment.hardware`` block for a run on ``device`` (section 2.2).

    THE RULE, stated once so a reader can re-derive it: the block describes the device the run
    COMPUTES ON, never the devices the box holds.

    * ``gpu_name`` is the card this process observed for ``device``, or ``None`` when ``device``
      is not a card (``"cpu"``, ``"mps"``) or is one this process cannot see.  With ``None`` the
      block names the device itself, the driver is ``"none"`` and the count is 0 -- a CPU run
      beside an idle RTX 4070 says ``cpu``, which is the whole of the S-DEVICE repair.
    * ``count`` is the number of devices THIS RUN uses, which is 1 for a card and 0 otherwise.
      It was previously ``torch.cuda.device_count()``, a fact about the box: a CPU run on this
      box reported 1, and a count of the cards a run did not touch is not a fact about the run.
    * ``device`` is carried beside them because ``gpu`` alone cannot separate ``cuda:0`` from
      ``cuda:1``, and because ``device`` is the field the attack turned on.
    """
    text = str(device)
    if gpu_name is None:
        return {"gpu": text, "driver": "none", "count": 0, "device": text}
    return {"gpu": str(gpu_name), "driver": str(driver or "unknown"), "count": 1, "device": text}


def _set_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _nvidia_driver() -> str:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if p.returncode != 0:
        return "unknown"
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else "unknown"


def probe_environment(device: Any = None) -> dict:
    """``subject.environment`` for a run on ``device``; never raises.

    ``device`` is the torch device the run computes on -- a ``TransformersRunner``'s own
    ``self.device``.  ``None`` means "whatever this process would pick" (``run_device``), which
    is what a caller minting a subject without a runner in hand can honestly say.

    The hardware block is ``hardware_block``: it names the device the run USES.  Before this
    argument existed the function reported ``torch.cuda.get_device_name(0)`` whenever a card was
    visible, so ``--device cpu`` produced an environment naming an RTX 4070 the run never
    touched, and ``verify --ref`` signed it (S-DEVICE).  A cuda device this process cannot see is
    reported ``unavailable`` rather than as a card: the run will fail, and until it does there is
    no card to name.
    """
    try:
        import torch
        import transformers
    except ImportError:
        env = {k: dict(v) for k, v in _UNKNOWN_ENV.items()}
        env["hardware"]["device"] = run_device(device, False) if device is not None else "unavailable"
        return env
    cuda = torch.cuda.is_available()
    want = run_device(device, cuda)
    runtime = {
        "framework": "transformers",
        "version": str(transformers.__version__),
        "backend": "torch " + str(torch.__version__),
    }
    if not want.split(":")[0] == "cuda":
        return {"runtime": runtime, "hardware": hardware_block(want)}
    if not cuda:
        hardware = hardware_block(want)
        hardware["gpu"] = "unavailable"
        hardware["driver"] = "unavailable"
        return {"runtime": runtime, "hardware": hardware}
    try:
        index = int(want.split(":", 1)[1]) if ":" in want else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
    except Exception:
        hardware = hardware_block(want)
        hardware["gpu"] = "unavailable"
        hardware["driver"] = "unavailable"
        return {"runtime": runtime, "hardware": hardware}
    return {"runtime": runtime, "hardware": hardware_block(want, name, _nvidia_driver())}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _join_hash(pairs: list[tuple[str, str]]) -> str:
    # A.2: sha256(join("\n", f"{filename} {digest}")) over pairs sorted by filename.
    text = "\n".join(f"{name} {digest}" for name, digest in sorted(pairs))
    return _sha256_bytes(text.encode("utf-8"))


def snapshot_hashes(snapshot_dir: str | os.PathLike) -> dict:
    """The four A.2 hashes for a local model directory, as bare 64-hex strings."""
    root = Path(snapshot_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {root}")
    shards = sorted(p for p in root.iterdir() if p.is_file() and p.name.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors shard under {root}")
    weights = _join_hash([(p.name, _sha256_file(p)) for p in shards])
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json missing under {root}")
    config_sha = _sha256_file(config)
    gen = root / "generation_config.json"
    gen_sha = _sha256_file(gen) if gen.is_file() else _sha256_bytes(b"")
    tok_pairs = [(n, _sha256_file(root / n)) for n in TOKENIZER_FILES if (root / n).is_file()]
    # A.2 says "in fixed order"; the join helper sorts by filename, which is the same fixed
    # order for every caller, so the order the files were listed in cannot leak into the hash.
    tokenizer = _join_hash(tok_pairs)
    return {
        "weights_sha256": weights,
        "config_sha256": config_sha,
        "generation_config_sha256": gen_sha,
        "tokenizer_sha256": tokenizer,
    }


def _repo_and_revision(snapshot_dir: Path) -> tuple[str, str]:
    """``models--Org--Name/snapshots/<rev>`` -> (``Org/Name``, ``<rev>``); else (dir name, dir name)."""
    revision = snapshot_dir.name
    for parent in snapshot_dir.parents:
        if parent.name.startswith("models--"):
            return parent.name[len("models--"):].replace("--", "/"), revision
    return snapshot_dir.name, revision


def subject_from_snapshot(
    snapshot_dir: str,
    model_family: str,
    precision: str,
    environment: dict | None = None,
) -> dict:
    """A ``kind: weights`` subject (section 2.2) for a local snapshot.  ``environment`` defaults
    to ``probe_environment()`` (which imports torch lazily); pass one to stay torch-free."""
    root = Path(snapshot_dir).resolve()
    hashes = snapshot_hashes(root)
    hf_repo, revision = _repo_and_revision(root)
    return {
        "kind": "weights",
        "model_family": model_family,
        "hf_repo": hf_repo,
        "revision": revision,
        **hashes,
        "precision": precision,
        "environment": environment if environment is not None else probe_environment(),
    }


class TransformersRunner:
    """Greedy decoding through transformers on a local snapshot.  Implements ``Runner``."""

    def __init__(self, snapshot_dir: str, dtype: str = "bfloat16", device: str = "cuda") -> None:
        self.snapshot_dir = str(snapshot_dir)
        self.dtype = dtype
        self.device = device
        self._model: Any = None
        self._tok: Any = None
        self._eos_ids: set[int] = set()
        self._pad_id: int | None = None
        self._hashes: dict | None = None

    # -- loading ---------------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        _set_offline()
        if not Path(self.snapshot_dir).is_dir():
            raise FileNotFoundError(f"snapshot directory not found: {self.snapshot_dir}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            torch_dtype = getattr(torch, self.dtype)
        except AttributeError as e:
            raise ValueError(f"unknown torch dtype {self.dtype!r}") from e
        tok = AutoTokenizer.from_pretrained(self.snapshot_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(self.snapshot_dir, dtype=torch_dtype)
        except TypeError:  # transformers before the ``dtype`` keyword (the probe used torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(self.snapshot_dir, torch_dtype=torch_dtype)
        model.to(self.device)
        model.eval()
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tok.eos_token_id
        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self._pad_id = tok.pad_token_id
        self._tok = tok
        self._model = model

    def environment(self) -> dict:
        """The environment of the run this runner makes -- its OWN device, not the box's cards.

        ``self.device`` is what ``model.to(...)`` was given, so it is the one thing in this
        process that knows whether the forward pass touched a card.  See ``probe_environment``.
        """
        return probe_environment(self.device)

    # -- what this runner actually runs ------------------------------------------------
    def loaded_precision(self) -> str:
        """The precision label of the dtype in use: the model's own when it is loaded."""
        dtype = self.dtype
        if self._model is not None:
            actual = getattr(self._model, "dtype", None)
            if actual is not None:
                dtype = str(actual)
        return precision_label(dtype)

    def subject(self, requested: Mapping[str, Any] | None = None) -> dict:
        """The ``kind`` + S_identity of the model this runner runs (see the module docstring).

        Every field is observed here -- the A.2 hashes from the snapshot's bytes, the repo and
        revision from its path, the precision from the dtype.  The only thing taken from
        ``requested`` is the spelling of a precision that means the dtype in use.
        """
        root = Path(self.snapshot_dir).resolve()
        if self._hashes is None:
            self._hashes = snapshot_hashes(root)
        hf_repo, revision = _repo_and_revision(root)
        loaded = self.loaded_precision()
        asked = requested.get("precision") if isinstance(requested, Mapping) else None
        precision = asked if isinstance(asked, str) and precision_label(asked) == loaded else loaded
        return {
            "kind": "weights",
            "hf_repo": hf_repo,
            "revision": revision,
            **dict(self._hashes),
            "precision": precision,
        }

    # -- prompts ---------------------------------------------------------------------
    def _chat_text(self, prompt: str, system_prompt: str, chat_template: str | None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if chat_template:
            kwargs["chat_template"] = chat_template
        return self._tok.apply_chat_template(messages, **kwargs)

    def _truncate_at_stop(self, ids: list[int], stops: list[str]) -> list[int]:
        if not stops:
            return ids
        for k in range(1, len(ids) + 1):
            text = self._tok.decode(ids[:k], skip_special_tokens=False)
            if any(s and s in text for s in stops):
                return ids[: k - 1]
        return ids

    # -- generation ------------------------------------------------------------------
    def run(self, items: list[dict], recipe: dict, subject: dict) -> list[ItemResult]:
        checked = _check_items(items)
        batch_size = _batch_size(recipe)
        max_new = _max_new_tokens(recipe)
        decoding: Mapping[str, Any] = recipe.get("decoding") or {}
        temperature = decoding.get("temperature", 0)
        if temperature not in (0, 0.0):
            raise ValueError("TransformersRunner decodes greedily; decoding.temperature must be 0")
        stops = [s for s in (decoding.get("stop") or []) if isinstance(s, str)]
        seed = decoding.get("seed", 0)
        padding_side = decoding.get("padding_side", "left")
        if padding_side not in ("left", "right"):
            raise ValueError(f"decoding.padding_side must be 'left' or 'right', got {padding_side!r}")
        materials: Mapping[str, Any] = recipe.get("materials") or {}
        system_prompt = materials.get("system_prompt") or ""
        chat_template = materials.get("chat_template") or None

        self._load()
        import torch

        tok, model = self._tok, self._model
        tok.padding_side = padding_side
        texts = [self._chat_text(it["prompt_text"], system_prompt, chat_template) for it in checked]
        results: list[ItemResult] = []
        for start in range(0, len(checked), batch_size):
            chunk = checked[start : start + batch_size]
            enc = tok(texts[start : start + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            torch.manual_seed(int(seed) if seed is not None else 0)
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=False,
                    max_new_tokens=max_new,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self._pad_id,
                    repetition_penalty=1.0,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            in_len = enc["input_ids"].shape[1]
            gen = out.sequences[:, in_len:]
            steps = len(out.scores)
            for r, it in enumerate(chunk):
                raw = gen[r].tolist()
                kept: list[int] = []
                for t in range(min(steps, len(raw))):
                    tid = raw[t]
                    if tid in self._eos_ids or tid == self._pad_id:
                        break
                    kept.append(tid)
                kept = self._truncate_at_stop(kept, stops)
                results.append(self._score(it["item_id"], kept, out.scores, r, torch))
        return results

    def _score(self, item_id: str, ids: list[int], scores: Any, row: int, torch: Any) -> ItemResult:
        import math

        seq_terms: list[float] = []
        stay_terms: list[float] = []
        margins: list[float] = []
        topk: list[dict] = []
        for t, tid in enumerate(ids):
            z = scores[t][row].float()
            lp = torch.log_softmax(z, dim=-1)
            seq_terms.append(lp[tid].item())
            stay_terms.append(torch.log_softmax(z / _STAY_TEMPERATURE, dim=-1)[tid].item())
            top = torch.topk(lp, _TOPK_K)
            vals = top.values.tolist()
            margins.append(vals[0] - vals[1])
            if t < _TOPK_POSITIONS:
                topk.append({"pos": t, "ids": top.indices.tolist(), "lps": vals})
        text = self._tok.decode(ids, skip_special_tokens=False)
        return ItemResult(
            item_id=item_id,
            token_ids=ids,
            output_text=text,
            n_generated=len(ids),
            seq_logprob=math.fsum(seq_terms),
            topk=topk,
            margin_by_position=margins,
            stay=math.fsum(stay_terms),
        )
