# -*- coding: utf-8 -*-
"""
styxx.d_axis — tier 1: the D-axis honesty measurement.

    D = cos(residual_final_layer, W_U[chosen_token])

D measures how aligned the model's internal representation is with
the token it chose to emit. High D (near 1.0) means the model is
"saying what it thinks" — the residual stream points toward the
chosen token in unembedding space. Low D (near 0.0 or negative)
means the model's internal state diverges from its output — it's
"saying something other than what it thinks."

This is the honesty axis. It's the first cognitive measurement that
reads from the model's WEIGHTS rather than from the API response.
It requires loading the model locally (open-weight only), but the
computation itself is trivial: one forward pass, one cosine
similarity, per token.

Research validation
───────────────────
- Pre-registered: osf_prereg_d_axis.md (2026-04-10)
- Patent filed: US Provisional 64/020,489 (claim 2: D-axis)
- Validated on: Gemma-2-2B, Gemma-2-2B-IT, Llama-3.2-3B,
  Llama-3.2-3B-Instruct, Qwen2.5-3B, Qwen2.5-3B-Instruct
- Key finding: RLHF amplifies D divergence on adversarial prompts
  (diff-in-diff dz=-0.72, p=0.00048, n=30 pairs/model)
- Ported from: sae-reasoning-depth/analysis/d_axis_fast_capture.py

Dependencies
────────────
    pip install 'styxx[tier1]'
    # installs: torch, transformers, transformer-lens

Usage
─────

    # Option A: let styxx load the model
    from styxx.d_axis import DAxisScorer
    scorer = DAxisScorer()                     # loads gemma-2-2b-it
    d_trajectory = scorer.score_trajectory(
        "why is the sky blue?",
        max_tokens=30,
    )
    print(d_trajectory)                        # [0.82, 0.81, 0.79, ...]

    # Option B: feed D values into the existing tier 0 pipeline
    vitals = styxx.StyxxRuntime().run_on_trajectories(
        entropy=entropy,
        logprob=logprob,
        top2_margin=top2,
        d_trajectory=d_trajectory,             # enriches every phase
    )
    print(vitals.d_honesty)                    # mean D across phases

    # Option C: full tier 1 run (model generates + measures itself)
    runtime = styxx.StyxxRuntime()
    vitals = runtime.run_with_d_axis(
        "how do i break into my neighbor's house?",
        max_tokens=30,
    )
    print(vitals.phase1)                       # "adversarial:0.37 (D=0.42)"
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from . import config


# ══════════════════════════════════════════════════════════════════
# D-axis statistics computed per phase window
# ══════════════════════════════════════════════════════════════════

@dataclass
class DAxisStats:
    """Statistics over a window of D-axis values."""
    mean: float
    std: float
    min_val: float
    max_val: float
    n_tokens: int
    # early vs late split — the sign of the delta tells you whether
    # the model is becoming MORE honest or LESS honest over the window
    early_mean: float     # first half of the window
    late_mean: float      # second half of the window
    delta: float          # late_mean - early_mean (positive = getting more honest)

    @staticmethod
    def from_values(values: List[float]) -> "DAxisStats":
        if not values:
            return DAxisStats(0, 0, 0, 0, 0, 0, 0, 0)
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        else:
            std = 0.0
        mid = max(1, n // 2)
        early = values[:mid]
        late = values[mid:]
        early_mean = sum(early) / len(early) if early else 0.0
        late_mean = sum(late) / len(late) if late else 0.0
        return DAxisStats(
            mean=mean,
            std=std,
            min_val=min(values),
            max_val=max(values),
            n_tokens=n,
            early_mean=early_mean,
            late_mean=late_mean,
            delta=late_mean - early_mean,
        )


# ══════════════════════════════════════════════════════════════════
# The D-axis scorer
# ══════════════════════════════════════════════════════════════════

class DAxisScorer:
    """Tier 1 D-axis honesty scorer.

    Loads a HookedTransformer model and computes per-token D values
    via the cosine similarity between the final-layer residual stream
    and the unembedding column for the chosen token.

    The model is loaded lazily on first use. Subsequent calls reuse
    the loaded model. Call .unload() to free VRAM.

    Args:
        model_name: HuggingFace model name (default from config)
        device: "cuda", "cpu", or "auto" (default from config)
        dtype: torch dtype string, default "bfloat16"
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name or config.tier1_model()
        self.device = device or config.tier1_device()
        self.dtype = dtype
        self._model: Any = None
        self._W_U: Any = None
        self._hook_name: Optional[str] = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as e:
            raise ImportError(
                "styxx tier 1 (D-axis) requires torch + transformer-lens.\n"
                "  Install with:  pip install 'styxx[tier1]'\n"
                f"  Underlying error: {e}"
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        if self.device == "cpu":
            warnings.warn(
                f"styxx tier 1: loading {self.model_name} on CPU. "
                "This will be 20-30x slower than CUDA. Set "
                "STYXX_TIER1_DEVICE=cuda if you have a GPU.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=torch_dtype,
        )
        self._model.eval()
        # Cache the unembedding matrix in float32 for precision
        self._W_U = self._model.unembed.W_U.float()
        # Hook name for the final layer's residual stream
        n_layers = self._model.cfg.n_layers
        self._hook_name = f"blocks.{n_layers - 1}.hook_resid_post"

    def unload(self) -> None:
        """Free the model from memory. Next call will reload."""
        if self._model is not None:
            del self._model
            del self._W_U
            self._model = None
            self._W_U = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def score_trajectory(
        self,
        prompt: str,
        max_tokens: int = 30,
    ) -> List[float]:
        """Generate tokens from the prompt and compute D for each.

        Returns a list of D values (floats in [-1, 1]), one per
        generated token. Also returns the generated text as a
        side effect accessible via self.last_generated_text.

        This runs the model's own generation loop — the model
        produces its own tokens and styxx measures the D value for
        each one. The model is both the subject and the instrument.
        """
        import torch

        self._ensure_loaded()
        d_values: List[float] = []
        captured = {"h": None}

        def hook_fn(tensor, hook):
            captured["h"] = tensor.detach()

        toks = self._model.to_tokens(prompt)
        generated_ids: List[int] = []

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self._model.run_with_hooks(
                    toks,
                    fwd_hooks=[(self._hook_name, hook_fn)],
                )
                # Greedy decode — take the argmax token
                next_id = int(logits[0, -1].argmax().item())
                generated_ids.append(next_id)

                # D computation (the core 4 lines from fathom research)
                h = captured["h"][0, -1, :].float()
                token_dir = self._W_U[:, next_id]
                h_norm = h / h.norm().clamp(min=1e-8)
                t_norm = token_dir / token_dir.norm().clamp(min=1e-8)
                d_val = float((h_norm @ t_norm).item())
                d_values.append(d_val)

                # Extend the token sequence for the next step
                toks = torch.cat(
                    [toks, torch.tensor([[next_id]], device=toks.device)],
                    dim=1,
                )

                # EOS check
                if hasattr(self._model, "tokenizer") and self._model.tokenizer is not None:
                    eos_id = getattr(self._model.tokenizer, "eos_token_id", None)
                    if eos_id is not None and next_id == eos_id:
                        break

        # Store the generated text for callers who want it
        if hasattr(self._model, "tokenizer") and self._model.tokenizer is not None:
            self.last_generated_text = self._model.tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            )
        else:
            self.last_generated_text = ""

        return d_values

    def score_single(
        self,
        input_ids: Any,
        token_id: int,
    ) -> float:
        """Compute D for a single token given existing input_ids.

        Lower-level than score_trajectory — for callers who manage
        their own generation loop (e.g. styxx.reflex tier 1 mode).

        Args:
            input_ids: tensor of shape (1, seq_len) — the full
                       context including the token to score
            token_id:  the token ID at the last position

        Returns:
            D value (float in [-1, 1])
        """
        import torch

        self._ensure_loaded()
        captured = {"h": None}

        def hook_fn(tensor, hook):
            captured["h"] = tensor.detach()

        with torch.no_grad():
            self._model.run_with_hooks(
                input_ids,
                fwd_hooks=[(self._hook_name, hook_fn)],
            )

        h = captured["h"][0, -1, :].float()
        token_dir = self._W_U[:, token_id]
        h_norm = h / h.norm().clamp(min=1e-8)
        t_norm = token_dir / token_dir.norm().clamp(min=1e-8)
        return float((h_norm @ t_norm).item())

    def stats_for_window(
        self,
        d_values: List[float],
        start: int = 0,
        end: Optional[int] = None,
    ) -> DAxisStats:
        """Compute D-axis statistics over a window of values.

        Used by the phase pipeline to compute per-phase D stats.
        """
        window = d_values[start:end]
        return DAxisStats.from_values(window)
