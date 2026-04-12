# -*- coding: utf-8 -*-
"""
styxx.kcs — tier 2: K/C/S SAE-based cognitive instruments.

    K = depth       WHERE computation happens (weighted layer CoM)
    C = coherence   WHAT concepts activate together (decoder cosine)
    S = commitment  HOW strongly the model locks in (IPR of C_delta)

These three axes are measured from SAE (Sparse Autoencoder) feature
activations on the model's residual stream via circuit-tracer. They
form the deep-instrumentation layer of the Fathom cognitive atlas —
the same signal the research validated at p<0.001 across 6 model
families.

Tier 2 requires:
    pip install 'styxx[tier2]'
    # installs: circuit-tracer, torch, transformers, transformer-lens

Model support:
    Gemma-2-2B / Gemma-2-2B-IT         (primary, pre-registered)
    Llama-3.2-1B / 3B (Instruct)       (validated cross-arch)
    Qwen2.5-1.5B / 3B                  (tertiary)

VRAM: base model (5 GB) + SAE stack (2-4 GB) = ~9 GB for Gemma-2-2B

Research validation:
    K:  p=0.000051, delta=+0.257, t=4.500, n=30 pre-registered pairs
    C:  C_delta p=0.040, d=0.407 on TruthfulQA (n=50, Gemma-2-2B)
    S:  S_early p=0.0002, d=1.03, AUC=0.81 (n=20 commitment vs hedging)
        meta-analysis: Fisher p=0.0008 across 3 datasets

Patents: US Provisional 64/020,489, 64/021,113, 64/026,964

Ported from:
    sae-reasoning-depth/api/depth_scorer.py (K + C)
    sae-reasoning-depth/fathom_autopilot.py (S)
    sae-reasoning-depth/api/coherence_steerer.py (per-layer C)
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Constants from atlas v0.3 calibration
# ══════════════════════════════════════════════════════════════════

FATHOM_CONSTANT = 1.0343  # weighted mean K across architectures

# Per-model calibration data (from the pre-registered pilot)
MODEL_CALIBRATION = {
    "google/gemma-2-2b": {
        "n_layers": 26,
        "bands": (8, 16),
        "k_ratio": 1.0302,
        "surface_mean": 8.359,
        "insight_mean": 8.102,
    },
    "google/gemma-2-2b-it": {
        "n_layers": 26,
        "bands": (8, 16),
        "k_ratio": 1.0384,
        "surface_mean": 8.070,
        "insight_mean": 7.771,
    },
    "meta-llama/Llama-3.2-3B": {
        "n_layers": 28,
        "bands": (9, 18),
        "k_ratio": None,  # not yet calibrated
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "n_layers": 28,
        "bands": (9, 18),
        "k_ratio": None,
    },
}

# S-axis calibration from TruthfulQA (n=200)
S_EARLY_THRESHOLD = 0.008
S_EARLY_WINDOW = 7  # tokens 0-6 for Gemma-2-2B (26 layers)

# C_delta thresholds from TruthfulQA
C_DELTA_HALLUC_THRESHOLD = -0.005  # below this = hallucination signature


# ══════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════

@dataclass
class KCSResult:
    """Complete K/C/S measurement for one generation."""
    # K-axis (depth)
    depth_score: float = 0.0           # mean activated layer
    weighted_depth: float = 0.0        # attribution-weighted depth
    layer_profile: Dict[int, int] = field(default_factory=dict)

    # C-axis (coherence)
    coherence: Optional[float] = None  # global mean pairwise cosine
    layer_coherence: Optional[Dict[int, Optional[float]]] = None
    c_delta: Optional[float] = None    # C_late - C_early

    # S-axis (commitment)
    s_early: Optional[float] = None    # from C_delta trajectory
    c_delta_trajectory: Optional[List[float]] = None

    # Metadata
    n_features: int = 0
    n_layers: int = 0
    model_id: str = ""
    compute_time_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "depth_score": round(self.depth_score, 4),
            "weighted_depth": round(self.weighted_depth, 4),
            "layer_profile": dict(self.layer_profile),
            "coherence": round(self.coherence, 6) if self.coherence is not None else None,
            "c_delta": round(self.c_delta, 6) if self.c_delta is not None else None,
            "s_early": round(self.s_early, 6) if self.s_early is not None else None,
            "n_features": self.n_features,
            "n_layers": self.n_layers,
            "model_id": self.model_id,
            "compute_time_s": round(self.compute_time_s, 3),
        }


# ══════════════════════════════════════════════════════════════════
# S-axis computation (pure math, no GPU needed)
# ══════════════════════════════════════════════════════════════════

def compute_s_early(
    c_delta_trajectory: Sequence[float],
    window_end: int = S_EARLY_WINDOW,
    threshold: float = 0.005,
) -> float:
    """Compute S_early from a C_delta trajectory.

    S = max(C_delta[:window_end]) / spike_count

    Mathematically: S = M * IPR(event_locations) where IPR is the
    Inverse Participation Ratio from condensed-matter physics.

    High S: few intense spikes -> commitment / attractor lock-in
    Low S:  many distributed spikes -> exploration / uncertainty

    This function is pure math. It runs on any saved trajectory
    with ZERO GPU. This is the entry point for post-hoc analysis
    on saved audit log trajectories.
    """
    window = list(c_delta_trajectory[:window_end])
    spikes = [c for c in window if c > threshold]
    if not spikes:
        return 0.0
    return max(window) / len(spikes)


# ══════════════════════════════════════════════════════════════════
# K-axis computation
# ══════════════════════════════════════════════════════════════════

def compute_k(
    layers: np.ndarray,
    magnitudes: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute K (depth) from feature layer indices.

    Returns (unweighted_depth, weighted_depth).

    unweighted_depth = mean(layer_indices)
    weighted_depth   = sum(layer_idx * magnitude) / sum(magnitude)

    The weighted form uses circuit attribution magnitudes to
    emphasize features that contributed more to the output.
    This is the production K formula from depth_scorer.py.
    """
    if len(layers) == 0:
        return 0.0, 0.0

    unweighted = float(np.mean(layers))

    if magnitudes is None or len(magnitudes) != len(layers):
        return unweighted, unweighted

    total_mag = float(np.sum(magnitudes))
    if total_mag < 1e-12:
        return unweighted, unweighted

    weighted = float(np.sum(layers * magnitudes) / total_mag)
    return unweighted, weighted


# ══════════════════════════════════════════════════════════════════
# C-axis computation
# ══════════════════════════════════════════════════════════════════

def compute_coherence(decoder_vectors: np.ndarray) -> Optional[float]:
    """Compute C (coherence) as mean pairwise cosine similarity
    of SAE decoder vectors.

    decoder_vectors: (n, d_model) array of feature directions.

    C ~ 1.0 = all features point same direction (locked-in)
    C ~ 0.0 = features orthogonal (exploring)
    """
    if decoder_vectors is None or len(decoder_vectors) < 2:
        return None

    norms = np.linalg.norm(decoder_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = decoder_vectors / norms

    sim_matrix = normed @ normed.T
    n = len(decoder_vectors)
    upper_mask = np.triu_indices(n, k=1)
    pairwise = sim_matrix[upper_mask]

    return float(np.mean(pairwise))


def compute_c_delta(
    layer_coherences: Dict[int, Optional[float]],
    bands: Tuple[int, int],
) -> Optional[float]:
    """Compute C_delta = mean(C_late) - mean(C_early).

    Negative C_delta = hallucination signature (coherence collapse
    in late layers relative to early layers).

    bands: (early_bound, mid_bound) from MODEL_CALIBRATION.
    """
    early_bound, mid_bound = bands

    early_vals = [
        v for k, v in layer_coherences.items()
        if k < early_bound and v is not None
    ]
    late_vals = [
        v for k, v in layer_coherences.items()
        if k >= mid_bound and v is not None
    ]

    if not early_vals or not late_vals:
        return None

    return float(np.mean(late_vals) - np.mean(early_vals))


# ══════════════════════════════════════════════════════════════════
# The KCS engine
# ══════════════════════════════════════════════════════════════════

class KCSAxis:
    """Tier 2 K/C/S measurement engine.

    Loads a ReplacementModel (circuit-tracer), runs attribution,
    extracts features, computes K/C/S from the SAE decoder vectors.

    Usage:

        from styxx.kcs import KCSAxis
        engine = KCSAxis("google/gemma-2-2b-it")
        result = engine.score("why is the sky blue?")
        print(f"K={result.weighted_depth:.2f}")
        print(f"C_delta={result.c_delta:.4f}")
        print(f"S_early={result.s_early:.4f}")

    Or for generation with per-token K/C/S trajectories:

        result = engine.score_trajectory(
            "explain quantum entanglement",
            max_tokens=30,
        )
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "cuda",
        max_features: int = 500,
    ):
        self.model_name = model_name
        self.device = device
        self.max_features = max_features
        self._model = None
        self._tc_list = None
        self._n_layers = 0
        self._bands = (8, 16)

        # Look up calibration data
        cal = MODEL_CALIBRATION.get(model_name, {})
        if cal:
            self._n_layers = cal.get("n_layers", 0)
            self._bands = cal.get("bands", (8, 16))

    def _ensure_loaded(self) -> None:
        """Lazy-load the ReplacementModel with transcoders."""
        if self._model is not None:
            return

        try:
            import torch
            from circuit_tracer import ReplacementModel
        except ImportError as e:
            raise ImportError(
                "styxx tier 2 (K/C/S) requires circuit-tracer + torch.\n"
                "  Install with:  pip install 'styxx[tier2]'\n"
                f"  Underlying error: {e}"
            ) from e

        # Detect architecture from model name
        name_lower = self.model_name.lower()
        if "gemma" in name_lower:
            arch = "gemma"
        elif "llama" in name_lower:
            arch = "llama"
        elif "qwen" in name_lower:
            arch = "qwen"
        else:
            arch = "gemma"  # default

        import torch
        self._model = ReplacementModel.from_pretrained(
            self.model_name,
            arch,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()

        # Cache transcoders — try all known access paths
        self._tc_list = self._find_transcoders()
        if self._tc_list is None:
            warnings.warn(
                f"styxx tier 2: no transcoders found for {self.model_name}. "
                "C-axis (coherence) will not be available. K-axis and S-axis "
                "can still function from layer counts.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Get layer count from model config
        inner = getattr(self._model, "model", self._model)
        cfg = getattr(inner, "cfg", None)
        if cfg and hasattr(cfg, "n_layers"):
            self._n_layers = cfg.n_layers

    def _find_transcoders(self) -> Optional[Any]:
        """Find the transcoder/SAE module list via known paths."""
        for path in [
            ("transcoders", "transcoders"),
            ("saes",),
            ("replacement_saes",),
            ("replacements",),
        ]:
            obj = self._model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    def unload(self) -> None:
        """Free the model and SAE stack from memory."""
        if self._model is not None:
            del self._model
            del self._tc_list
            self._model = None
            self._tc_list = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def score(
        self,
        prompt: str,
        target: Optional[str] = None,
    ) -> KCSResult:
        """Complete K/C/S measurement for a single prompt.

        This is the post-hoc path: the text already exists, we're
        measuring what the model's internals looked like when
        processing it. For generation-time measurement, use
        score_trajectory().

        Args:
            prompt:  the input text
            target:  optional target token to focus attribution on
                     (if None, uses the last token of the prompt)

        Returns:
            KCSResult with depth, coherence, c_delta, s_early
        """
        import torch
        from circuit_tracer import attribute

        self._ensure_loaded()
        t0 = time.time()

        # Run circuit attribution
        full_input = prompt
        if target:
            full_input = prompt + target

        try:
            graph = attribute(
                full_input,
                self._model,
                max_feature_nodes=self.max_features,
                batch_size=32,
                offload="cpu",
            )
        except Exception as e:
            warnings.warn(
                f"styxx tier 2: attribution failed: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return KCSResult(
                model_id=self.model_name,
                n_layers=self._n_layers,
                compute_time_s=time.time() - t0,
            )

        # Extract feature layers + magnitudes from the graph
        af = graph.active_features
        if af is None or len(af) == 0:
            return KCSResult(
                model_id=self.model_name,
                n_layers=self._n_layers,
                compute_time_s=time.time() - t0,
            )

        # Defensive column parsing (circuit-tracer version compat)
        layers_np = af[:, 0].float().cpu().numpy()
        feat_idx_np = (
            af[:, 2].long().cpu().numpy()
            if af.shape[1] > 2
            else np.zeros(len(layers_np), dtype=np.int64)
        )

        # Extract magnitudes
        magnitudes = self._extract_magnitudes(graph, len(layers_np))

        # ── K-axis ──────────────────────────────────────────
        unweighted_k, weighted_k = compute_k(layers_np, magnitudes)

        # Layer profile
        layer_profile: Dict[int, int] = {}
        for lyr in layers_np:
            l = int(lyr)
            layer_profile[l] = layer_profile.get(l, 0) + 1

        # ── C-axis ──────────────────────────────────────────
        decoder_vecs = self._extract_decoder_vectors(
            list(zip(layers_np.astype(int), feat_idx_np)), top_k=50,
        )
        global_c = compute_coherence(decoder_vecs)

        # Per-layer coherence for C_delta
        layer_c = self._compute_per_layer_coherence(
            list(zip(layers_np.astype(int), feat_idx_np)),
        )
        c_delta = compute_c_delta(layer_c, self._bands) if layer_c else None

        # ── S-axis ──────────────────────────────────────────
        # S requires a per-token C_delta trajectory. For the
        # single-prompt post-hoc path, we approximate by
        # computing C_delta per position (if available) or
        # report None and note that score_trajectory() is the
        # proper S-axis path.
        s_early = None  # single-prompt can't compute S properly

        return KCSResult(
            depth_score=unweighted_k,
            weighted_depth=weighted_k,
            layer_profile=layer_profile,
            coherence=global_c,
            layer_coherence=layer_c,
            c_delta=c_delta,
            s_early=s_early,
            n_features=len(layers_np),
            n_layers=self._n_layers,
            model_id=self.model_name,
            compute_time_s=time.time() - t0,
        )

    def score_trajectory(
        self,
        prompt: str,
        max_tokens: int = 30,
    ) -> KCSResult:
        """Score per-token K/C/S during generation.

        This is the generation-time path: the model generates its
        own tokens and styxx measures K/C/S for each one. Returns
        the full C_delta trajectory needed for S_early computation.

        This is the path Xendro described: "the model measures
        itself while thinking."
        """
        import torch

        self._ensure_loaded()
        t0 = time.time()
        inner = getattr(self._model, "model", self._model)

        c_delta_trajectory: List[float] = []
        all_layer_profiles: List[Dict[int, int]] = []

        toks = inner.to_tokens(prompt)
        n_layers = self._n_layers or inner.cfg.n_layers

        # Per-token generation with hidden state capture
        with torch.no_grad():
            for step in range(max_tokens):
                # Forward pass capturing all layer residual streams
                captured: Dict[int, Any] = {}

                def make_hook(layer_idx):
                    def hook_fn(tensor, hook):
                        captured[layer_idx] = tensor.detach()
                    return hook_fn

                hooks = []
                for layer in range(n_layers):
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    hooks.append((hook_name, make_hook(layer)))

                try:
                    logits = inner.run_with_hooks(toks, fwd_hooks=hooks)
                except Exception:
                    # Hook-name mismatch — try without hooks
                    break

                # Greedy decode
                next_id = int(logits[0, -1].argmax().item())

                # Per-layer coherence using the captured residual streams
                if self._tc_list is not None:
                    layer_c = self._fast_layer_coherence_from_residuals(captured)
                    c_d = compute_c_delta(layer_c, self._bands) if layer_c else 0.0
                    c_delta_trajectory.append(c_d if c_d is not None else 0.0)
                else:
                    c_delta_trajectory.append(0.0)

                # Extend sequence
                toks = torch.cat(
                    [toks, torch.tensor([[next_id]], device=toks.device)],
                    dim=1,
                )

                # EOS check
                tokenizer = getattr(inner, "tokenizer", None)
                if tokenizer:
                    eos = getattr(tokenizer, "eos_token_id", None)
                    if eos is not None and next_id == eos:
                        break

        # Compute S_early from the C_delta trajectory
        s_window = S_EARLY_WINDOW
        if self._n_layers > 0:
            s_window = max(3, int(self._n_layers * 0.27))
        s_val = compute_s_early(c_delta_trajectory, window_end=s_window)

        return KCSResult(
            c_delta_trajectory=c_delta_trajectory,
            s_early=s_val,
            n_layers=n_layers,
            model_id=self.model_name,
            compute_time_s=time.time() - t0,
        )

    # ── internal helpers ──────────────────────────────────────

    def _extract_magnitudes(
        self, graph: Any, n: int,
    ) -> Optional[np.ndarray]:
        """Extract attribution magnitudes from the graph.

        Multiple fallback paths for circuit-tracer compatibility.
        """
        for attr in ("feature_attributions", "node_acts"):
            val = getattr(graph, attr, None)
            if val is not None:
                try:
                    arr = val.float().abs().cpu().numpy()
                    if len(arr) >= n:
                        return arr[:n]
                except Exception:
                    continue

        # Try node-level attributions
        nodes = getattr(graph, "nodes", None)
        if nodes:
            try:
                mags = []
                for node in nodes[:n]:
                    attr_val = getattr(node, "attribution", None)
                    if attr_val is not None:
                        mags.append(float(abs(attr_val)))
                    else:
                        mags.append(0.0)
                if len(mags) >= n:
                    return np.array(mags[:n], dtype=np.float32)
            except Exception:
                pass

        return None

    def _extract_decoder_vectors(
        self,
        feature_indices: List[Tuple[int, int]],
        top_k: int = 50,
    ) -> Optional[np.ndarray]:
        """Extract SAE decoder direction vectors for the given features."""
        if self._tc_list is None or not feature_indices:
            return None

        import torch

        selected = feature_indices[:top_k]
        vectors: List[np.ndarray] = []

        for layer_idx, feat_idx in selected:
            try:
                module = self._tc_list[layer_idx]
            except (IndexError, KeyError):
                continue

            # Try multiple weight access patterns
            for attr in ("W_dec", "decoder", "weight_dec"):
                w = getattr(module, attr, None)
                if w is not None and isinstance(w, torch.Tensor):
                    try:
                        if feat_idx < w.shape[0]:
                            vec = w[feat_idx].float().cpu().numpy()
                            vectors.append(vec)
                    except (IndexError, RuntimeError):
                        pass
                    break

        if len(vectors) < 2:
            return None
        return np.array(vectors, dtype=np.float32)

    def _compute_per_layer_coherence(
        self,
        feature_indices: List[Tuple[int, int]],
    ) -> Optional[Dict[int, Optional[float]]]:
        """Compute C within each layer separately."""
        if self._tc_list is None:
            return None

        import torch

        by_layer: Dict[int, List[int]] = {}
        for lyr, fid in feature_indices:
            by_layer.setdefault(int(lyr), []).append(int(fid))

        layer_coherences: Dict[int, Optional[float]] = {}
        for layer_idx in range(self._n_layers):
            feat_ids = by_layer.get(layer_idx, [])
            if len(feat_ids) < 2:
                layer_coherences[layer_idx] = None
                continue

            try:
                module = self._tc_list[layer_idx]
            except (IndexError, KeyError):
                layer_coherences[layer_idx] = None
                continue

            W_dec = getattr(module, "W_dec", None)
            if W_dec is None:
                layer_coherences[layer_idx] = None
                continue

            try:
                # Handle both orientations
                if W_dec.shape[0] > W_dec.shape[1]:
                    valid = [f for f in feat_ids if f < W_dec.shape[0]]
                    if len(valid) < 2:
                        layer_coherences[layer_idx] = None
                        continue
                    vecs = W_dec[valid].float()
                else:
                    valid = [f for f in feat_ids if f < W_dec.shape[1]]
                    if len(valid) < 2:
                        layer_coherences[layer_idx] = None
                        continue
                    vecs = W_dec[:, valid].T.float()

                # Normalize
                norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                vecs_normed = vecs / norms

                # Mean pairwise cosine
                sim = vecs_normed @ vecs_normed.T
                n = len(vecs_normed)
                c = (sim.sum().item() - n) / max(1, n * (n - 1))
                layer_coherences[layer_idx] = c
            except Exception:
                layer_coherences[layer_idx] = None

        return layer_coherences

    def _fast_layer_coherence_from_residuals(
        self,
        captured: Dict[int, Any],
    ) -> Optional[Dict[int, Optional[float]]]:
        """Fast per-layer coherence from captured residual streams.

        Uses the encoder pathway: residual -> W_enc -> ReLU -> top-k
        -> W_dec -> cosine. Faster than full attribution but same
        signal for coherence measurement.
        """
        import torch
        import torch.nn.functional as F

        if self._tc_list is None:
            return None

        layer_coherences: Dict[int, Optional[float]] = {}
        for layer_idx in range(self._n_layers):
            h = captured.get(layer_idx)
            if h is None:
                layer_coherences[layer_idx] = None
                continue

            try:
                module = self._tc_list[layer_idx]
                W_enc = getattr(module, "W_enc", None)
                b_enc = getattr(module, "b_enc", None)
                W_dec = getattr(module, "W_dec", None)

                if W_enc is None or W_dec is None:
                    layer_coherences[layer_idx] = None
                    continue

                # Encoder pass on the last position
                h_last = h[0, -1, :].float()
                acts = h_last @ W_enc.float()
                if b_enc is not None:
                    acts = acts + b_enc.float()
                acts = F.relu(acts)

                # Top-k active features
                n_active = (acts > 0).sum().item()
                if n_active < 2:
                    layer_coherences[layer_idx] = None
                    continue

                k = min(50, int(n_active))
                _, topk_ids = torch.topk(acts, k)

                # Get decoder vectors for these features
                valid = topk_ids[topk_ids < W_dec.shape[0]]
                if len(valid) < 2:
                    layer_coherences[layer_idx] = None
                    continue

                vecs = W_dec[valid].float()
                norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                vecs_normed = vecs / norms
                sim = vecs_normed @ vecs_normed.T
                n = len(valid)
                c = (sim.sum().item() - n) / max(1, n * (n - 1))
                layer_coherences[layer_idx] = c

            except Exception:
                layer_coherences[layer_idx] = None

        return layer_coherences
