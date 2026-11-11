# -*- coding: utf-8 -*-
"""
styxx._position0: refuse to read the hidden state at position 0 of a
one-token input whose tokenizer added no start token.

Why
---
Some tokenizers add no start token (for example Qwen2.5, Phi-3.5, GPT-2
and Pythia). With such a tokenizer and a raw prompt of one token, that
token sits alone at position 0. Models carry a very large activation at
position 0, so a hidden state read there is mostly that activation and
says little about the prompt. In the lab's sealed tests, the norm of
that state (a word alone at position 0, under the bare template) was
159.7 times (Qwen2.5-1.5B), 43.4 times (Qwen2.5-3B) and 29.1 times
(Phi-3.5-mini) its norm for the same words under each model's other
seven prompt templates; for Llama-3.2-1B and Llama-3.2-3B, whose
tokenizer adds a start token, the same ratio was 0.93 and 0.91. Source:
papers/real-convergence/ERRATUM_real_convergence_2026_11_10.md. Those
ratios were not measured again for this module, and it loads no model.

What this module does
---------------------
``reads_position0`` says whether a prefill is in that case.
``check_position0`` raises ``Position0Error`` in that case, or, when the
caller passes ``allow_position0=True``, emits a ``Position0Warning`` and
returns True so the caller can mark its result.

Standard library only: torch is never imported here. Inputs may be torch
tensors, numpy arrays, nested lists, or a mapping with an "input_ids" key.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Any, List, Optional, Tuple

ERRATUM_PATH = "papers/real-convergence/ERRATUM_real_convergence_2026_11_10.md"

_PKG_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


def _stacklevel_outside_package() -> int:
    """The stacklevel, for a warnings.warn in the function that calls this
    helper, that names the nearest frame outside the styxx package, so a
    warning raised through a wrapper (gate, Program.run,
    detect_hallucination) points at the caller's line, not at styxx."""
    frame = sys._getframe(1)  # the function that will call warnings.warn
    level = 1
    while frame is not None and os.path.normcase(
            os.path.abspath(frame.f_code.co_filename)).startswith(
                _PKG_DIR + os.sep):
        frame = frame.f_back
        level += 1
    return level


class Position0Error(ValueError):
    """Raised when a function is about to read the hidden state at
    position 0 of a one-token input whose tokenizer added no start token."""


class Position0Warning(UserWarning):
    """Marks a reading that was taken anyway (``allow_position0=True``),
    or a score that was not measured and holds a default value."""


class ChatTemplateFallbackWarning(UserWarning):
    """The chat template raised and raw tokenization was used instead."""


def _shape_and_values(input_ids: Any, want_values: bool) -> Tuple[int, List[Any]]:
    """Return (sequence length, flat list of ids or [] if not wanted).

    The sequence length is the last dimension: T for a (T,) input and for
    a (B, T) input."""
    if isinstance(input_ids, dict) or (
        hasattr(input_ids, "keys") and hasattr(input_ids, "__getitem__")
        and not hasattr(input_ids, "shape")
    ):
        try:
            input_ids = input_ids["input_ids"]
        except (KeyError, TypeError):
            pass

    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        dims = tuple(int(d) for d in shape)
        length = dims[-1] if dims else 1
        values: List[Any] = []
        if want_values and length == 1:
            flat = input_ids.reshape(-1) if dims else input_ids.reshape(1)
            values = list(flat.tolist())
        return length, values

    # nested lists, tuples or ranges: walk to the innermost level
    node = input_ids
    while isinstance(node, (list, tuple, range)) and len(node) > 0 and isinstance(node[0], (list, tuple, range)):
        node = node[0]
    length = len(node)
    values = []
    if want_values and length == 1:
        stack = [input_ids]
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple, range)):
                stack.extend(item)
            else:
                values.append(item)
    return length, values


def _start_token_id(tokenizer: Any, bos_token_id: Optional[int]) -> Optional[int]:
    if bos_token_id is not None:
        return bos_token_id
    return getattr(tokenizer, "bos_token_id", None)


def reads_position0(input_ids: Any, tokenizer: Any = None, *,
                    bos_token_id: Optional[int] = None) -> bool:
    """True exactly when the prefill is one token long and that token is
    not the start token.

    The sequence length is the last dimension of ``input_ids`` (a (1, T)
    tensor has length T). The start-token id is ``bos_token_id`` when given,
    else ``tokenizer.bos_token_id``; a tokenizer with no start-token id
    counts as adding none. An empty input returns False. For a batch of
    one-token rows, True when any row's token is not the start token.
    """
    length, values = _shape_and_values(input_ids, want_values=True)
    if length != 1 or not values:
        return False
    bos = _start_token_id(tokenizer, bos_token_id)
    if bos is None:
        return True
    return any(int(v) != int(bos) for v in values)


def _message(where: str) -> str:
    name = where or "this function"
    return (
        f"{name} was about to read the hidden state at position 0 of a "
        "one-token input whose tokenizer added no start token. That state "
        "is dominated by a very large activation models carry at position "
        "0, so the reading is not a measurement of the prompt (see "
        f"{ERRATUM_PATH}). Ways on: use the chat template "
        "(apply_chat_template=True), pass a prompt of two or more tokens, "
        "or pass allow_position0=True to read it anyway (the result is "
        "then marked)."
    )


def check_position0(input_ids: Any, tokenizer: Any = None, *,
                    allow_position0: bool = False, where: str = "",
                    bos_token_id: Optional[int] = None) -> bool:
    """Guard a reading against position 0 of a one-token input.

    Returns False, with no side effect, when ``reads_position0`` is False.
    Otherwise raises ``Position0Error``, or, when ``allow_position0`` is
    True, emits a ``Position0Warning`` and returns True so the caller can
    mark its result.

    Raises:
        Position0Error: the prefill is one token long, that token is not
            the start token, and ``allow_position0`` is False.
    """
    if not reads_position0(input_ids, tokenizer, bos_token_id=bos_token_id):
        return False
    if not allow_position0:
        raise Position0Error(_message(where))
    name = where or "this function"
    warnings.warn(
        f"{name} read the hidden state at position 0 of a one-token input "
        "whose tokenizer added no start token, because allow_position0=True. "
        "The reading is dominated by the large activation at position 0 and "
        f"is not a measurement of the prompt (see {ERRATUM_PATH}).",
        Position0Warning,
        stacklevel=_stacklevel_outside_package(),
    )
    return True


__all__ = [
    "Position0Error",
    "Position0Warning",
    "ChatTemplateFallbackWarning",
    "reads_position0",
    "check_position0",
]
