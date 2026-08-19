# -*- coding: utf-8 -*-
"""The OpenAI adapter's injection retry costs a SECOND PAID CALL.

It used to fire on `except Exception` — every rate limit, timeout, auth failure
and server error silently doubled the caller's spend, immediately re-hit a
rate-limited endpoint, and (because the retry popped both fields regardless of
who supplied them) stripped a `logprobs=True` the caller had passed explicitly,
so their own request came back unscored.

The retry is now narrow: only a bad-request-shaped rejection of the fields
styxx itself injected, only those fields removed, and it says so out loud.
"""
import warnings

import pytest

from styxx.adapters.openai import _CompletionsShim, _is_param_rejection


class RateLimitError(Exception):
    status_code = 429


class APITimeoutError(Exception):
    pass


class AuthenticationError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


class _Recorder:
    """Inner client that records every call it is paid for."""

    def __init__(self, fail_with=None, fail_times=1):
        self.calls = []
        self._fail_with = fail_with
        self._fail_times = fail_times

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self._fail_with is not None and len(self.calls) <= self._fail_times:
            raise self._fail_with
        class _R:
            model = "gpt-4"
            choices = []
        return _R()


@pytest.mark.parametrize("exc", [
    RateLimitError("Rate limit reached for gpt-4"),
    APITimeoutError("Request timed out"),
    AuthenticationError("Invalid API key provided"),
])
def test_transient_errors_do_not_buy_a_second_call(exc):
    inner = _Recorder(fail_with=exc, fail_times=99)
    shim = _CompletionsShim(inner, None)
    with pytest.raises(type(exc)):
        shim.create(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
    assert len(inner.calls) == 1, "a transient failure must not be retried on the user's dime"


def test_parameter_rejection_retries_once_and_warns():
    inner = _Recorder(fail_with=BadRequestError("Unknown parameter: 'top_logprobs'"))
    shim = _CompletionsShim(inner, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim.create(model="gpt-4", messages=[])
    assert len(inner.calls) == 2                       # the one retry it is for
    assert any("rejected" in str(c.message) for c in caught)


def test_retry_never_strips_a_value_the_caller_supplied():
    """`injected` was True whenever EITHER field was missing, so the retry
    dropped an explicit logprobs=True along with styxx's own top_logprobs."""
    inner = _Recorder(fail_with=BadRequestError("Unknown parameter: 'top_logprobs'"))
    shim = _CompletionsShim(inner, None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        shim.create(model="gpt-4", logprobs=True, messages=[])
    retry = inner.calls[1]
    assert retry.get("logprobs") is True, "the caller asked for logprobs; keep it"
    assert "top_logprobs" not in retry, "only styxx's own injection is removed"


def test_param_rejection_discriminator():
    keys = ["top_logprobs"]
    assert not _is_param_rejection(RateLimitError("Rate limit reached"), keys)
    assert not _is_param_rejection(APITimeoutError("timed out"), keys)
    assert not _is_param_rejection(AuthenticationError("Invalid API key"), keys)
    assert _is_param_rejection(BadRequestError("Unknown parameter: 'top_logprobs'"), keys)
    assert _is_param_rejection(BadRequestError("unsupported value"), keys)
    # an explicit non-400 status is never a parameter problem, whatever it says
    class Weird(Exception):
        status_code = 500
    assert not _is_param_rejection(Weird("top_logprobs exploded"), keys)
