# -*- coding: utf-8 -*-
"""
styxx.adapters — per-provider drop-in wrappers.

Each adapter wraps a specific LLM SDK and emits a .vitals attribute
on every response. All adapters share the fail-open contract: if
vital reading fails for any reason, the underlying SDK call returns
normally and the user's agent continues unaffected.
"""
