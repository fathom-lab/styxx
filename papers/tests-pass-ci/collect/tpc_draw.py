# -*- coding: utf-8 -*-
"""The draw (PREREG P3): every kept frame row on its own value u.

    u = int(sha256((beacon_hex + ":tests-pass-ci:" + str(pr_id)).encode("utf-8")).hexdigest(), 16) / 2**256

- D (drawn): kept rows with u < P_AGENT[agent]; Claude Code rows exist only for enumerated slices.
- M (main): drawn rows with u < 0.020 for every agent but Claude Code, and every drawn Claude Code
  row, so every kept pull request of any agent is in M with probability 0.020.
- A_g: the drawn rows of agent g.

Comparisons are exact: u < p is tested as h * den < p_num * 2**256 on integers. `v` = u / p_M is
stored beside u (p_M = 0.120 for Claude Code, 0.020 otherwise): v is uniform on [0, 1) inside M for
every agent; the PREREG's "lowest u" orders (audits, the cap) read u and the RESULT prints which.
"""
from __future__ import annotations

import hashlib
import re
from fractions import Fraction

from tpc_const import DRAW_TAG, P0, P_AGENT, SLICED_AGENT

HEX64 = re.compile(r"^[0-9a-f]{64}$")
TWO256 = 2 ** 256


def draw_int(beacon_hex: str, pr_id: int) -> int:
    if not HEX64.match(beacon_hex):
        raise ValueError("beacon_hex must be 64 lowercase hex characters")
    return int(hashlib.sha256((beacon_hex + DRAW_TAG + str(pr_id)).encode("utf-8")).hexdigest(), 16)


def below(h: int, p: Fraction) -> bool:
    """h / 2**256 < p, exactly."""
    return h * p.denominator < p.numerator * TWO256


def p_main(agent: str) -> Fraction:
    return P_AGENT[SLICED_AGENT] if agent == SLICED_AGENT else P0


def membership(beacon_hex: str, pr_id: int, agent: str) -> dict:
    h = draw_int(beacon_hex, pr_id)
    in_d = below(h, P_AGENT[agent])
    in_m = in_d and below(h, p_main(agent))
    u = h / TWO256
    return {"u": u, "v": float(Fraction(h, TWO256) / p_main(agent)), "in_D": in_d, "in_M": in_m,
            "h_hex": format(h, "064x")}
