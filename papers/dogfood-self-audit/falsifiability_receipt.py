"""A falsifiability receipt: could THIS number have come out differently?

Three literatures each answer a neighbouring question and none answers this one.

  Mutation testing asks whether a TEST SUITE would notice a change in the code.
  Execution provenance asks where a claim's INPUTS came from.
  PROBE E (probe_e_runtime.py) asks whether a GATE could have decided otherwise.

A published measurement needs the conjunction, scoped to itself. Not "is this
instrument healthy in general" but: **of the decision terms that actually executed while
producing this specific value, how many could have gone the other way?** A number
computed entirely by terms that never varied is not a weak measurement or a noisy one.
It is a number the apparatus could not have failed to produce, and it carries no
information about the world -- it reports the constants, in the units of the thing it
claims to measure.

That is checkable, cheaply, and it is what this file does. Wrap a computation in
`scope()`; every decision term evaluated inside gets its own within-scope verdict from
the counts observed DURING that computation, not from the module's lifetime history. A
term that varies elsewhere in the program but was pinned throughout this measurement is
dead FOR THIS NUMBER, and lifetime statistics would hide exactly that.

The receipt is deliberately refusable. `REFUSED__no_live_terms` is a verdict the caller
must handle, not a warning to log: the discipline this whole program is built on is that
an instrument which cannot fail must not be allowed to certify. Applied here to the
instrument's own output.

    from falsifiability_receipt import scope, install_for
    install_for(["styxx"])
    with scope("cave_rate") as sc:
        rate = measure_cave_rate(battery)
    print(sc.receipt(value=rate))

Limits, stated plainly. Attribution is by observation delta across the scope, not by
data-flow taint: a term evaluated inside the block is credited to the block even if its
result did not reach the value. That over-attributes, which biases the receipt toward
LOOKING falsifiable, so a REFUSED verdict is strong evidence and an OK verdict is weak
evidence. It is single-threaded-honest only; concurrent work inside the scope is
misattributed. Both limits are recorded in every receipt rather than left to the reader.
"""
from __future__ import annotations

import json
import time

import probe_e_runtime as pe

# A scope needs at least this many observations of a term before its within-scope
# constancy means anything. Deliberately lower than PROBE E's lifetime floor: a single
# measurement is a smaller population by construction, and holding it to the lifetime
# floor would mark every honest short computation UNDERPOWERED and thereby make the
# receipt unfalsifiable in the other direction.
SCOPE_OBS_FLOOR = 4


def install_for(prefixes):
    """Instrument these packages. Must run before the subject is imported."""
    pe.install(list(prefixes))


class scope:                                                 # noqa: N801
    """Context manager attributing decision-term observations to one measurement."""

    def __init__(self, name):
        self.name = name
        self._before = {}
        self.delta = {}

    def __enter__(self):
        self._before = {tid: (o["t"], o["f"]) for tid, o in pe._OBS.items()}
        return self

    def __exit__(self, *exc):
        self.delta = {}
        for tid, o in pe._OBS.items():
            bt, bf = self._before.get(tid, (0, 0))
            dt, df = o["t"] - bt, o["f"] - bf
            if dt or df:
                self.delta[tid] = (dt, df)
        return False

    def rows(self):
        out = []
        for tid, (dt, df) in self.delta.items():
            n = dt + df
            if n < SCOPE_OBS_FLOOR:
                verdict = "UNDERPOWERED"
            elif dt == 0:
                verdict = "CONSTANT_FALSE"
            elif df == 0:
                verdict = "CONSTANT_TRUE"
            else:
                verdict = "LIVE"
            meta = pe._META.get(tid, {})
            out.append({"term_id": tid, "n": n, "n_true": dt, "n_false": df,
                        "verdict": verdict, "module": meta.get("module"),
                        "func": meta.get("func"), "line": meta.get("line"),
                        "op": meta.get("op"), "src": meta.get("src")})
        return sorted(out, key=lambda r: -r["n"])

    def receipt(self, value=None, min_live_fraction=0.10):
        """Emit the certificate. The verdict field is the part meant to be obeyed."""
        rows = self.rows()
        live = [r for r in rows if r["verdict"] == "LIVE"]
        const = [r for r in rows if r["verdict"].startswith("CONSTANT")]
        under = [r for r in rows if r["verdict"] == "UNDERPOWERED"]
        adjudicable = len(live) + len(const)
        frac = (len(live) / adjudicable) if adjudicable else None

        if adjudicable == 0:
            verdict = "REFUSED__nothing_adjudicable"
            why = ("No decision term inside this measurement was evaluated often "
                   "enough to say whether it could have varied. The receipt cannot "
                   "speak, which is not the same as the measurement being sound.")
        elif not live:
            verdict = "REFUSED__no_live_terms"
            why = (f"All {len(const)} adjudicable decision terms on this value's path "
                   f"were pinned throughout the measurement. The apparatus could not "
                   f"have produced a different answer, so the value reports its own "
                   f"constants rather than the world.")
        elif frac < min_live_fraction:
            verdict = "WEAK__mostly_pinned"
            why = (f"Only {len(live)} of {adjudicable} adjudicable terms varied "
                   f"({frac:.1%}, below the {min_live_fraction:.0%} floor). Most of the "
                   f"path was fixed; the value is closer to a constant than the "
                   f"apparatus suggests.")
        else:
            verdict = "OK__path_could_have_failed"
            why = (f"{len(live)} of {adjudicable} adjudicable terms varied "
                   f"({frac:.1%}). The apparatus could have returned a different "
                   f"answer on this population.")

        return {
            "measurement": self.name,
            "value": value,
            "verdict": verdict,
            "why": why,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope_obs_floor": SCOPE_OBS_FLOOR,
            "n_terms_on_path": len(rows),
            "n_live": len(live),
            "n_constant": len(const),
            "n_underpowered": len(under),
            "live_fraction_of_adjudicable": (round(frac, 4)
                                             if frac is not None else None),
            "pinned_terms": [
                {k: r[k] for k in ("module", "func", "line", "op", "verdict", "n",
                                   "src")}
                for r in const[:25]],
            "limits": [
                "Attribution is by observation delta, not data-flow taint: a term "
                "evaluated inside the scope is credited even if its result never "
                "reached the value. This biases the receipt toward looking "
                "FALSIFIABLE, so REFUSED is strong evidence and OK is weak evidence.",
                "Single-threaded only. Concurrent work inside the scope is "
                "misattributed to this measurement.",
                "A CONSTANT verdict is about this measurement's population, not about "
                "the term in principle.",
            ],
        }


def selftest():
    """Two measurements with known answers: one pinned, one genuinely contingent.

    A receipt generator that has never refused anything is itself an instrument that
    cannot fail, and shipping one inside this program would be the joke writing itself.
    """
    import ast                                               # noqa: PLC0415

    src = ("def pinned(x):\n"
           "    return FLAG or x > 1000000\n"
           "def contingent(x):\n"
           "    return x > 0 and x < 100\n")
    tree = ast.parse(src)
    rw = pe._Rewriter("<fixture>", "fixture")
    tree = rw.visit(tree)
    ast.fix_missing_locations(tree)
    ns = {"_probe_e_rec": pe._rec, "FLAG": True}
    exec(compile(tree, "<fixture>", "exec"), ns)             # noqa: S102

    with scope("pinned_metric") as s1:
        pinned_rate = sum(ns["pinned"](i) for i in range(50)) / 50
    r1 = s1.receipt(value=pinned_rate)

    with scope("contingent_metric") as s2:
        cont_rate = sum(ns["contingent"](i - 25) for i in range(200)) / 200
    r2 = s2.receipt(value=cont_rate)

    checks = [
        ("pinned metric is REFUSED", r1["verdict"] == "REFUSED__no_live_terms"),
        ("pinned metric names the pinned term", bool(r1["pinned_terms"])),
        ("contingent metric is OK", r2["verdict"] == "OK__path_could_have_failed"),
        ("contingent metric has live terms", r2["n_live"] > 0),
        # The pinned fixture still returns a perfectly ordinary-looking rate of 1.0.
        # That is the entire point: the VALUE gives no hint, only the receipt does.
        ("the pinned value looks unremarkable on its own", pinned_rate == 1.0),
    ]
    ok = True
    for label, good in checks:
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}")
    print(f"\n  pinned    -> {r1['verdict']}  (value {pinned_rate})")
    print(f"  contingent-> {r2['verdict']}  (value {cont_rate})")
    print(f"\n  RECEIPT VALIDATION: "
          f"{'PASS — refuses a pinned number, passes a contingent one' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
    print("run with --selftest")
