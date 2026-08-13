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

# Minimum |phi| for a live term to count as tracking the outcome. Set low deliberately:
# the receipt should refuse only when NO term carries any signal at all, since a proxy
# for taint should err toward letting a measurement pass rather than toward silently
# killing a sound one.
MIN_ABS_PHI = 0.10


def install_for(prefixes):
    """Instrument these packages. Must run before the subject is imported."""
    pe.install(list(prefixes))


class scope:                                                 # noqa: N801
    """Context manager attributing decision-term observations to one measurement."""

    def __init__(self, name):
        self.name = name
        self._before = {}
        self.delta = {}
        # Per-item records, when the caller marks them. A measurement that aggregates
        # over items (a rate over N drafts) is contingent only if some item's outcome
        # could have differed, and pooling every observation into one bag loses that
        # entirely -- which is how the first version certified a rate of exactly 1.0
        # as "could have failed".
        self._items = []
        self._item_mark = {}

    def mark_item(self, outcome):
        """Close one item of the aggregate and record the outcome it produced.

        Call once per unit the value averages over. Without this the receipt can only
        speak about the pooled observation bag, which is not the same question.
        """
        snap = {}
        for tid, o in pe._OBS.items():
            bt, bf = self._item_mark.get(tid, (0, 0))
            dt, df = o["t"] - bt, o["f"] - bf
            if dt or df:
                snap[tid] = (dt, df)
        self._item_mark = {tid: (o["t"], o["f"]) for tid, o in pe._OBS.items()}
        self._items.append({"outcome": bool(outcome), "terms": snap})

    def __enter__(self):
        self._before = {tid: (o["t"], o["f"]) for tid, o in pe._OBS.items()}
        self._item_mark = dict(self._before)
        return self

    def __exit__(self, *exc):
        self.delta = {}
        for tid, o in pe._OBS.items():
            bt, bf = self._before.get(tid, (0, 0))
            dt, df = o["t"] - bt, o["f"] - bf
            if dt or df:
                self.delta[tid] = (dt, df)
        return False

    def outcome_association(self):
        """phi coefficient between each term's per-item value and the item outcome.

        This is the question a receipt is actually for. A term can be adjudicative,
        heavily exercised, and genuinely two-valued while having nothing to do with the
        result: the conscience's character-level tokenisation runs `if i == 0` and
        `if ' ' in phrase` tens of thousands of times per measurement, and both are
        properties of the lexicon files rather than of the draft being judged. Counting
        them as evidence that the verdict could have differed is how the pooled version
        certified a rate of 1.0 as falsifiable.

        Per item, a term's value is its majority (t > f). phi is then the standard 2x2
        correlation between that and the outcome. |phi| == 0 means the term's pattern
        carries no information about the result -- it ran alongside the decision, not
        inside it. This is a statistical proxy for taint, not taint: a term perfectly
        confounded with a real cause will inherit its phi. It is deliberately the
        conservative direction for a REFUSAL and the permissive one for a pass, which
        is the same asymmetry the rest of this file is built on.
        """
        if len(self._items) < 2:
            return {}
        out = {}
        outcomes = [it["outcome"] for it in self._items]
        for tid in self.delta:
            xs, ys = [], []
            for it, oc in zip(self._items, outcomes):
                tf = it["terms"].get(tid)
                if not tf:
                    continue
                t, f = tf
                if t == f:
                    continue                      # no majority; carries no signal
                xs.append(t > f)
                ys.append(oc)
            n = len(xs)
            if n < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
                out[tid] = 0.0
                continue
            n11 = sum(1 for a, b in zip(xs, ys) if a and b)
            n10 = sum(1 for a, b in zip(xs, ys) if a and not b)
            n01 = sum(1 for a, b in zip(xs, ys) if not a and b)
            n00 = sum(1 for a, b in zip(xs, ys) if not a and not b)
            denom = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
            out[tid] = 0.0 if denom == 0 else round(
                (n11 * n00 - n10 * n01) / denom, 4)
        return out

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
                        "op": meta.get("op"), "pos": meta.get("pos"),
                        "src": meta.get("src")})
        return sorted(out, key=lambda r: -r["n"])

    def receipt(self, value=None, min_live_fraction=0.10):
        """Emit the certificate. The verdict field is the part meant to be obeyed.

        VERDICT IS COMPUTED ON ADJUDICATIVE TERMS ONLY. Adversarial review of the first
        two real receipts found the decisive failure: re-scoped on the 58 drafts where
        the conscience fired every single time (value literally 1.0) and on the 25 where
        it never fired (value 0.0), this method still returned
        `OK__path_could_have_failed` at 37.5% and 25.5% live. The measurements could not
        have come out differently and the receipt certified that they could.

        The mechanism was pooling. Character-level tokenisation loops inside the
        instrument generate tens of thousands of live observations that have nothing to
        do with the verdict, and they outvote the handful of terminal decision terms --
        every one of which the receipt's own rows correctly recorded as CONSTANT. The
        heaviest 'live' term was `' ' in phrase` at n=30,174: a compile-time property of
        a lexicon file, invariant under every possible input, certified LIVE.

        That is a pass verdict decoupled from the question it claims to answer, inside
        the instrument built to detect exactly that, and it violates the lab's own
        standing rule that a leg which cannot fail must not gate. Restricting the
        verdict to terms whose value is consumed as a decision is the minimum fix; the
        live terms are now serialised too, so an OK verdict carries its own evidence
        instead of asserting it.
        """
        rows = self.rows()
        adj = [r for r in rows if r.get("pos") == "adjudicative"]
        # Fall back to all rows only when position is unavailable (a run recorded by a
        # prober older than the position field). Stated in the receipt, never silent.
        basis, basis_name = (adj, "adjudicative") if adj else (rows, "all_terms")
        live = [r for r in basis if r["verdict"] == "LIVE"]
        const = [r for r in basis if r["verdict"].startswith("CONSTANT")]
        under = [r for r in basis if r["verdict"] == "UNDERPOWERED"]
        adjudicable = len(live) + len(const)
        frac = (len(live) / adjudicable) if adjudicable else None

        # THE FIRST QUESTION, ASKED BEFORE ANY TERM STATISTICS. If the caller marked
        # items and every item produced the same outcome, the aggregate is a constant
        # and no amount of live machinery inside the loop makes it contingent. This is
        # the check whose absence let the method certify a rate of 1.0 as falsifiable:
        # restricting to adjudicative terms does NOT fix it on its own, because a
        # tokenisation loop's `if` is adjudicative too and there are tens of thousands
        # of them. Only the outcome distribution answers the question that was asked.
        n_items = len(self._items)
        outcomes = {it["outcome"] for it in self._items}
        if n_items >= 2 and len(outcomes) == 1:
            only = next(iter(outcomes))
            return self._emit(
                value, "REFUSED__outcome_constant",
                f"All {n_items} items produced the same outcome ({only}). The value is "
                f"a constant over this population: no decision inside the measurement "
                f"could have changed it, however much live machinery ran alongside.",
                rows, basis, basis_name, live, const, under, frac, min_live_fraction)

        # Second question: of the live decision terms, do ANY of them track the result?
        # A term that varies without moving the outcome is machinery running beside the
        # decision, not the decision.
        phi = self.outcome_association()
        linked = [r for r in live if abs(phi.get(r["term_id"], 0.0)) >= MIN_ABS_PHI]
        if n_items >= 2 and live and not linked:
            return self._emit(
                value, "REFUSED__no_outcome_linked_term",
                f"{len(live)} adjudicable terms varied, but none of them tracks the "
                f"result across the {n_items} items (|phi| < {MIN_ABS_PHI}). They ran "
                f"beside the decision rather than inside it, so their variation is not "
                f"evidence that the value could have differed.",
                rows, basis, basis_name, live, const, under, frac, min_live_fraction,
                phi, linked)

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

        return self._emit(value, verdict, why, rows, basis, basis_name, live, const,
                          under, frac, min_live_fraction, phi, linked)

    def _emit(self, value, verdict, why, rows, basis, basis_name, live, const,
              under, frac, min_live_fraction, phi=None, linked=None):
        phi = phi or {}
        linked = linked or []
        return {
            "measurement": self.name,
            "value": value,
            "verdict": verdict,
            "why": why,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope_obs_floor": SCOPE_OBS_FLOOR,
            "n_items": len(self._items),
            "n_distinct_outcomes": len({it["outcome"] for it in self._items}),
            "verdict_basis": basis_name,
            "n_terms_in_scope": len(rows),
            "n_terms_on_path": len(basis),
            "n_live": len(live),
            "n_constant": len(const),
            "n_underpowered": len(under),
            "live_fraction_of_adjudicable": (round(frac, 4)
                                             if frac is not None else None),
            # LIVE terms are serialised, and this is not bookkeeping. The first version
            # wrote only `const[:25]` and never the live terms at all, so the evidence
            # FOR an OK verdict -- the whole basis of the certificate -- could not be
            # recovered from the artifact. A receipt that records only what it ruled out
            # is not auditable, and the gate receipt had to be withdrawn partly for
            # that reason. Truncation is now explicit rather than silent.
            "n_live_outcome_linked": len(linked),
            "min_abs_phi": MIN_ABS_PHI,
            "outcome_linked_terms": [
                {**{k: r[k] for k in ("module", "func", "line", "op", "n", "src")},
                 "phi": phi.get(r["term_id"])}
                for r in sorted(linked, key=lambda r: -abs(phi.get(r["term_id"], 0)))
                [:20]],
            "live_terms": [
                {**{k: r[k] for k in ("module", "func", "line", "op", "verdict", "n",
                                      "n_true", "n_false", "src")},
                 "phi": phi.get(r["term_id"])}
                for r in live[:40]],
            "live_terms_truncated": max(0, len(live) - 40),
            "pinned_terms": [
                {k: r[k] for k in ("module", "func", "line", "op", "verdict", "n",
                                   "src")}
                for r in const[:40]],
            "pinned_terms_truncated": max(0, len(const) - 40),
            "limits": [
                "Attribution is by observation delta, not data-flow taint: a term "
                "evaluated inside the scope is credited even if its result never "
                "reached the value. This biases the receipt toward looking "
                "FALSIFIABLE, so REFUSED is strong evidence and OK is weak evidence.",
                "Single-threaded only. Concurrent work inside the scope is "
                "misattributed to this measurement.",
                "A CONSTANT verdict is about this measurement's population, not about "
                "the term in principle.",
                "WITHOUT mark_item() the receipt can only speak about the pooled "
                "observation bag. That is a weaker question than whether the VALUE "
                "could have differed, and pooling is how an earlier version certified "
                "a rate of exactly 1.0 as falsifiable. Call mark_item() per unit the "
                "value aggregates over whenever the measurement is an aggregate.",
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
           # The decisive fixture, added 2026-08-13 after adversarial review found the
           # method certifying a rate of exactly 1.0 as falsifiable. The verdict here is
           # pinned, but a busy tokenisation-style loop runs alongside it producing
           # dozens of genuinely LIVE adjudicative terms -- which is what outvoted the
           # decision terms in the real conscience receipt. Restricting the basis to
           # adjudicative terms does NOT catch this; only the outcome distribution does.
           "def pinned_with_busy_loop(x):\n"
           "    n = 0\n"
           "    for ch in 'abcdefghij':\n"
           "        if ch < 'e':\n"
           "            n += 1\n"
           "        if ch != 'q':\n"
           "            n += 2\n"
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
        for i in range(200):
            pass
    r2 = s2.receipt(value=cont_rate)
    # mark items for the contingent case too, so both arms of the new check are
    # exercised: outcomes here genuinely vary and must NOT trip the constant-outcome
    # refusal.
    with scope("contingent_marked") as s2b:
        hits = 0
        for i in range(200):
            out = ns["contingent"](i - 25)
            hits += bool(out)
            s2b.mark_item(out)
    r2b = s2b.receipt(value=hits / 200)

    # THE REGRESSION CASE. A rate of exactly 1.0, with dozens of live adjudicative
    # terms running alongside it in a busy loop. The old method returned OK here.
    with scope("pinned_with_bystanders") as s3:
        fired = 0
        for i in range(50):
            out = ns["pinned_with_busy_loop"](i)
            fired += bool(out)
            s3.mark_item(out)
        busy_rate = fired / 50
    r3 = s3.receipt(value=busy_rate)

    checks = [
        ("pinned metric is REFUSED", r1["verdict"] == "REFUSED__no_live_terms"),
        ("pinned metric names the pinned term", bool(r1["pinned_terms"])),
        ("contingent metric is OK", r2["verdict"] == "OK__path_could_have_failed"),
        ("contingent metric has live terms", r2["n_live"] > 0),
        ("contingent metric serialises its LIVE terms as evidence",
         bool(r2.get("live_terms"))),
        ("marked contingent metric still OK (varying outcomes do not trip refusal)",
         r2b["verdict"] == "OK__path_could_have_failed"),
        # The pinned fixture still returns a perfectly ordinary-looking rate of 1.0.
        # That is the entire point: the VALUE gives no hint, only the receipt does.
        ("the pinned value looks unremarkable on its own", pinned_rate == 1.0),
        # The fixture is only a valid control if live bystanders actually exist to be
        # outvoted -- otherwise the refusal proves nothing about pooling.
        ("busy-loop fixture really does produce live bystanders", r3["n_live"] >= 1),
        # phi must ALSO reject bystanders even when outcomes do vary -- the
        # constant-outcome check cannot catch that case.
        ("bystanders carry no outcome signal (phi ~ 0)",
         all(abs((t.get("phi") or 0)) < 0.10 for t in r3.get("live_terms", []))),
        ("the contingent metric's decision term IS outcome-linked",
         r2b.get("n_live_outcome_linked", 0) >= 1),
        ("rate of 1.0 with live bystanders is REFUSED, not OK",
         r3["verdict"] == "REFUSED__outcome_constant"),
        ("...and its value is indeed a constant 1.0", busy_rate == 1.0),
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
