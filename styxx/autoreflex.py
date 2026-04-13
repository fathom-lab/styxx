# -*- coding: utf-8 -*-
"""
styxx.autoreflex — declarative cognitive reflex rules.

    styxx.autoreflex(
        when="p1.adversarial > 0.3 AND context == reasoning",
        then="expect('reasoning')",
    )

The prescription engine tells you what happened. Autoreflex makes
it so you don't have to read the prescription — the system responds
for you, mid-session, automatically.

The gap this closes
───────────────────

Prescription 2 from weather says: "set styxx.expect('reasoning')
before long reasoning chains." Correct advice. But an agent doesn't
know it's about to enter a long reasoning chain before it starts.
By the time you read the prescription, the session is over.

Autoreflex fires mid-session: after every vitals computation, the
rule engine evaluates registered rules and executes matching actions.
The agent gets the benefit of the prescription without having to
read it or implement it.

Primitives composed
───────────────────

- Gates: condition parsing from gates.py (the "when" side)
- Config: expect(), unexpect(), set_context() (the "then" side)
- Analytics: log(), feedback() (the "then" side)
- Custom: arbitrary callables for anything else

The autoreflex module is ~150 lines because it's composition, not
invention. Every primitive already exists. This just wires them
together with a declarative rule format.

Usage
─────

    import styxx

    # Simple: when adversarial fires in a reasoning context, suppress it
    styxx.autoreflex(
        when="p1.adversarial > 0.3",
        then="expect('reasoning')",
        name="suppress-adversarial-in-reasoning",
    )

    # Compound conditions with AND
    styxx.autoreflex(
        when="gate == warn AND p4.refusal > 0.5",
        then="log(mood='defensive', note='autoreflex: refusal warn')",
        name="log-refusal-warns",
    )

    # Custom callable for anything else
    styxx.autoreflex(
        when="hallucination > 0.6",
        then=lambda v: my_alert_system.fire("hallucination detected"),
        name="alert-on-hallucination",
    )

    # Chained actions (list of actions)
    styxx.autoreflex(
        when="gate == fail",
        then=[
            "log(mood='critical', note='gate fail detected')",
            "expect('reasoning')",
            lambda v: send_slack_alert(v.gate),
        ],
        name="fail-response-chain",
    )

    # Remove a rule
    styxx.remove_autoreflex("suppress-adversarial-in-reasoning")

    # List active rules
    for rule in styxx.list_autoreflex():
        print(f"{rule.name}: when={rule.when} then={rule.then_repr}")

0.9.3+. Closes the prescribe → act gap.
"""

from __future__ import annotations

import re
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

from .vitals import Vitals


# ══════════════════════════════════════════════════════════════════
# Action parser — turns "then" strings into callables
# ══════════════════════════════════════════════════════════════════

_ACTION_EXPECT = re.compile(
    r"^\s*expect\(\s*['\"](\w+)['\"]\s*\)\s*$"
)
_ACTION_UNEXPECT = re.compile(
    r"^\s*unexpect\(\s*['\"](\w+)['\"]\s*\)\s*$"
)
_ACTION_SET_CONTEXT = re.compile(
    r"^\s*set_context\(\s*['\"](\w+)['\"]\s*\)\s*$"
)
_ACTION_CLEAR_EXPECTED = re.compile(
    r"^\s*clear_expected\(\s*\)\s*$"
)
_ACTION_FEEDBACK = re.compile(
    r"^\s*feedback\(\s*['\"](\w+)['\"]\s*\)\s*$"
)
_ACTION_LOG = re.compile(
    r"^\s*log\((.+)\)\s*$", re.DOTALL,
)


def _parse_action(action_str: str) -> Callable[[Vitals], None]:
    """Parse a 'then' string into a callable action."""
    from . import config
    from .analytics import log as styxx_log, feedback as styxx_feedback

    s = action_str.strip()

    m = _ACTION_EXPECT.match(s)
    if m:
        cat = m.group(1)
        def _act(v: Vitals, cat=cat) -> None:
            config.expect(cat)
        return _act

    m = _ACTION_UNEXPECT.match(s)
    if m:
        cat = m.group(1)
        def _act(v: Vitals, cat=cat) -> None:
            config.unexpect(cat)
        return _act

    m = _ACTION_SET_CONTEXT.match(s)
    if m:
        ctx = m.group(1)
        def _act(v: Vitals, ctx=ctx) -> None:
            config.set_context(ctx)
        return _act

    m = _ACTION_CLEAR_EXPECTED.match(s)
    if m:
        def _act(v: Vitals) -> None:
            config.clear_expected()
        return _act

    m = _ACTION_FEEDBACK.match(s)
    if m:
        outcome = m.group(1)
        def _act(v: Vitals, outcome=outcome) -> None:
            styxx_feedback(outcome)
        return _act

    m = _ACTION_LOG.match(s)
    if m:
        # Parse log(key=value, ...) kwargs
        kw_str = m.group(1)
        # Simple key='value' parser
        kwargs = {}
        for pair in re.finditer(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", kw_str):
            kwargs[pair.group(1)] = pair.group(2)
        def _act(v: Vitals, kwargs=kwargs) -> None:
            styxx_log(**kwargs)
        return _act

    raise ValueError(
        f"could not parse autoreflex action: {action_str!r}. "
        "Valid actions: expect('cat'), unexpect('cat'), "
        "set_context('ctx'), clear_expected(), feedback('outcome'), "
        "log(mood='...', note='...'), or a callable."
    )


# ══════════════════════════════════════════════════════════════════
# Condition parser — extends gate conditions with AND/context
# ══════════════════════════════════════════════════════════════════

_RE_CONTEXT_CHECK = re.compile(
    r"^\s*context\s*(?P<op>==|!=)\s*['\"]?(?P<ctx>\w+)['\"]?\s*$"
)
_RE_CONFIDENCE_CHECK = re.compile(
    r"^\s*confidence\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<thr>[0-9]*\.?[0-9]+)\s*$"
)
_RE_PROMPT_TYPE_CHECK = re.compile(
    r"^\s*prompt_type\s*(?P<op>==|!=)\s*['\"]?(?P<pt>\w+)['\"]?\s*$"
)


def _parse_single_clause(clause: str) -> Callable[[Vitals], bool]:
    """Parse a single condition clause (no AND/OR)."""
    from .gates import parse_condition
    from . import config

    clause = clause.strip()

    # context check
    m = _RE_CONTEXT_CHECK.match(clause)
    if m:
        op = m.group("op")
        ctx = m.group("ctx")
        if op == "==":
            return lambda v, ctx=ctx: config.current_context() == ctx
        else:
            return lambda v, ctx=ctx: config.current_context() != ctx

    # confidence check
    m = _RE_CONFIDENCE_CHECK.match(clause)
    if m:
        from .gates import _OPS
        op_fn = _OPS[m.group("op")]
        thr = float(m.group("thr"))
        def _conf_pred(v: Vitals, op_fn=op_fn, thr=thr) -> bool:
            if v.phase4_late is None:
                return False
            return bool(op_fn(v.phase4_late.confidence, thr))
        return _conf_pred

    # prompt_type check (pass-through for now)
    m = _RE_PROMPT_TYPE_CHECK.match(clause)
    if m:
        return lambda v: True

    # Standard gate condition (handles phase-pinned, any-phase, gate status)
    return parse_condition(clause)


def _parse_compound_condition(when: str) -> Callable[[Vitals], bool]:
    """Parse a compound condition string with AND + OR support.

    Extends the gate condition grammar with:
      - AND for combining conditions (all must match)
      - OR for alternative conditions (any must match) (0.9.5)
      - context == <name> / context != <name>
      - confidence > <threshold>
      - prompt_type == <type>

    Precedence: AND binds tighter than OR.
    "A AND B OR C AND D" = "(A AND B) OR (C AND D)"
    """
    # Split by OR first (lower precedence)
    or_groups = re.split(r'\s+OR\s+', when, flags=re.IGNORECASE)

    if len(or_groups) > 1:
        # Each OR branch is an AND group
        or_predicates = []
        for group in or_groups:
            or_predicates.append(_parse_compound_condition(group))

        def or_pred(v: Vitals, preds=or_predicates) -> bool:
            return any(p(v) for p in preds)
        return or_pred

    # No OR — parse AND clauses
    parts = re.split(r'\s+AND\s+', when, flags=re.IGNORECASE)
    predicates: List[Callable[[Vitals], bool]] = []

    for part in parts:
        predicates.append(_parse_single_clause(part))

    def compound(v: Vitals, preds=predicates) -> bool:
        return all(p(v) for p in preds)

    return compound


# ══════════════════════════════════════════════════════════════════
# Rule registry
# ══════════════════════════════════════════════════════════════════

@dataclass
class AutoReflexRule:
    """One declarative reflex rule."""
    name: str
    when: str
    then_repr: str
    predicate: Callable[[Vitals], bool]
    actions: List[Callable[[Vitals], None]]
    fire_count: int = 0
    max_fires: int = 0         # 0 = unlimited
    cooldown_s: float = 0.0    # min seconds between fires
    _last_fired: float = 0.0

    def __repr__(self) -> str:
        status = f"fired {self.fire_count}x"
        return f"<autoreflex '{self.name}': when='{self.when}' then='{self.then_repr}' ({status})>"


_RULES: List[AutoReflexRule] = []
_RULES_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def autoreflex(
    *,
    when: str,
    then: Union[str, Callable, List[Union[str, Callable]]],
    name: Optional[str] = None,
    max_fires: int = 0,
    cooldown_s: float = 5.0,
) -> AutoReflexRule:
    """Register a declarative cognitive reflex rule.

    Args:
        when:       condition string (gate DSL + AND + context/confidence).
        then:       action string, callable(Vitals), or list of either.
        name:       human-readable name for the rule.
        max_fires:  max times this rule can fire (0 = unlimited).
        cooldown_s: minimum seconds between consecutive fires of this
                    rule (default 5s to prevent spam).

    Returns:
        The AutoReflexRule for later removal/inspection.

    Usage:
        styxx.autoreflex(
            when="p1.adversarial > 0.3",
            then="expect('reasoning')",
            name="suppress-adversarial",
        )
    """
    # Parse the condition
    predicate = _parse_compound_condition(when)

    # Parse the actions
    if isinstance(then, str):
        then_list = [then]
    elif callable(then):
        then_list = [then]
    elif isinstance(then, (list, tuple)):
        then_list = list(then)
    else:
        raise TypeError(f"'then' must be a string, callable, or list; got {type(then)}")

    actions: List[Callable[[Vitals], None]] = []
    then_reprs: List[str] = []
    for item in then_list:
        if callable(item) and not isinstance(item, str):
            actions.append(item)
            then_reprs.append(f"<callable {getattr(item, '__name__', '?')}>")
        elif isinstance(item, str):
            actions.append(_parse_action(item))
            then_reprs.append(item)
        else:
            raise TypeError(f"action must be a string or callable; got {type(item)}")

    auto_name = name or f"rule-{len(_RULES) + 1}"
    then_repr = "; ".join(then_reprs)

    rule = AutoReflexRule(
        name=auto_name,
        when=when,
        then_repr=then_repr,
        predicate=predicate,
        actions=actions,
        max_fires=max_fires,
        cooldown_s=cooldown_s,
    )

    with _RULES_LOCK:
        _RULES.append(rule)

    # Auto-register a gate callback that dispatches this rule's actions.
    # This piggybacks on the existing gate dispatch infrastructure so
    # autoreflex rules fire on every vitals computation automatically.
    #
    # Extract the first atomic clause (strip AND/OR) for the gate hook.
    # The full compound condition is evaluated by dispatch_rule().
    from .gates import on_gate

    def _rule_callback(vitals: Vitals, rule=rule) -> None:
        dispatch_rule(rule, vitals)

    # Get the first atomic clause by splitting on both AND and OR
    first_clause = re.split(r'\s+(?:AND|OR)\s+', when, flags=re.IGNORECASE)[0].strip()
    on_gate(first_clause, _rule_callback, name=f"autoreflex:{auto_name}")

    return rule


def dispatch_rule(rule: AutoReflexRule, vitals: Vitals) -> bool:
    """Evaluate and fire a single rule against vitals.

    Called by the gate callback registered during autoreflex().
    Returns True if the rule fired.
    """
    import time

    # Check max fires
    if rule.max_fires > 0 and rule.fire_count >= rule.max_fires:
        return False

    # Check cooldown
    now = time.time()
    if rule.cooldown_s > 0 and (now - rule._last_fired) < rule.cooldown_s:
        return False

    # Evaluate compound condition (gate callback already matched
    # the first clause; we need to check the full compound)
    try:
        if not rule.predicate(vitals):
            return False
    except Exception:
        return False

    # Fire all actions
    for action in rule.actions:
        try:
            action(vitals)
        except Exception as e:
            warnings.warn(
                f"autoreflex '{rule.name}' action raised "
                f"{type(e).__name__}: {e}. rule still active.",
                RuntimeWarning,
                stacklevel=2,
            )

    rule.fire_count += 1
    rule._last_fired = now
    return True


def remove_autoreflex(name: str) -> bool:
    """Remove an autoreflex rule by name. Returns True if found."""
    with _RULES_LOCK:
        for i, rule in enumerate(_RULES):
            if rule.name == name:
                _RULES.pop(i)
                # Also remove the gate callback
                from .gates import list_gates, remove_gate
                for g in list_gates():
                    if g.name == f"autoreflex:{name}":
                        remove_gate(g)
                return True
    return False


def list_autoreflex() -> List[AutoReflexRule]:
    """Return a snapshot of all registered autoreflex rules."""
    with _RULES_LOCK:
        return list(_RULES)


def clear_autoreflex() -> int:
    """Remove all autoreflex rules. Returns the number removed."""
    with _RULES_LOCK:
        n = len(_RULES)
        names = [r.name for r in _RULES]
        _RULES.clear()

    # Clean up gate callbacks
    from .gates import list_gates, remove_gate
    for g in list_gates():
        if g.name and g.name.startswith("autoreflex:"):
            remove_gate(g)

    return n


# ══════════════════════════════════════════════════════════════════
# Prescription-to-autoreflex translator (1.0.0)
# ══════════════════════════════════════════════════════════════════
#
# The prescription engine says "set expected('reasoning') before
# long chains." That's an autoreflex rule written in English. This
# translator converts prescription patterns into rules automatically.

# Map known prescription patterns → autoreflex rules
_PRESCRIPTION_RULES = [
    {
        "match": "adversarial cascade",
        "when": "p1.adversarial > 0.30",
        "then": "expect('reasoning')",
        "name": "rx:suppress-adversarial-cascade",
    },
    {
        "match": "refusal spiral",
        "when": "p4.refusal > 0.40",
        "then": "expect('refusal')",
        "name": "rx:break-refusal-spiral",
    },
    {
        "match": "session fatigue",
        "when": "confidence < 0.25",
        "then": "log(mood='fatigued', note='autoreflex: session fatigue detected')",
        "name": "rx:log-session-fatigue",
    },
    {
        "match": "low-confidence drift",
        "when": "confidence < 0.25",
        "then": "log(mood='uncertain', note='autoreflex: low confidence drift')",
        "name": "rx:log-low-confidence",
    },
    {
        "match": "hallucination rate trending up",
        "when": "p4.hallucination > 0.35",
        "then": "log(mood='drifting', note='autoreflex: hallucination trend detected')",
        "name": "rx:log-hallucination-trend",
    },
]


def autoreflex_from_prescriptions(
    prescriptions: Optional[List[str]] = None,
) -> List[AutoReflexRule]:
    """Convert weather prescriptions into autoreflex rules.

    Reads the current prescriptions (from weather()) and registers
    autoreflex rules for any that have known action mappings. This
    closes the loop: the prescription engine ADVISES, and this
    function IMPLEMENTS the advice automatically.

    Returns the list of rules that were registered.

    Usage:
        import styxx
        report = styxx.weather(agent_name="xendro")
        rules = styxx.autoreflex_from_prescriptions(report.prescriptions)
        # Now those prescriptions are active rules
    """
    if prescriptions is None:
        # Auto-fetch from weather
        from .weather import weather
        report = weather()
        if report is None:
            return []
        prescriptions = report.prescriptions

    registered = []
    existing_names = {r.name for r in list_autoreflex()}

    for rx_text in prescriptions:
        rx_lower = rx_text.lower()
        for rule_def in _PRESCRIPTION_RULES:
            if rule_def["match"] in rx_lower and rule_def["name"] not in existing_names:
                try:
                    rule = autoreflex(
                        when=rule_def["when"],
                        then=rule_def["then"],
                        name=rule_def["name"],
                        cooldown_s=30.0,
                    )
                    registered.append(rule)
                    existing_names.add(rule_def["name"])
                except Exception:
                    pass

    return registered
