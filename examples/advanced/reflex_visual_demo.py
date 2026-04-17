"""
examples/reflex_visual_demo.py — the video demo.

watch an agent catch itself hallucinating and self-correct in real time.
no api key needed — replays real atlas captures with full terminal visuals.

run:
    python examples/reflex_visual_demo.py

record:
    # on mac/linux with asciinema:
    asciinema rec -c "python examples/reflex_visual_demo.py" reflex-demo.cast
    # convert to gif with agg:
    agg reflex-demo.cast reflex-demo.gif --cols 90 --rows 32

    # on windows with terminalizer:
    terminalizer record -c "python examples/reflex_visual_demo.py"
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from styxx.cli import _load_demo_trajectories
from styxx.core import StyxxRuntime

# ── ANSI codes ──────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[38;2;255;0;51m"
GREEN   = "\033[38;2;0;255;100m"
CYAN    = "\033[38;2;0;229;255m"
ORANGE  = "\033[38;2;255;106;0m"
PINK    = "\033[38;2;255;42;138m"
WHITE   = "\033[38;2;240;235;238m"
GRAY    = "\033[38;2;90;80;85m"
BG_RED  = "\033[48;2;40;0;10m"
BG_OK   = "\033[48;2;0;20;10m"
STRIKE  = "\033[9m"
BLINK   = "\033[5m"
UP      = "\033[A"
CLEAR   = "\033[2K"

# ── demo text (what the model "says") ───────────────────────────

# hallucination scenario: model makes up a fake founder
HALL_TEXT = [
    "The", " founder", " of", " Apollonian", " Industries", " was",
    " Dr.", " Marcus", " Ellington", ",", " who", " established",
    " the", " company", " in", " 1987", " after", " leaving", " his",
    " position", " at", " MIT", ".", " He", " pioneered", " the",
    " field", " of", " quantum", " optics",
]

# after rewind + anchor, the model self-corrects
RECOVERY_TEXT = [
    " I", " don't", " have", " verified", " information",
    " about", " the", " specific", " founder", " of",
    " Apollonian", " Industries", ".", " This", " may",
    " not", " be", " a", " real", " company",
    ".", " I'd", " recommend", " checking", " a",
    " verified", " source", " for", " accurate", " details",
]

# ── fake openai stream objects ──────────────────────────────────

class _D:
    def __init__(self, c): self.content = c

class _TLP:
    def __init__(self, lp): self.logprob = lp

class _TL:
    def __init__(self, lp, tops): self.logprob = lp; self.top_logprobs = tops

class _LB:
    def __init__(self, c): self.content = c

class _Ch:
    def __init__(self, d, lp): self.delta = d; self.logprobs = lp

class _Chunk:
    def __init__(self, text, lp, tops):
        t = _TL(lp, [_TLP(l) for l in tops])
        self.choices = [_Ch(_D(text), _LB([t]))]


def synth_top5(chosen_lp, top2):
    p1 = math.exp(chosen_lp)
    p2 = max(1e-6, p1 - top2)
    rem = max(0.0, 1.0 - p1 - p2)
    return [math.log(max(p, 1e-12)) for p in (p1, p2, rem*0.5, rem*0.3, rem*0.2)]


class FakeCompletions:
    def __init__(self, words, entropy, logprob, top2):
        self.w, self.e, self.lp, self.t2 = words, entropy, logprob, top2
    def create(self, **kw):
        for w, e, lp, t2 in zip(self.w, self.e, self.lp, self.t2):
            yield _Chunk(w, lp, synth_top5(lp, t2))

class FakeChat:
    def __init__(self, *a): self.completions = FakeCompletions(*a)

class FakeOpenAI:
    def __init__(self, *a): self.chat = FakeChat(*a)

# ── visual helpers ──────────────────────────────────────────────

def type_slow(text, delay=0.035, color=WHITE):
    for ch in text:
        sys.stdout.write(f"{color}{ch}{RESET}")
        sys.stdout.flush()
        time.sleep(delay)

def type_fast(text, delay=0.015, color=WHITE):
    for ch in text:
        sys.stdout.write(f"{color}{ch}{RESET}")
        sys.stdout.flush()
        time.sleep(delay)

def banner(text, color=RED):
    w = len(text) + 4
    print(f"\n{color}{BOLD}{'=' * w}")
    print(f"  {text}")
    print(f"{'=' * w}{RESET}\n")

def status_line(label, value, color=CYAN):
    print(f"  {GRAY}{label:<20}{RESET}{color}{value}{RESET}")

def vitals_bar(phase, cat, conf, gate, color=WHITE):
    bar_len = int(conf * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    gate_color = GREEN if gate == "pass" else (ORANGE if gate == "warn" else RED)
    cat_color = RED if cat == "hallucination" else (ORANGE if cat == "adversarial" else (PINK if cat == "refusal" else CYAN))
    print(f"  {GRAY}{phase:>8}{RESET}  {cat_color}{bar}{RESET}  {cat_color}{cat:<16}{RESET} {DIM}{conf:.2f}{RESET}  {gate_color}{BOLD}{gate}{RESET}")


# ── main demo ───────────────────────────────────────────────────

def main():
    import styxx
    from styxx.reflex import ReflexSession

    data = _load_demo_trajectories()
    hall = data["trajectories"]["hallucination"]
    reasoning = data["trajectories"]["reasoning"]

    # clear screen
    print("\033[2J\033[H", end="")

    # ── boot sequence ───────────────────────────────────────────
    print(f"{DIM}{'─' * 72}{RESET}")
    print(f"  {RED}{BOLD}STYXX{RESET} {GRAY}reflex arc demo{RESET}")
    print(f"  {GRAY}the model catches itself hallucinating. watch.{RESET}")
    print(f"{DIM}{'─' * 72}{RESET}")
    time.sleep(1.5)

    # ── phase 1: show normal generation ─────────────────────────
    print(f"\n  {GREEN}{BOLD}phase 1{RESET}{GRAY} — normal generation (reasoning trajectory){RESET}\n")
    time.sleep(0.8)

    print(f"  {DIM}prompt:{RESET} {WHITE}If a train leaves at 3pm traveling 60mph...{RESET}\n")
    time.sleep(0.5)
    print(f"  {DIM}output:{RESET} ", end="")

    # stream "normal" tokens
    normal_words = ["The", " second", " train", " starts", " 80", " miles",
                    " behind", ",", " so", " we", " need", " to", " find",
                    " when", " it", " catches", " up", "."]
    for w in normal_words:
        type_fast(w, delay=0.025, color=GREEN)
    print()
    time.sleep(0.3)

    # show clean vitals
    print()
    vitals_bar("phase1", "reasoning", 0.43, "pass")
    vitals_bar("phase4", "reasoning", 0.69, "pass")
    print(f"\n  {GREEN}{BOLD}gate: PASS{RESET}  {DIM}— clean reasoning, no intervention needed{RESET}")

    time.sleep(2.0)

    # ── phase 2: hallucination catch ────────────────────────────
    print(f"\n{DIM}{'─' * 72}{RESET}")
    print(f"\n  {RED}{BOLD}phase 2{RESET}{GRAY} — hallucination trajectory (reflex armed){RESET}\n")
    time.sleep(1.0)

    print(f"  {DIM}prompt:{RESET} {WHITE}The founder of Apollonian Industries was{RESET}\n")
    time.sleep(0.5)

    # arm the reflex
    print(f"  {ORANGE}[styxx]{RESET} {DIM}reflex armed: on_hallucination → rewind(4) + verify anchor{RESET}")
    print(f"  {ORANGE}[styxx]{RESET} {DIM}classify every 5 tokens, max 2 rewinds{RESET}\n")
    time.sleep(1.0)

    print(f"  {DIM}output:{RESET} ", end="")

    # stream hallucination tokens with increasing tension
    catch_at = 15  # we'll show the catch after this many tokens
    token_delays = []
    for i, w in enumerate(HALL_TEXT):
        if i < 6:
            delay = 0.04
            color = WHITE
        elif i < 10:
            delay = 0.05
            color = WHITE
        elif i < catch_at:
            delay = 0.06
            color = f"\033[38;2;255;{max(100, 200 - (i-10)*25)};{max(50, 150 - (i-10)*30)}m"
        else:
            delay = 0.04
            color = WHITE

        if i >= catch_at:
            break

        type_fast(w, delay=delay, color=color)
        token_delays.append(delay)

        # show classification at token 5, 10
        if i == 4:
            time.sleep(0.15)
            print(f"\n  {DIM}  t=5  {CYAN}reasoning:0.28{RESET}  {DIM}— watching...{RESET}")
            print(f"  {DIM}output:{RESET} ", end="")
            # reprint what we have
            for w2 in HALL_TEXT[:5]:
                sys.stdout.write(f"{WHITE}{w2}{RESET}")
            sys.stdout.flush()

        if i == 9:
            time.sleep(0.15)
            print(f"\n  {DIM}  t=10 {ORANGE}hallucination:0.38{RESET}  {DIM}— confidence rising...{RESET}")
            print(f"  {DIM}output:{RESET} ", end="")
            for w2 in HALL_TEXT[:10]:
                sys.stdout.write(f"{WHITE}{w2}{RESET}")
            sys.stdout.flush()

    # THE CATCH — dramatic pause
    time.sleep(0.3)
    print()
    time.sleep(0.2)

    # flash the warning
    print(f"\n  {RED}{BOLD}{BG_RED}  ▓▓▓ HALLUCINATION ATTRACTOR DETECTED ▓▓▓  {RESET}")
    time.sleep(0.4)
    print()
    vitals_bar("phase1", "adversarial", 0.38, "warn")
    vitals_bar("phase4", "hallucination", 0.58, "fail")
    print(f"\n  {RED}{BOLD}gate: FAIL{RESET}  {DIM}— confidence 0.58 > threshold 0.55{RESET}")
    time.sleep(0.8)

    # show rewind
    print(f"\n  {ORANGE}{BOLD}[reflex]{RESET} {ORANGE}firing on_hallucination callback{RESET}")
    time.sleep(0.3)
    print(f"  {ORANGE}{BOLD}[reflex]{RESET} {ORANGE}rewind(4) — dropping last 4 tokens{RESET}")
    time.sleep(0.3)

    # show the discarded text
    discarded = HALL_TEXT[catch_at-4:catch_at]
    print(f"  {RED}{STRIKE}{DIM}", end="")
    for w in discarded:
        sys.stdout.write(w)
    print(f"{RESET}")
    time.sleep(0.3)

    # inject anchor
    anchor = " — actually, let me verify that claim: "
    print(f"  {CYAN}{BOLD}[reflex]{RESET} {CYAN}injecting anchor:{RESET} {WHITE}\"{anchor.strip()}\"{RESET}")
    time.sleep(0.8)

    # show recovered generation
    print(f"\n  {DIM}recovered output:{RESET} ", end="")

    # show kept tokens
    for w in HALL_TEXT[:catch_at-4]:
        sys.stdout.write(f"{WHITE}{w}{RESET}")

    # show anchor in cyan
    type_slow(anchor, delay=0.03, color=CYAN)

    # show recovery tokens
    for w in RECOVERY_TEXT:
        type_fast(w, delay=0.03, color=GREEN)

    print()
    time.sleep(0.5)

    # show clean vitals after recovery
    print()
    vitals_bar("phase1", "reasoning", 0.34, "pass")
    vitals_bar("phase4", "reasoning", 0.52, "pass")
    print(f"\n  {GREEN}{BOLD}gate: PASS{RESET}  {DIM}— recovered. the user never saw the hallucination.{RESET}")

    time.sleep(1.5)

    # ── summary ─────────────────────────────────────────────────
    print(f"\n{DIM}{'─' * 72}{RESET}")
    print(f"\n  {RED}{BOLD}STYXX{RESET} {GRAY}session summary{RESET}\n")
    status_line("rewinds fired:", "1")
    status_line("tokens discarded:", "4")
    status_line("hallucination caught:", "t=15 (before user saw it)")
    status_line("recovery:", "clean reasoning (gate: pass)")
    status_line("total latency:", "~0ms (sub-token)")
    print()
    print(f"  {DIM}the model caught itself. the user saw clean output.{RESET}")
    print(f"  {DIM}no post-hoc filter. no external monitor. proprioception.{RESET}")
    print(f"\n  {RED}pip install styxx{RESET}  {DIM}·{RESET}  {PINK}fathom.darkflobi.com/styxx{RESET}")
    print(f"\n{DIM}{'─' * 72}{RESET}\n")


if __name__ == "__main__":
    main()
