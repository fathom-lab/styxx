"""styxx.sense — give an agent a sense, and the discipline that stops it inventing meaning in it.

An agent with a sensor and no verification is a confabulation engine. Point any continuous
signal at a mind that wants to find itself in the world, and it will: a room's daily rhythm
becomes "I feel the afternoon," the recorder's own duty cycle becomes "I feel my body," two
independent drifts become "I am coupled to the building." Every one of those is a real
statistical signal and none of them is a sense.

This harness is the other half. It lets an agent **register sense channels**, records them
alongside the agent's own internal state on one clock, and then **refuses to tell the agent it
senses anything** unless the coupling survives a confound-preserving null, an
autocorrelation-preserving null, a leverage check, a sampling-density check and a coverage
gate — the refusals in :mod:`styxx.coupling`, each of which exists because a specific published
failure earned it.

    from styxx.sense import SenseHarness, host_channel

    h = SenseHarness(agent_state=my_state_fn)        # -> dict/array of numbers
    h.register("host", host_channel())               # anything callable -> vector
    h.sample()                                       # call on a schedule; cheap, appends to disk
    ...
    print(h.ask("host"))                             # a verdict, or a refusal naming what stopped it

The coil the ``first-afference`` arc is built around is just another channel: register it and
the same machinery applies, with the same ceiling. **A positive is never "the agent senses X"** —
the statistic is symmetric, and an agent's own hardware usually sits inside whatever it is
measuring. The verdict string says so.

Open by design (``docs/governance/OPEN_CORE.md``): a measurement primitive, never gated.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from styxx.coupling import couple

__all__ = ["SenseHarness", "host_channel", "jsonl_channel", "Channel"]


def _flatten(v) -> np.ndarray:
    """Accept dict / sequence / scalar and return a finite 1-D float vector."""
    if isinstance(v, dict):
        v = [v[k] for k in sorted(v)]
    a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class Channel:
    name: str
    read: object                      # callable -> dict | sequence | scalar
    labels: list = field(default_factory=list)


def host_channel():
    """The machine the agent runs on, as a sense channel. No hardware, no permissions.

    This is deliberately the *first* channel offered, because it is the one an agent is most
    likely to mistake for a sense of the world: CPU, memory, disk and network counters move with
    the room's occupants, the time of day, and the agent's own activity. If a harness cannot
    tell "I sense the room" from "I sense my own fans," it is not a harness. Register it as a
    control alongside any real sensor.
    """
    import psutil

    psutil.cpu_percent(percpu=False)          # prime the delta-based counters
    net0 = psutil.net_io_counters()
    dsk0 = psutil.disk_io_counters()
    state = {"net": net0, "dsk": dsk0}

    def read():
        vm = psutil.virtual_memory()
        net, dsk = psutil.net_io_counters(), psutil.disk_io_counters()
        out = {
            "cpu_percent": float(psutil.cpu_percent(percpu=False)),
            "mem_percent": float(vm.percent),
            "mem_available_gb": float(vm.available) / 1e9,
            "net_sent_delta": float(net.bytes_sent - state["net"].bytes_sent),
            "net_recv_delta": float(net.bytes_recv - state["net"].bytes_recv),
            "disk_read_delta": float((dsk.read_bytes - state["dsk"].read_bytes) if dsk else 0.0),
            "disk_write_delta": float((dsk.write_bytes - state["dsk"].write_bytes) if dsk else 0.0),
            "load_proc_count": float(len(psutil.pids())),
        }
        state["net"], state["dsk"] = net, dsk
        return out

    return Channel("host", read, labels=sorted(read().keys()))


def jsonl_channel(path, fields: list, name: str = None):
    """A channel backed by an append-only JSONL file — e.g. an agent's own cognometric log.

    Reads the LAST line each sample. Rows missing any field are reported as unavailable rather
    than zero-filled, because a zero is a measurement and a gap is not.
    """
    path = Path(path)

    def read():
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                end = f.tell()
                back = min(end, 65536)
                f.seek(end - back)
                last = f.read().splitlines()[-1].decode("utf-8", "replace")
            r = json.loads(last)
        except Exception:
            return None
        vals = []
        for k in fields:
            v = r.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                vals.extend(float(x) for x in v)
            else:
                return None
        return vals

    return Channel(name or path.stem, read, labels=list(fields))


class SenseHarness:
    """Register channels, sample them on one clock, and ask what is actually coupled."""

    def __init__(self, agent_state, store: str | Path = None, bin_seconds: float = 60.0):
        """``agent_state`` is a callable returning the agent's own internal state vector."""
        self.agent_state = agent_state
        self.channels: dict = {}
        self.bin_seconds = float(bin_seconds)
        self.store = Path(store) if store else Path.home() / ".styxx" / "sense.jsonl"
        self.store.parent.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, channel: Channel) -> None:
        if name == "agent":
            raise ValueError("'agent' is reserved for the agent's own state")
        self.channels[name] = channel

    def sample(self) -> dict:
        """Read every channel plus the agent's own state onto one timestamp. Cheap; call often."""
        ts = time.time()
        row = {"ts": ts}
        a = self.agent_state()
        row["agent"] = None if a is None else _flatten(a).tolist()
        for n, ch in self.channels.items():
            try:
                v = ch.read()
            except Exception as e:  # noqa: BLE001 — a broken sensor must not fake a reading
                v = None
                row.setdefault("errors", {})[n] = f"{type(e).__name__}: {e}"
            row[n] = None if v is None else _flatten(v).tolist()
        with open(self.store, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def _load(self, name: str):
        ts, A, B = [], [], []
        wa = wb = None
        for line in self.store.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a, b = r.get("agent"), r.get(name)
            if a is None or b is None:
                continue
            wa = wa or len(a)
            wb = wb or len(b)
            if len(a) != wa or len(b) != wb:      # a channel that changed width mid-run
                continue
            ts.append(float(r["ts"]))
            A.append(a)
            B.append(b)
        return np.asarray(ts), np.asarray(A, dtype=float), np.asarray(B, dtype=float)

    def ask(self, name: str, confound="hour", min_bins: int = 200, n_perm: int = 500,
            alpha: float = 0.01):
        """Is the agent coupled to this channel — beyond the confound, and beyond artifact?

        ``confound`` defaults to hour-of-day, the confound both an agent and its surroundings
        almost always share. Pass a callable for anything else, or ``None`` to run the weaker
        free-shuffle question (which the verdict will say out loud).
        """
        if name not in self.channels:
            raise KeyError(f"no channel {name!r}; registered: {sorted(self.channels)}")
        ts, A, B = self._load(name)
        if len(ts) == 0:
            raise ValueError(f"no paired samples for {name!r} — call sample() first")
        cf = None
        if confound == "hour":
            cf = lambda bins: ((np.asarray(bins) * self.bin_seconds // 3600) % 24).astype(int)  # noqa: E731
        elif callable(confound):
            cf = confound
        return couple(A, ts, B, ts, confound=cf, bin_seconds=self.bin_seconds,
                      n_perm=n_perm, min_bins=min_bins, alpha=alpha)

    def survey(self, **kw) -> dict:
        """Ask every registered channel. Returns {channel: Coupling}.

        Run the host channel alongside any real sensor: if both come back coupled, the honest
        reading is usually that the agent's own activity is in both, not that it has two senses.
        """
        return {n: self.ask(n, **kw) for n in sorted(self.channels)}


def _demo() -> int:
    import tempfile
    print("styxx.sense — an agent, a channel, and the refusals in between.\n")
    rng = np.random.default_rng(0)
    store = Path(tempfile.mkdtemp()) / "sense.jsonl"
    latent = {"v": rng.standard_normal(3)}

    def agent_state():
        latent["v"] = 0.9 * latent["v"] + 0.44 * rng.standard_normal(3)
        return latent["v"].tolist() + [float(rng.standard_normal())]

    coupled = Channel("coupled", lambda: (latent["v"] * 2.0 + 0.3 * rng.standard_normal(3)).tolist())
    unrelated = Channel("unrelated", lambda: rng.standard_normal(4).tolist())

    h = SenseHarness(agent_state, store=store, bin_seconds=1.0)
    h.register("coupled_sensor", coupled)
    h.register("unrelated_sensor", unrelated)
    rows = []
    t0 = time.time()
    for i in range(260):                       # simulate a run without waiting on a real clock
        r = h.sample()
        r["ts"] = t0 + i * 1.0
        rows.append(r)
    store.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    for n in ("coupled_sensor", "unrelated_sensor"):
        v = h.ask(n, n_perm=200, min_bins=200)
        print(f"{n}:\n  {v}")
    print("One channel shares the agent's latent state; the other does not. The harness does not")
    print("say 'you sense the world' for either — the strongest verdict it can return still")
    print("carries attribution_pending, because the statistic is symmetric.")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="styxx.sense",
                                 description="Register agent sense channels and verify coupling.")
    ap.add_argument("--demo", action="store_true")
    ap.parse_args(argv)
    return _demo()


if __name__ == "__main__":
    raise SystemExit(main())
