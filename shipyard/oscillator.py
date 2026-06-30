"""The ship's motion: a gentle continuous sway, punctuated by sudden hard lurches.

Two regimes:

- **Sway** — a small, slow side-to-side roll the whole plank rides. Loose blocks
  sit through it (the friction holds them); it's just the ship breathing.
- **Lurch** — every ``lurch_every`` seconds the deck snaps hard to one side and
  back (a big swell hitting the hull). This is a real, sharp slam of the deck
  actuator (see :meth:`ShipRuntime.tick`): the acceleration is high enough that
  friction can't hold the loose blocks through it, so MuJoCo itself resolves which
  blocks slip and topple — no inertial-impulse fakery.

``ShipOscillator`` is a pure function of time. ``position`` is the gentle sway the
deck actuator follows; ``lurch_index`` lets the runtime trigger each sharp slam
exactly once (and ``lurch_side`` says which way it throws).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ShipOscillator:
    sway_amp: float = 0.03           # gentle continuous roll (rad)
    sway_period: float = 2.4         # seconds per sway
    lurch_every: float = 6.0         # a hard lurch roughly this often (s); 0 disables
    lurch_amp: float = 0.24          # how far the deck rolls on a lurch (rad)
    lurch_dur: float = 0.6           # how long the sharp roll lasts (s)

    # -- sway --------------------------------------------------------------------
    def _sway(self, t: float) -> float:
        return self.sway_amp * math.sin(2.0 * math.pi * t / self.sway_period)

    # -- the deck command --------------------------------------------------------
    def position(self, t: float) -> float:
        # just the gentle sway; the hard lurch is a sharp deck slam driven by the
        # runtime (see ShipRuntime.tick), not a smooth term here.
        return self._sway(t)

    def velocity(self, t: float) -> float:
        w = 2.0 * math.pi / self.sway_period
        return self.sway_amp * w * math.cos(w * t)

    def lurch_index(self, t: float) -> int | None:
        """The index of the lurch just starting at ``t`` (for one-shot impulses),
        else ``None``. A lurch 'starts' as ``t`` crosses its centre time."""
        if self.lurch_every <= 0:
            return None
        idx = round(t / self.lurch_every)
        if idx <= 0:
            return None
        return idx if abs(t - idx * self.lurch_every) < 0.012 else None

    def lurch_side(self, idx: int) -> float:
        return 1.0 if (idx % 2) else -1.0
