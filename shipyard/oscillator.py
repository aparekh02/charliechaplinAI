"""The ship's motion: a gentle continuous sway, punctuated by sudden hard lurches.

Two regimes:

- **Sway** — a small, slow side-to-side roll the whole plank rides. Loose blocks
  sit through it (the friction holds them); it's just the ship breathing.
- **Lurch** — every ``lurch_every`` seconds the deck snaps hard to one side and
  back (a big swell hitting the hull). The runtime pairs each lurch with an
  inertial impulse on the loose blocks (see :meth:`ShipRuntime._maybe_lurch`), so
  the tower really gets thrown about — several blocks shift and some tumble off.

``ShipOscillator`` is a pure function of time. ``position`` is what the deck mocap
follows; ``lurch_index`` lets the runtime fire each lurch's impulse exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ShipOscillator:
    sway_amp: float = 0.022          # gentle continuous roll (m)
    sway_period: float = 2.4         # seconds per sway
    lurch_every: float = 6.0         # a hard lurch roughly this often (s); 0 disables
    lurch_amp: float = 0.13          # how far the deck snaps on a lurch (m)
    lurch_dur: float = 0.6           # how long the lurch transient lasts (s)
    lurch_impulse: float = 1.15      # strength of the inertial kick to the blocks

    # -- sway --------------------------------------------------------------------
    def _sway(self, t: float) -> float:
        return self.sway_amp * math.sin(2.0 * math.pi * t / self.sway_period)

    # -- lurch -------------------------------------------------------------------
    def _lurch_centre(self, t: float) -> float:
        return round(t / self.lurch_every) * self.lurch_every if self.lurch_every > 0 else -1e9

    def _lurch(self, t: float) -> float:
        """A fast damped swing centred on each lurch time (the hard hit)."""
        if self.lurch_every <= 0:
            return 0.0
        idx = round(t / self.lurch_every)
        if idx <= 0:
            return 0.0
        tc = idx * self.lurch_every
        dt = t - tc
        if abs(dt) > self.lurch_dur:
            return 0.0
        phase = dt / self.lurch_dur
        side = 1.0 if (idx % 2) else -1.0          # alternate sides, like real swells
        return side * self.lurch_amp * math.sin(2.0 * math.pi * 1.5 * phase) \
            * math.exp(-3.0 * abs(phase))

    def position(self, t: float) -> float:
        return self._sway(t) + self._lurch(t)

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

    def is_lurching(self, t: float) -> bool:
        return abs(self._lurch(t)) > 0.4 * self.lurch_amp
