"""The pyramid: where the 14 blocks belong, and how healthy the tower is.

Pure geometry — no MuJoCo, no torch — so the layout and the health metric can be
unit-tested on their own and imported by the scene generator.

Layout (14 blocks, in the deck/table frame; the deck sways in +y so add the live
deck offset to turn these into world coordinates):

    tier 2:            [#]            1 block   (the prize — fragile, falls first)
    tier 1:         [#][#]            2x2 = 4
                    [#][#]
    tier 0:      [#][#][#]            3x3 = 9
                 [#][#][#]
                 [#][#][#]

Blocks are 0.052 m cubes (half-extent 0.026). Faces touch, so each upper block is
fully supported by the four below it. The table top is at z = 0.40, so resting
centre heights are 0.426 / 0.478 / 0.530.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── Geometry constants (metres, deck/table frame) ────────────────────────────
TABLE_CENTER = (0.5, 0.0)        # table centre in the arm base frame
SURFACE_Z = 0.40                 # table top
BLOCK_HALF = 0.026               # cube half-extent (full edge 0.052)
# A small gap between neighbouring blocks: real free bodies need clearance so the
# arm can set one down next to another without the carried block shoving its
# neighbour. The faces don't touch, so each upper block rests on the inner edges
# of the four below it — stable enough with the high block friction.
_GAP = 0.018
_EDGE = 2.0 * BLOCK_HALF + _GAP  # centre-to-centre spacing in a layer

# Per-tier centre heights: surface + half, then stack by a full block edge each
# (the vertical step is the cube edge, independent of the in-layer gap).
_STACK_H = 2.0 * BLOCK_HALF
_TIER_Z = (SURFACE_Z + BLOCK_HALF,
           SURFACE_Z + BLOCK_HALF + _STACK_H,
           SURFACE_Z + BLOCK_HALF + 2.0 * _STACK_H)

# Upper tiers are worth more: the whole point is to keep the *top* standing, and
# the top block topples first, so losing it should cost the most integrity.
TIER_WEIGHT = (1.0, 2.0, 4.0)


@dataclass(frozen=True)
class Slot:
    name: str        # the block that lives here, e.g. "block07"
    tier: int        # 0 = base, 2 = top
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def _build_slots() -> list[Slot]:
    cx, cy = TABLE_CENTER
    slots: list[Slot] = []
    # tier 0 — 3x3 base, centres at {-edge, 0, +edge}
    for gy in (-_EDGE, 0.0, _EDGE):
        for gx in (-_EDGE, 0.0, _EDGE):
            slots.append(Slot("", 0, cx + gx, cy + gy, _TIER_Z[0]))
    # tier 1 — 2x2 middle, centred over the gaps so each rests on 4 base blocks
    for gy in (-_EDGE / 2, _EDGE / 2):
        for gx in (-_EDGE / 2, _EDGE / 2):
            slots.append(Slot("", 1, cx + gx, cy + gy, _TIER_Z[1]))
    # tier 2 — the single capstone
    slots.append(Slot("", 2, cx, cy, _TIER_Z[2]))
    # name them block00..block13 in build order (bottom first)
    return [Slot(f"block{i:02d}", s.tier, s.x, s.y, s.z) for i, s in enumerate(slots)]


SLOTS: list[Slot] = _build_slots()
N_BLOCKS = len(SLOTS)                       # 14
assert N_BLOCKS == 14, N_BLOCKS

# The 3x3 base (tier 0) is rock-stable under rocking; the upper 5 (2x2 + capstone)
# are what the waves knock off and the arm must keep stacking back.
BASE_NAMES = [s.name for s in SLOTS if s.tier == 0]      # block00..08
UPPER_NAMES = [s.name for s in SLOTS if s.tier > 0]      # block09..13
SLOT_BY_NAME = {s.name: s for s in SLOTS}


def knocked_scatter() -> dict[str, tuple[float, float, float]]:
    """Resting spots on the table for the upper 5 blocks in the 'knocked-down'
    start state — laid flat around the base, all within the arm's workspace so it
    can pick each one up and stack it."""
    z = SURFACE_Z + BLOCK_HALF
    spots = [(0.41, -0.20), (0.41, 0.20), (0.59, -0.20), (0.59, 0.20), (0.50, 0.23)]
    return {name: (x, y, z) for name, (x, y) in zip(UPPER_NAMES, spots)}

# Tolerances for "is this block sitting in its slot?"
XY_TOL = 0.020                              # ~0.4 of a block edge
Z_TOL = 0.018


def slot_world(slot: Slot, deck_y: float) -> np.ndarray:
    """The slot's centre in world coordinates given the live deck sway."""
    return np.array([slot.x, slot.y + deck_y, slot.z], dtype=float)


def integrity(block_pos: dict[str, np.ndarray], deck_y: float = 0.0) -> float:
    """Pyramid health in [0, 1]: tier-weighted fraction of slots correctly filled.

    A slot counts as filled if *some* block sits within tolerance of it in the
    deck frame (slot y shifted by the live ``deck_y``). Matching is greedy by
    distance so one block can't satisfy two slots. Locked-in blocks ride the deck
    exactly, so this is precise; blocks a lurch has thrown out of place score 0.
    """
    pts = {n: np.array([p[0], p[1] - deck_y, p[2]], dtype=float)
           for n, p in block_pos.items()}
    used: set[str] = set()
    total = sum(TIER_WEIGHT[s.tier] for s in SLOTS)
    got = 0.0
    for slot in SLOTS:
        target = np.array([slot.x, slot.y, slot.z])
        best, best_d = None, 1e9
        for name, p in pts.items():
            if name in used:
                continue
            dxy = math.hypot(p[0] - target[0], p[1] - target[1])
            dz = abs(p[2] - target[2])
            if dxy <= XY_TOL and dz <= Z_TOL:
                d = dxy + dz
                if d < best_d:
                    best, best_d = name, d
        if best is not None:
            used.add(best)
            got += TIER_WEIGHT[slot.tier]
    return float(got / total)


def displaced_blocks(block_pos: dict[str, np.ndarray], deck_y: float = 0.0
                     ) -> list[tuple[str, float, Slot]]:
    """Which blocks are out of place, worst first.

    Each block's home slot is fixed by its name (build order). Returns
    ``(name, error, slot)`` for blocks whose deck-frame position is off their
    home slot by more than the tolerance, sorted by error descending. This is
    what the planner consumes to decide what to re-seat next.
    """
    by_name = {s.name: s for s in SLOTS}
    out: list[tuple[str, float, Slot]] = []
    for name, p in block_pos.items():
        slot = by_name.get(name)
        if slot is None:
            continue
        local = np.array([p[0], p[1] - deck_y, p[2]])
        err = float(np.linalg.norm(local - np.array(slot.xyz)))
        if err > XY_TOL:
            out.append((name, err, slot))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def fallen_off(block_pos: dict[str, np.ndarray], floor_z: float = 0.30) -> int:
    """How many blocks have dropped below the table (off the edge to the sea)."""
    return sum(1 for p in block_pos.values() if p[2] < floor_z)


def scatter_positions(seed: int = 0) -> list[np.ndarray]:
    """14 loose, disorganized, non-overlapping rest positions on the table — the
    disconnected start the robot builds the whole pyramid from. They're strewn
    around the *edges* of the workspace (front, back and the two sides), leaving
    the centre — where the pyramid goes — clear, so the arm has somewhere to stack
    without dropping blocks onto each other. Lightly jittered so they look strewn,
    and spaced so they don't interpenetrate at spawn."""
    rng = np.random.default_rng(seed)
    z = SURFACE_Z + BLOCK_HALF
    spots = []
    for x in np.linspace(0.40, 0.60, 5):               # front row
        spots.append((x, -0.20))
    for x in np.linspace(0.40, 0.60, 5):               # back row
        spots.append((x, 0.20))
    for y in (-0.085, 0.085):                          # left edge
        spots.append((0.38, y))
    for y in (-0.085, 0.085):                          # right edge
        spots.append((0.62, y))
    out = []
    for x, y in spots[:N_BLOCKS]:
        jx, jy = rng.uniform(-0.012, 0.012, 2)
        out.append(np.array([x + jx, y + jy, z], dtype=float))
    return out


# Distinct-ish colors for the 14 blocks (RGBA), warm->cool by build order.
def block_colors() -> list[tuple[float, float, float, float]]:
    cols = []
    for i in range(N_BLOCKS):
        h = i / N_BLOCKS
        # simple HSV->RGB at full sat/val
        r, g, b = _hsv(h, 0.75, 0.95)
        cols.append((r, g, b, 1.0))
    return cols


def _hsv(h: float, s: float, v: float):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i % 6]
