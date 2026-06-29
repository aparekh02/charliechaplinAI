"""shipyard — Building pyramids on a moving ship.

A Cadenza 6-axis arm is bolted to a ship's deck with a table in front of it.
Fourteen colored blocks sit on the table in a three-tier pyramid (a 3x3 base, a
2x2 middle, one block on top). The deck rocks side to side in a rhythm — with the
occasional stronger gust (2x) — which slowly skids the blocks and topples the
upper rows.

The arm runs a frozen VLA policy that *already* knows how to build the pyramid.
On a still deck it keeps it standing forever. On a rocking deck it cannot: by the
time a block is lowered the table has swung out from under it, so re-placed blocks
land off-slot and the tower drifts apart.

``megantk`` is what closes that gap, online and without retraining:

- the neuro-symbolic governor (:mod:`megantk.ns_governor`) watches the project's
  progress float (pyramid integrity), and when it stalls it *diagnoses* the cause
  symbolically (``OSCILLATION`` — the deck is rocking), then applies a PEFT
  micro-kick to the VLA's attention weights that teaches it to **lead** the sway,
  placing each block where the table *will* be; and
- the efficiency governor (:mod:`megantk.efficiency`) then ratchets the re-place
  speed up, one committed step at a time, as long as the pyramid keeps standing.

The headline metric is *survival*: how long the pyramid stays standing. The
governed arm keeps it up far longer than the frozen policy alone.

Public pieces:
    build_scene_xml        generate the rocking-ship + pyramid MJCF (scene_builder)
    ShipOscillator         the deck's rhythm (oscillator)
    SLOTS / integrity      the pyramid layout + its health metric (pyramid_plan)
    ShipRuntime            the live MuJoCo session (ship_runtime)
    PyramidVLA / GovernedArm  the policy and its governed wrapper (vla)
"""

from shipyard.pyramid_plan import SLOTS, integrity, scatter_positions
from shipyard.oscillator import ShipOscillator

__all__ = ["SLOTS", "integrity", "scatter_positions", "ShipOscillator",
           "build_scene_xml", "ShipRuntime", "PyramidVLA", "GovernedArm"]


def __getattr__(name):  # lazy imports so pure-python users don't need mujoco/torch
    if name == "build_scene_xml":
        from shipyard.scene_builder import build_scene_xml
        return build_scene_xml
    if name == "ShipRuntime":
        from shipyard.ship_runtime import ShipRuntime
        return ShipRuntime
    if name in ("PyramidVLA", "GovernedArm"):
        from shipyard import vla
        return getattr(vla, name)
    raise AttributeError(name)
