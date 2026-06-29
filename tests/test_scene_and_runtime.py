"""Scene-compile + runtime mechanics tests. Needs mujoco/cadenza (skips otherwise)."""

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("cadenza")

import numpy as np

from shipyard.scene_builder import build_scene_xml
from shipyard.ship_runtime import ShipRuntime
from shipyard.oscillator import ShipOscillator
from shipyard import pyramid_plan as plan


def test_scene_compiles_with_arm_joints_first():
    m = mujoco.MjModel.from_xml_string(build_scene_xml("scatter"))
    assert m.nmocap == 1                              # the deck
    # cadenza's IK slices jnt_range[:6], so the arm's joints must come first
    first6 = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(6)]
    assert first6 == ["j1", "j2", "j3", "j4", "j5", "j6"]
    assert m.nq == 6 + 2 + 14 * 7


def test_scatter_blocks_rest_without_exploding():
    rt = ShipRuntime(start="scatter",
                     oscillator=ShipOscillator(sway_amp=0.0, lurch_every=0))
    rt.reset(); rt.settle(1.0)
    pos = rt.block_states()
    assert plan.fallen_off(pos) == 0
    assert all(0.40 < p[2] < 0.46 for p in pos.values())   # still on the table
    rt.close()


def test_arm_builds_pyramid_from_scatter():
    rt = ShipRuntime(start="scatter",
                     oscillator=ShipOscillator(sway_amp=0.02, sway_period=2.4,
                                               lurch_every=0))
    rt.reset(); rt.settle(0.3); rt.home()
    for name in [s.name for s in plan.SLOTS]:
        rt.pick_led(name, lead=0.0, speed=3.0)
        rt.place_led(name, lead=0.0, speed=3.0)
    assert rt.integrity() >= 0.9
    assert len(rt._seated) >= 13
    rt.close()


def test_lurch_throws_several_upper_blocks_base_survives():
    rt = ShipRuntime(start="built",
                     oscillator=ShipOscillator(sway_amp=0.02, sway_period=2.4,
                                               lurch_every=4.0, lurch_amp=0.13,
                                               lurch_impulse=1.0))
    rt.reset(); rt.settle(0.3)
    for s in plan.SLOTS:
        rt.seat(s.name)
    assert rt.integrity() > 0.95
    rt.settle(5.0)                                  # past the t=4 lurch
    rt.relock_survivors()
    assert rt.lurches_fired >= 1
    # the bolted base mostly rides it out; several upper rows are gone
    base = sum(rt.is_seated(n) for n in plan.BASE_NAMES)
    upper = sum(rt.is_seated(n) for n in plan.UPPER_NAMES)
    assert base >= 7
    assert upper <= 2
    rt.close()
