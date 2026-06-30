"""Scene-compile + real-physics runtime tests. Needs mujoco/cadenza (skips else)."""

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("cadenza")

import numpy as np

from shipyard.scene_builder import build_scene_xml
from shipyard.ship_runtime import ShipRuntime
from shipyard.oscillator import ShipOscillator
from shipyard import pyramid_plan as plan


def _still():
    return ShipOscillator(sway_amp=0.0, lurch_every=0)


def test_scene_compiles_with_deck_then_arm_joints():
    m = mujoco.MjModel.from_xml_string(build_scene_xml("scatter"))
    # the deck is a real roll hinge (first), then the arm's six
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(7)]
    assert names == ["deck_roll", "j1", "j2", "j3", "j4", "j5", "j6"]
    assert m.nq == 1 + 6 + 2 + 14 * 7                 # deck + arm + grip + blocks


def test_built_tower_is_stable_and_free():
    # set the full pyramid as free bodies; it should stand on its own under sway
    rt = ShipRuntime(start="built",
                     oscillator=ShipOscillator(sway_amp=0.02, sway_period=2.4,
                                               lurch_every=0))
    rt.reset(); rt.settle(0.2)
    for s in plan.SLOTS:
        rt.set_in_slot(s.name)
    rt.settle(3.0)
    assert rt.integrity() > 0.95                       # no welds — held by friction
    assert plan.fallen_off(rt.block_states()) == 0


def test_arm_grasps_a_block_by_friction_no_teleport():
    rt = ShipRuntime(start="scatter", oscillator=_still())
    rt.reset(); rt.settle(0.3); rt.home()
    name = plan.BASE_NAMES[0]
    assert rt.pick(name, speed=2.0)                    # fingers close + lift it
    assert rt.holding() == name                        # really held (read from physics)
    rt.place(name, speed=2.0); rt.settle(0.3)
    assert rt.is_placed(name)                          # rests in its slot


def test_sharp_roll_topples_tower_and_real_brace_holds_it():
    osc = ShipOscillator(sway_amp=0.03, sway_period=2.4, lurch_every=4.0,
                         lurch_amp=0.24, lurch_dur=0.6)
    # without bracing: the sharp roll sends the loose blocks sliding downhill and
    # topples the tower (real contact physics, no impulse)
    rt = ShipRuntime(start="built", oscillator=osc)
    rt.reset(); rt.settle(0.2)
    for s in plan.SLOTS:
        rt.set_in_slot(s.name)
    rt.settle(0.6); intact = rt.integrity()
    rt.settle(4.2)                                     # ride through the t=4 lurch
    assert rt.lurches_fired >= 1
    unbraced = rt.integrity()
    rt.close()

    # with the steadying-hand brace (pin the capstone, ride the deck) it holds
    rt = ShipRuntime(start="built", oscillator=osc)
    rt.reset(); rt.settle(0.2)
    for s in plan.SLOTS:
        rt.set_in_slot(s.name)
    rt.settle(0.6); rt.home()
    rt.brace_engage(strategy="pin_cap", speed=5.0)
    while rt.sim_t() < 4.4:
        rt.brace_hold(0.3)
    braced = rt.integrity()
    rt.close()

    assert intact > 0.95
    assert unbraced < 0.7                              # the roll really topples it
    assert braced > unbraced + 0.3                     # the brace clearly saves it
    assert braced > 0.9
