"""The sensor bridge must make the neuro-symbolic governor diagnose OSCILLATION
(and nothing else) on the rocking ship. Needs mujoco/torch/cadenza -> skips if
they aren't importable (run under the cadenza venv)."""

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cadenza")

from megantk.ns_governor import KnowledgeBase, PerceptionModule, GovConfig, FailureMode
from shipyard.ship_runtime import ShipRuntime
from shipyard.oscillator import ShipOscillator
from shipyard.sensors import build_bundle


def test_rocking_ship_diagnoses_oscillation():
    rt = ShipRuntime(start="scatter",
                     oscillator=ShipOscillator(sway_amp=0.05, sway_period=5.0,
                                               lurch_every=0))
    rt.reset()
    perc = PerceptionModule(GovConfig)
    kb = KnowledgeBase(GovConfig)
    modes = []
    for _ in range(6):
        rt.settle(1.3)                  # advance roughly a quarter roll
        kb.retract_all()
        perc.extract_facts(build_bundle(rt), kb)
        modes.append(kb.infer().mode)
    # once the rolling history has built up, OSCILLATION should dominate
    assert modes[-3:].count(FailureMode.OSCILLATION) >= 2
    # and the higher-confidence failure modes must stay silent
    for bad in (FailureMode.GEOMETRIC_BLOCKAGE, FailureMode.GRASP_FAILURE,
                FailureMode.PERCEPTION_ERROR, FailureMode.GOAL_DRIFT):
        assert bad not in modes
    rt.close()
