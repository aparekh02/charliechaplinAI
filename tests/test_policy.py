"""The trained VLA policy: shapes, scene encoding, and that it actually drives the
arm to keep the pyramid standing. Needs torch (+ mujoco/cadenza for the rollout)."""

import pytest

pytest.importorskip("torch")

import numpy as np

from shipyard.policy import (PyramidVLA, encode_obs, expert_action, MODES,
                             STRATEGIES, N_BLK, BLK_FEAT, SHIP_FEAT, HAND_FEAT)


def test_model_emits_organised_action_tokens():
    import torch
    m = PyramidVLA()
    b = torch.zeros(2, N_BLK, BLK_FEAT)
    s = torch.zeros(2, SHIP_FEAT)
    h = torch.zeros(2, HAND_FEAT)
    mode, target, strat = m(b, s, h)
    assert mode.shape == (2, len(MODES))            # build / brace / wait
    assert target.shape == (2, N_BLK)               # one score per block
    assert strat.shape == (2, len(STRATEGIES))      # which brace


@pytest.mark.parametrize("_", [0])
def test_policy_drives_the_arm_to_keep_the_tower_up(_):
    pytest.importorskip("mujoco")
    pytest.importorskip("cadenza")
    from pathlib import Path
    weights = Path(__file__).resolve().parents[1] / "shipyard" / "assets" / "policy.pt"
    if not weights.exists():
        pytest.skip("policy.pt not trained yet (run train_policy.py)")
    from shipyard.vla import GovernedArm
    from shipyard.policy import Policy

    # frozen baseline: no governor, no brace
    fr = GovernedArm(governed=False, seed=0, lurch_every=10.0)
    frozen = fr.survival(governed=False, duration=22); fr.close()

    # the trained VLA in control: it builds, then chooses to brace each lurch
    gv = GovernedArm(governed=True, seed=0, lurch_every=10.0); gv.speed = 4.8
    gv.learn_pattern(observe=3); gv.learn_brace(trials=1)
    gv.policy = Policy.load(str(weights))
    res = gv.survival(governed=True, duration=22); gv.close()

    assert res["policy_driven"] is True
    assert res["braces"] >= 1                                   # the VLA chose to brace
    assert res["mean_integrity"] > frozen["mean_integrity"] + 0.3
    assert res["mean_integrity"] > 0.85                         # tower stays up
