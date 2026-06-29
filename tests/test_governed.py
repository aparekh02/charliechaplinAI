"""Light wiring tests for the governed arm — that both governors hook up and run.
Kept short (mujoco is slow); the full three-act demo is the real integration run."""

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cadenza")

from shipyard.vla import GovernedArm


def test_efficiency_governor_speeds_up_without_toppling():
    g = GovernedArm(governed=True, seed=0)
    rows = g.learn_speed(episodes=3)
    g.close()
    assert len(rows) == 3
    assert rows[0]["status"] == "baseline"
    # rebuilds stay intact and the committed speed never drops below baseline
    assert all(r["integrity"] >= 0.95 for r in rows)
    assert rows[-1]["speed"] >= rows[0]["speed"]
    # at least one speed-up committed (faster, tower still intact)
    assert any(r["status"] == "commit" for r in rows)


def test_neuro_symbolic_governor_diagnoses_oscillation():
    g = GovernedArm(governed=True, seed=0)
    diag = g.diagnose(cycles=8)
    g.close()
    assert diag["hypotheses"].get("oscillation", 0) >= 1
    assert diag["kicks"], "expected at least one PEFT kick"
    # the oscillation repair must target the attention layer
    osc = [k for k in diag["kicks"] if k["hypothesis"] == "oscillation"]
    assert osc and any("self_attn_proj" in k["layers"] for k in osc)


def test_frozen_baseline_runs_a_short_survival():
    g = GovernedArm(governed=False, seed=1)
    res = g.survival(speed=2.0, duration=8.0)
    g.close()
    assert 0.0 <= res["mean_integrity"] <= 1.0
    assert res["series"]
