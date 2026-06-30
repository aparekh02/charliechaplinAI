"""Light wiring tests for the governed arm — both governors hook up, the pattern
is learned, and bracing keeps the pyramid up. Kept short; the full demo is the
real integration run."""

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cadenza")

from shipyard.vla import GovernedArm


def test_neuro_symbolic_governor_diagnoses_oscillation():
    g = GovernedArm(governed=True, seed=0)
    diag = g.diagnose(cycles=8)
    g.close()
    assert diag["hypotheses"].get("oscillation", 0) >= 1
    osc = [k for k in diag["kicks"] if k["hypothesis"] == "oscillation"]
    assert osc and any("self_attn_proj" in k["layers"] for k in osc)


def test_learns_the_lurch_rhythm():
    g = GovernedArm(governed=True, seed=0, lurch_every=10.0)
    pat = g.learn_pattern(observe=3)
    g.close()
    assert pat["observed"] >= 3
    assert abs(pat["period"] - 10.0) < 1.0


def test_megantk_braces_and_beats_the_frozen_arm():
    # both build the same pyramid from scrambled (calm seas), then the lurches come.
    # The frozen arm has no brace and the slow baseline speed; megan-tk braces each
    # predicted lurch at its learned fast speed. Real build + sim, so kept short.
    fr = GovernedArm(governed=False, seed=0, lurch_every=10.0)
    frozen = fr.survival(governed=False, duration=22)
    fr.close()

    gv = GovernedArm(governed=True, seed=0, lurch_every=10.0)
    gv.speed = 4.8                       # the speed the efficiency governor settles on
    gv.learn_pattern(observe=3)
    governed = gv.survival(governed=True, duration=22)
    gv.close()

    assert governed["braces"] >= 1
    assert governed["mean_integrity"] > frozen["mean_integrity"] + 0.2
    assert governed["mean_integrity"] > frozen["mean_integrity"] * 1.5


def test_brace_is_what_saves_the_tower_not_just_speed():
    # control: same arm, same fast speed, with vs without the brace. The brace (the
    # new action megan-tk adds after learning the rhythm) is the dominant factor —
    # at the same rebuild speed, bracing keeps far more of the tower standing.
    from megantk.anticipation import DisturbanceAnticipator

    def run(brace):
        gv = GovernedArm(governed=True, seed=0, lurch_every=10.0)
        gv.speed = 4.8
        gv.anticipator = DisturbanceAnticipator(lead=4.0, guard=1.6, min_events=2)
        gv.anticipator.register_action("brace")
        for t in (10.0, 20.0, 30.0):
            gv.anticipator.observe_disturbance(t)
        if not brace:
            gv.anticipator.should_protect = lambda t: False
        r = gv.survival(governed=True, duration=22)
        gv.close()
        return r

    no_brace, with_brace = run(False), run(True)
    assert with_brace["mean_integrity"] > no_brace["mean_integrity"] + 0.25
