"""Tests for the ship's rhythm — gentle sway plus hard lurches (pure, no mujoco)."""

from shipyard.oscillator import ShipOscillator


def test_sway_starts_at_centre():
    osc = ShipOscillator(sway_amp=0.05, sway_period=4.0, lurch_every=0)
    assert abs(osc.position(0.0)) < 1e-9
    assert abs(osc.position(1.0) - 0.05) < 1e-6     # quarter period -> peak


def test_velocity_sign_tracks_sway():
    osc = ShipOscillator(sway_amp=0.05, sway_period=4.0, lurch_every=0)
    assert osc.velocity(0.0) > 0
    assert osc.velocity(2.0) < 0


def test_position_is_just_the_gentle_sway():
    # the hard lurch is a sharp deck slam driven by the runtime, NOT a term in
    # position() — so position stays the small sway everywhere
    osc = ShipOscillator(sway_amp=0.02, sway_period=2.4, lurch_every=6.0,
                         lurch_amp=0.13)
    assert max(abs(osc.position(t * 0.05)) for t in range(200)) <= 0.0201


def test_lurch_index_fires_once_per_lurch():
    osc = ShipOscillator(lurch_every=6.0)
    assert osc.lurch_index(6.0) == 1
    assert osc.lurch_index(6.05) is None        # not at the centre
    assert osc.lurch_index(12.0) == 2
    assert osc.lurch_index(0.0) is None         # no lurch at t=0
    assert osc.lurch_side(1) == 1.0 and osc.lurch_side(2) == -1.0
