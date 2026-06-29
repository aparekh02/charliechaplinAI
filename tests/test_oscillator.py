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


def test_lurch_is_a_big_transient_at_its_time():
    osc = ShipOscillator(sway_amp=0.02, sway_period=2.4, lurch_every=6.0,
                         lurch_amp=0.13, lurch_dur=0.6)
    # near a lurch the deck swings far beyond the gentle sway (the transient peaks
    # just off the centre time — it's zero exactly at the centre)
    near = max(abs(osc.position(6.0 + d * 0.05)) for d in range(-4, 5))
    assert near > 0.06
    assert osc.is_lurching(6.1)
    # between lurches it's just the small sway
    assert abs(osc.position(3.0)) < 0.03
    assert not osc.is_lurching(3.0)


def test_lurch_index_fires_once_per_lurch():
    osc = ShipOscillator(lurch_every=6.0)
    assert osc.lurch_index(6.0) == 1
    assert osc.lurch_index(6.05) is None        # not at the centre
    assert osc.lurch_index(12.0) == 2
    assert osc.lurch_index(0.0) is None         # no lurch at t=0
