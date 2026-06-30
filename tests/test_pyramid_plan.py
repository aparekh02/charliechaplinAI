"""Pure-geometry tests for the pyramid layout + integrity metric (no mujoco)."""

import numpy as np

from shipyard import pyramid_plan as plan


def test_fourteen_blocks_in_three_tiers():
    assert plan.N_BLOCKS == 14
    tiers = [s.tier for s in plan.SLOTS]
    assert tiers.count(0) == 9 and tiers.count(1) == 4 and tiers.count(2) == 1
    assert len(plan.BASE_NAMES) == 9 and len(plan.UPPER_NAMES) == 5


def test_full_pyramid_scores_one():
    pos = {s.name: np.array(s.xyz) for s in plan.SLOTS}
    assert plan.integrity(pos) == 1.0


def test_integrity_tracks_the_deck_frame():
    # the blocks ride the deck, so a tower swaying with it scores ~1.0 once the
    # live deck offset is supplied; without it the score drops
    for dy in (-0.08, 0.0, 0.08):
        pos = {s.name: np.array([s.x, s.y + dy, s.z]) for s in plan.SLOTS}
        assert plan.integrity(pos, deck_y=dy) > 0.99
        assert plan.is_block_placed(pos, "block13", deck_y=dy)
        if dy:
            assert plan.integrity(pos, deck_y=0.0) < 0.6


def test_losing_capstone_costs_the_most():
    base_only = {s.name: np.array(s.xyz) for s in plan.SLOTS if s.tier == 0}
    full = {s.name: np.array(s.xyz) for s in plan.SLOTS}
    no_top = {s.name: np.array(s.xyz) for s in plan.SLOTS if s.tier < 2}
    assert plan.integrity(base_only) < plan.integrity(no_top) < plan.integrity(full)
    # the single capstone is worth more than any single base block
    assert plan.TIER_WEIGHT[2] > plan.TIER_WEIGHT[0]


def test_displaced_blocks_flags_moved_ones():
    pos = {s.name: np.array(s.xyz) for s in plan.SLOTS}
    pos["block13"] = pos["block13"] + np.array([0.1, 0.0, -0.1])   # capstone knocked off
    disp = plan.displaced_blocks(pos)
    assert disp and disp[0][0] == "block13"


def test_knocked_scatter_covers_upper_blocks():
    sc = plan.knocked_scatter()
    assert set(sc) == set(plan.UPPER_NAMES)
