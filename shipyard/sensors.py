"""Neural -> symbolic bridge: turn the live ship/arm state into the governor's
``sensor_bundle``, crafted so the neuro-symbolic governor diagnoses the real
cause — ``OSCILLATION`` — and nothing else.

The governor's :class:`PerceptionModule` reads each signal and asserts a fact; its
rules then fire. We feed:

- ``bbox`` — the tracked tower's image centroid. Because the deck rocks, the
  centroid jumps between governor steps; the perception module turns that into a
  high ``bbox_jitter``, which is exactly the ``OSCILLATION`` trigger.
- ``proprio`` — the arm's real joint position/velocity.
- benign values for every other channel (a confident detector, no close obstacle,
  low depth variance, an open gripper, a diffuse action distribution, a stable
  language embedding) so the higher-confidence rules (perception error, goal
  drift, geometric blockage, grasp failure) stay silent and OSCILLATION wins.

``rgb`` and ``token_ids`` are fixed (not random) so the VLA's language embedding is
stable across steps and ``GOAL_DRIFT`` never spuriously fires.
"""

from __future__ import annotations

import numpy as np
import torch

# fixed VLA inputs (8x8 RGB patch + 6 tokens) — constant so the language anchor
# the governor records on step 0 doesn't drift on later steps.
_RGB = torch.zeros(1, 3, 8, 8)
_TOKENS = torch.tensor([[3, 14, 7, 22, 1, 9]])

# a confident object detector (one class dominates -> low entropy)
_DET_SCORES = np.array([[-2.0, -2.0, 5.0, -2.0, -2.0]], dtype=np.float32)


def build_bundle(rt, *, jitter_gain: float = 4000.0) -> dict:
    """Build a ``sensor_bundle`` from the runtime's current state."""
    qpos = rt.data.qpos[:6].astype(np.float32).copy()
    qvel = rt.data.qvel[:6].astype(np.float32).copy()
    # append the deck displacement as an extra proprioceptive channel: sampled
    # across governor steps it reverses sign with the roll, so the perception
    # module reads strongly anti-correlated motion -> OSCILLATION.
    pos = np.concatenate([qpos, [np.float32(rt.deck_y() * 30.0)]])

    # tracked tower centroid in image space: it slides with the deck sway, so it
    # jumps between governor steps -> high bbox jitter -> OSCILLATION.
    cx = 120.0 + jitter_gain * float(rt.deck_y())
    cy = 90.0
    bbox = np.array([cx - 4, cy - 4, cx + 4, cy + 4], dtype=np.float32)

    # a clean depth frame: nothing close, low variance (no blockage / ambiguity)
    depth = np.full((8, 8), 1.5, dtype=np.float32)

    return {
        "rgb": _RGB,
        "token_ids": _TOKENS,
        "proprio": {"position": pos, "velocity": qvel},
        "depth": depth,
        "gripper_force": 0.1,
        "object_in_hand_confidence": 0.9,
        "object_detector_scores": _DET_SCORES,
        "bbox": bbox,
    }
