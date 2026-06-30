"""Collect behaviour-cloning demos on the real ship and train the PyramidVLA policy.

Run:  PYTHONPATH=. python train_policy.py
Writes the trained weights to shipyard/assets/policy.pt.
"""
from pathlib import Path

import numpy as np

from shipyard.policy import collect_demos, train_policy

OUT = Path(__file__).resolve().parent / "shipyard" / "assets" / "policy.pt"


def main():
    print("collecting expert demos on the rocking ship (real physics)...")
    data = collect_demos(seeds=(0, 1, 2, 3, 4), duration=24.0)
    n = len(data["mode"])
    counts = {m: int((data["mode"] == i).sum()) for i, m in enumerate(["build", "brace", "wait"])}
    print(f"  collected {n} (obs, action) samples  modes={counts}")
    print("behaviour-cloning the transformer VLA...")
    train_policy(data, epochs=400, lr=2e-3, path=str(OUT))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
