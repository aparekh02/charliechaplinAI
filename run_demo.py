#!/usr/bin/env python3
"""Launch the charliechaplinAI demo (building pyramids on a moving ship, kept
standing by megan-tk).

Must run under the cadenza venv (it has mujoco + torch + the cadenza arm)::

    /Users/akshparekh/Documents/cadenza/.venv/bin/python run_demo.py

Options: ``--episodes N`` (efficiency-learning reps), ``--duration S`` (survival
run length), ``--no-render`` (skip the MP4), ``--out DIR``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shipyard.demo import main

if __name__ == "__main__":
    main()
