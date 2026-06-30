# charliechaplinAI

**Building pyramids on a moving ship** — a 6-joint arm, real MuJoCo physics, and
[`megan-tk`](../megantk) keeping the tower standing as the deck rocks.

A Cadenza 6-axis arm is bolted to a ship's deck — one solid plank, arm and table
and all, riding the sea. **14 colored blocks** start strewn across the table,
disconnected, and the arm builds them into a three-tier pyramid: a **3×3** base
(9), a **2×2** middle (4), and **1** capstone on top, bottom-up.

The deck sways gently the whole time, and every so often it **lurches hard** — a
big swell hits the hull. That lurch is real physics: every loose block gets thrown,
the exposed upper rows tumble off (several at once, sometimes a base block too),
while the bolted-down base mostly rides it out. The arm runs a frozen VLA that
knows how to stack — but on a rocking ship it can't rebuild fast enough between
lurches, and the pyramid spends its life in pieces.

`megan-tk` closes that gap **online, with no retraining**: it learns to rebuild
faster (and diagnoses *why* it was failing), so the pyramid spends its time
standing instead of scattered.

```
            [#]              tier 2  (capstone — first thrown off by a lurch)
          [#][#]
          [#][#]             tier 1  (2×2)
        [#][#][#]
        [#][#][#]
        [#][#][#]            tier 0  (3×3 base — bolted, rides the lurch)
   /=====================\
   |   arm  #   table    |   <- one solid plank, sways + LURCHES  <====>
   \=====================/
   ~~~~~~~~~~~ sea ~~~~~~~~~~~
```

## How megan-tk keeps it standing

The frozen VLA can build the pyramid, but it can't keep it up: speeding through the
rebuild just topples more blocks, and it has no answer to a lurch it didn't see
coming. megan-tk wins on **control, pattern, and a new action** — not raw speed —
all running for real on a small torch VLA with live sensors and real outcome
feedback:

1. **Neuro-symbolic governor** (`megantk.ns_governor`) — diagnoses the *cause* of
   the failure symbolically: **`OSCILLATION`** (the deck is rocking), and fires a
   **FourierFT** PEFT kick at the VLA's attention weights. *Understand the failure.*

2. **Efficiency governor** (`megantk.efficiency`) — finds the **max *safe* rebuild
   speed** with a residual ratchet: push the speed up until a faster build starts
   knocking blocks over, then hold the last speed that didn't. *Speed, but bounded
   by control — once it's destructive, the previous speed was the max.*

3. **Anticipation governor** (`megantk.anticipation`, added for this) — **learns
   the ship's lurch rhythm** from a few observations (period + phase) and, just
   before each predicted lurch, deploys a **new action: `BRACE`** — a steadying
   hand on the tower so the lurch physically can't throw it. *Learn the pattern,
   add an action, hold the system down.*

The headline metric is **average pyramid integrity** through the lurches. The
frozen VLA watches each lurch throw the upper rows off (**~52%**); megan-tk learns
the beat and braces every lurch, so the pyramid stays **standing the whole time
(~100%)** — a **+40-point** swing.

## Run it

Needs the cadenza venv (MuJoCo + torch + the Cadenza arm):

```bash
/Users/akshparekh/Documents/cadenza/.venv/bin/python run_demo.py
# options: --episodes N  --duration S  --no-render  --out DIR
```

It runs four acts and writes a narrative video to `out/`:

- **Act 1** — the efficiency governor finds the max *safe* rebuild speed (pushes
  until a faster build topples blocks, then backs off).
- **Act 2** — the neuro-symbolic governor diagnoses `OSCILLATION` and kicks
  FourierFT on the attention layer.
- **Act 3** — the anticipation governor watches a few lurches and **learns the
  rhythm** (period + phase), registering the `BRACE` action.
- **Act 4** — the payoff under sway + hard lurches: **frozen VLA** (collapses) vs
  **megan-tk** (learns the rhythm, braces each lurch, stays standing).

Outputs: `results.json`, `integrity.png` (the two integrity curves), and
`demo.mp4` — a 720p story cut: *intro → Regular VLA (can't keep up) → With
megan-tk (rebuilds and holds) → outro*, with a live integrity HUD.

## What's real, and the one modeling abstraction

Real: the MuJoCo physics, the Cadenza 6-DoF arm and its damped-least-squares IK on
the **moving (mocap) plank**, the gentle sway, the **hard lurches** (every loose
block is freed and gets a real inertial kick, then MuJoCo resolves the actual
tumbling, collisions and falls — several blocks at once), both `megan-tk` governors
(genuine diagnosis, PEFT kicks, commit/revert trials, efficiency memory), and the
outcome.

Abstraction: a block that lands within tolerance of its slot **locks in** (welds to
the deck and rides the ship rigidly) instead of being balanced by friction alone,
and between lurches the bolted base stays locked. This is a deliberate stand-in for
contact-rich interlocking — balancing free cubes on a continuously moving base is
intractably noisy and would swamp the thing the demo is about (`megan-tk` adapting
online). The arm, IK, sway, the lurch physics, and the governors are all real.

## Layout

```
shipyard/
  pyramid_plan.py    14-block layout, tier weights, deck-frame integrity metric
  oscillator.py      the ship's rhythm — gentle sway + scheduled hard lurches
  scene_builder.py   generates the ship MJCF (mocap plank, arm+table+14 free blocks)
  ship_runtime.py    live MuJoCo session: IK on the moving base, pick/place, lock/lurch
  sensors.py         scene -> sensor_bundle the governor reads as OSCILLATION
  vla.py             PyramidVLA + GovernedArm (ties in both governors + the policy)
  overlay.py         the integrity HUD drawn on the video
  demo.py            the three-act demo + metrics + narrative MP4
run_demo.py          launcher
tests/               geometry/metric/oscillator (pure) + scene/runtime/diagnosis (mujoco)
```
