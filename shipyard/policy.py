"""A small but real **VLA policy** that chooses the arm's actions on the ship.

This replaces the hand-written if/else planner with a learned transformer. It reads
a tokenised view of the scene and outputs *organised action tokens*:

    observation tokens                      action tokens (heads)
    ──────────────────                      ─────────────────────
    [GOAL]   learned goal embedding         mode      : BUILD / BRACE / WAIT
    [SHIP]   integrity, threat, deck roll   target    : which block to (re)place
    [HAND]   gripper / holding / ee height  strategy  : which brace to deploy
    [BLK00..BLK13]  per-block state    ─►  Transformer ─►  three softmax heads

It's a genuine Vision/scene-Language-Action policy — a goal token (the language/
intent), the scene tokens (perception), and tokenised action outputs — just small
and task-specific. It is **trained by behaviour cloning**: we roll out an expert
(build bottom-up; when the learned anticipator says a lurch is imminent and enough
tower is standing, deploy the discovered-best brace; otherwise wait) and fit the
transformer to reproduce its decisions. At inference the policy — not an if/else —
drives the arm.

The low-level motion (IK pick/place/brace) stays in :mod:`ship_runtime`; the policy
selects *which skill and target*, the option-level interface a real VLA exposes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shipyard import pyramid_plan as plan

# action vocabulary -----------------------------------------------------------
MODES = ["build", "brace", "wait"]
MODE_I = {m: i for i, m in enumerate(MODES)}
STRATEGIES = ["pin_cap", "pin_cap_firm", "pin_mid"]   # must match ShipRuntime
STRAT_I = {s: i for i, s in enumerate(STRATEGIES)}

N_BLK = plan.N_BLOCKS
BLK_FEAT = 6          # dx, dy, dz, placed, held, tier/2
SHIP_FEAT = 6         # integrity, frac_unplaced, ttn, should_protect, roll, roll_vel
HAND_FEAT = 4         # grip opening, holding, ee height, _pad


# ── observation encoding (pure function of runtime + anticipator state) ──────
def encode_obs(rt, antic=None) -> dict:
    """Build the token features from the live scene. Returns numpy arrays:
    ``blocks`` (14×BLK_FEAT), ``ship`` (SHIP_FEAT,), ``hand`` (HAND_FEAT,)."""
    loc = rt.block_states_local()
    held_name = rt.holding()
    blocks = np.zeros((N_BLK, BLK_FEAT), np.float32)
    for i, name in enumerate([s.name for s in plan.SLOTS]):
        s = plan.SLOT_BY_NAME[name]
        p = loc[name]
        blocks[i] = [p[0] - s.x, p[1] - s.y, p[2] - s.z,
                     float(rt.is_placed(name)), float(name == held_name), s.tier / 2.0]

    integ = rt.integrity()
    n_unplaced = sum(not rt.is_placed(s.name) for s in plan.SLOTS)
    t = rt.sim_t()
    # the threat only exists once the seas are running (lurches scheduled); during
    # the calm build there is nothing to brace against.
    threat_on = getattr(rt, "osc", None) is not None and rt.osc.lurch_every > 0
    live = antic is not None and antic.learned and threat_on
    ttn = antic.time_to_next(t) if live else 10.0
    ttn = float(np.clip((ttn if ttn is not None else 10.0) / 10.0, 0.0, 1.0))
    protect = float(antic.should_protect(t)) if live else 0.0
    ship = np.array([integ, n_unplaced / N_BLK, ttn, protect,
                     rt.deck_pos(), rt.deck_vel()], np.float32)

    ee_z = float(rt.data.site_xpos[rt._site][2])
    hand = np.array([rt._grip, float(held_name is not None), ee_z - 0.4, 0.0], np.float32)
    return {"blocks": blocks, "ship": ship, "hand": hand}


def expert_action(rt, antic, brace_strategy: str, build_order) -> tuple:
    """The teacher the policy is cloned from. Bottom-up build; brace the moment a
    lurch is imminent and there's a worthwhile tower to protect; else wait."""
    t = rt.sim_t()
    unplaced = [n for n in build_order if not rt.is_placed(n)]
    threat_on = getattr(rt, "osc", None) is not None and rt.osc.lurch_every > 0
    imminent = bool(antic and antic.learned and threat_on and antic.should_protect(t))
    if imminent and rt.integrity() >= 0.55:
        return "brace", 0, STRAT_I[brace_strategy]
    if unplaced:
        return "build", [s.name for s in plan.SLOTS].index(unplaced[0]), 0
    return "wait", 0, 0


# ── the model ────────────────────────────────────────────────────────────────
class PyramidVLA(nn.Module):
    """Transformer over [GOAL, SHIP, HAND, 14×BLOCK] tokens with three action heads.

    Named ``self_attn`` inside so the neuro-symbolic governor's layer-targeting can
    find and kick the attention weights when it diagnoses OSCILLATION."""

    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.goal = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blk_embed = nn.Linear(BLK_FEAT, d)
        self.ship_embed = nn.Linear(SHIP_FEAT, d)
        self.hand_embed = nn.Linear(HAND_FEAT, d)
        self.type_embed = nn.Parameter(torch.randn(4, d) * 0.02)   # goal/ship/hand/blk
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                         batch_first=True, dropout=0.0)
        self.self_attn = nn.TransformerEncoder(enc, layers)        # name: self_attn
        self.mode_head = nn.Linear(d, len(MODES))
        self.target_head = nn.Linear(d, 1)           # per-block token -> a score
        self.strategy_head = nn.Linear(d, len(STRATEGIES))

    def forward(self, blocks, ship, hand):
        # blocks B×14×F, ship B×F, hand B×F
        b = self.blk_embed(blocks) + self.type_embed[3]
        s = self.ship_embed(ship).unsqueeze(1) + self.type_embed[1]
        h = self.hand_embed(hand).unsqueeze(1) + self.type_embed[2]
        g = self.goal.expand(blocks.shape[0], -1, -1) + self.type_embed[0]
        x = torch.cat([g, s, h, b], dim=1)           # B×17×d (GOAL,SHIP,HAND,14×BLK)
        z = self.self_attn(x)
        cls = z[:, 0]                                 # GOAL token summarises the call
        mode_logits = self.mode_head(cls)            # B×3
        target_logits = self.target_head(z[:, 3:]).squeeze(-1)   # B×14 (one per block)
        strategy_logits = self.strategy_head(cls)    # B×3
        return mode_logits, target_logits, strategy_logits


# ── behaviour-cloning data collection ────────────────────────────────────────
def collect_demos(seeds=(0, 1, 2, 3), duration: float = 26.0) -> dict:
    """Roll out the expert on the real ship across seeds, logging (obs, action) at
    every decision: the whole build, then the protect-and-recover under lurches."""
    from shipyard.vla import GovernedArm, BUILD_ORDER, BUILD_SPEED

    B, S, H, M, T, G = [], [], [], [], [], []        # blocks, ship, hand, mode, tgt, strat

    def log(rt, antic, strat):
        o = encode_obs(rt, antic)
        m, tgt, g = expert_action(rt, antic, strat, BUILD_ORDER)
        B.append(o["blocks"]); S.append(o["ship"]); H.append(o["hand"])
        M.append(MODE_I[m]); T.append(tgt); G.append(g)
        return m, tgt, g

    for seed in seeds:
        arm = GovernedArm(governed=True, seed=seed, lurch_every=10.0)
        arm.speed = 4.8
        arm.learn_pattern(observe=3)
        arm.learn_brace(trials=1)
        antic, strat = arm.anticipator, arm.brace_strategy
        rt = arm.rt
        names = [s.name for s in plan.SLOTS]

        # (a) BUILD demos — building the pyramid from scatter (calm seas). Each step
        # the expert says 'build the next block', so the policy learns the build.
        arm.osc.sway_amp, arm.osc.lurch_every = 0.004, 0.0
        rt.reset(); rt.settle(0.4); rt.home()
        for name in BUILD_ORDER:
            log(rt, antic, strat)
            arm.reseat(name, speed=BUILD_SPEED)

        # (b) BRACE / WAIT demos — stand a finished tower, bring the seas up with the
        # lurches, and run the expert: it braces in each predicted window (the brace
        # holds, so the tower stays up) and waits between. This is where the policy
        # sees the protective action it must learn to deploy.
        arm.osc.sway_amp, arm.osc.lurch_every = 0.04, 10.0
        rt.reset(); rt.settle(0.2)
        for s in plan.SLOTS:
            rt.set_in_slot(s.name)
        rt.settle(0.5); rt.home()
        t_end, seen, bracing = rt.sim_t() + duration, 0, False
        while rt.sim_t() < t_end:
            for tt in rt.lurch_log[seen:]:
                antic.observe_disturbance(tt)
            seen = len(rt.lurch_log)
            m, tgt, g = log(rt, antic, strat)
            if m == "brace":
                if not bracing:
                    rt.brace_engage(strategy=STRATEGIES[g], speed=6.0); bracing = True
                rt.brace_hold(0.5)
            else:
                if bracing:
                    rt.brace_release(speed=6.0); bracing = False
                if m == "build":
                    rt.pick(names[tgt], speed=arm.speed); rt.place(names[tgt], speed=arm.speed)
                else:
                    rt.settle(0.4)
        arm.close()

    return {"blocks": np.array(B), "ship": np.array(S), "hand": np.array(H),
            "mode": np.array(M), "target": np.array(T), "strategy": np.array(G)}


def train_policy(data: dict, *, epochs: int = 300, lr: float = 2e-3,
                 path: str | None = None, log_every: int = 50) -> "PyramidVLA":
    """Behaviour-clone the policy on the collected demos. Masked losses: target only
    supervised on BUILD steps, strategy only on BRACE steps."""
    torch.manual_seed(0)
    model = PyramidVLA()
    blocks = torch.tensor(data["blocks"], dtype=torch.float32)
    ship = torch.tensor(data["ship"], dtype=torch.float32)
    hand = torch.tensor(data["hand"], dtype=torch.float32)
    mode = torch.tensor(data["mode"], dtype=torch.long)
    target = torch.tensor(data["target"], dtype=torch.long)
    strat = torch.tensor(data["strategy"], dtype=torch.long)
    is_build = mode == MODE_I["build"]
    is_brace = mode == MODE_I["brace"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        ml, tl, sl = model(blocks, ship, hand)
        loss = ce(ml, mode)
        if is_build.any():
            loss = loss + ce(tl[is_build], target[is_build])
        if is_brace.any():
            loss = loss + ce(sl[is_brace], strat[is_brace])
        loss.backward(); opt.step()
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            with torch.no_grad():
                acc = (ml.argmax(1) == mode).float().mean().item()
            print(f"  ep{ep:03d}  loss {loss.item():.4f}  mode-acc {acc:.3f}")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model


class Policy:
    """Inference wrapper: load the trained VLA and turn a live scene into an action
    (mode, target block name, brace strategy). This is what drives the arm."""

    def __init__(self, model: "PyramidVLA"):
        self.model = model.eval()

    @classmethod
    def load(cls, path: str) -> "Policy":
        m = PyramidVLA()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(m)

    @torch.no_grad()
    def act(self, rt, antic) -> dict:
        o = encode_obs(rt, antic)
        ml, tl, sl = self.model(torch.tensor(o["blocks"])[None],
                                torch.tensor(o["ship"])[None],
                                torch.tensor(o["hand"])[None])
        mode = MODES[int(ml.argmax())]
        names = [s.name for s in plan.SLOTS]
        # choose a target among blocks not yet placed (the policy ranks them)
        scores = tl[0].clone()
        for i, n in enumerate(names):
            if rt.is_placed(n):
                scores[i] = -1e9
        target = names[int(scores.argmax())]
        strategy = STRATEGIES[int(sl.argmax())]
        return {"mode": mode, "target": target, "strategy": strategy,
                "mode_probs": torch.softmax(ml[0], 0).tolist()}
