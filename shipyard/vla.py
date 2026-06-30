"""The policy and its governed wrapper — where megan-tk meets the rocking ship.

The arm runs a frozen VLA that knows how to build the pyramid: it picks the next
needed block (base first) and sets it in its slot. On a still deck it would build
the 14-block tower and be done. But the ship rocks, and every few seconds a hard
**lurch** throws the loose blocks about — several tumble off — and the arm has to
rebuild. Whether the pyramid spends its time standing or in pieces comes down to a
race: can the arm rebuild faster than the lurches knock it down?

``megan-tk`` wins that race online, with two governors running for real:

- **The neuro-symbolic governor** (over a small torch VLA) watches the progress
  float (pyramid integrity); when it stalls it diagnoses ``OSCILLATION`` (the deck
  is rocking) and fires a FourierFT PEFT kick at the VLA's attention weights —
  understanding the failure before adapting.
- **The efficiency governor** ratchets the build speed up one committed step at a
  time, keeping the tower intact, so the arm rebuilds faster and faster.

A frozen baseline (``governed=False``) keeps the slow speed and loses the race.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from megantk.vla_integration import VLAWithGovernor, _ToyVLA
from megantk.ns_governor import GovConfig as NSConfig
from megantk.efficiency import EfficiencyGovernor, MemoryBank
from megantk.efficiency.config import GovConfig as _EffBase
from megantk.anticipation import DisturbanceAnticipator

from shipyard.ship_runtime import ShipRuntime
from shipyard.oscillator import ShipOscillator
from shipyard.sensors import build_bundle
from shipyard import pyramid_plan as plan


class PyramidVLA(_ToyVLA):
    """The frozen block-stacking policy the governor adapts. A stand-in torch VLA
    whose layer names carry the keywords the governor's layer-targeting looks for
    (vision, lang/cross, self_attn, depth/spatial, action/head), so an OSCILLATION
    diagnosis lands a FourierFT kick on the attention layer. Swap in a real VLA and
    nothing else changes."""


class EffConfig(_EffBase):
    """Efficiency knobs. The governor works in its native normalised lever
    [0.3, 1.0] (its bottleneck finder assumes speeds <= ~1); we map that onto
    cadenza's place-speed multiplier [CAD_MIN, CAD_MAX] (see ``_cad``)."""
    SPEED_BASELINE = 0.3
    SPEED_MAX = 1.0
    SPEED_STEP = 0.15
    DISTURB_TOL = 0.06
    REVERT_COOLDOWN = 2
    PRUNE_EVERY = 10_000
    # Map the governor's normalised lever [0.3, 1.0] onto the place-speed band where
    # the arm can actually build: below ~3.5x the slow descent resonates with the
    # gentle sway and drifts; from ~3.5-5x it builds the tower perfectly; above ~5.5x
    # a fast placement starts knocking blocks. The governor ratchets up from the floor
    # and reverts at the first speed that topples a block — landing on the max safe one.
    CAD_MIN = 3.5
    CAD_MAX = 6.5


# build order: base (tier 0) first, then up — so the arm always builds bottom-up
BUILD_ORDER = [s.name for s in plan.SLOTS]

# The calm-seas opening build runs at a reliable place-speed for BOTH arms: in flat
# water either arm can stack the 14 blocks. The demo's contrast isn't "who can build"
# — it's "who survives the storm" (the brace) and "who rebuilds fastest" (the learned
# speed in PHASE B). Empirically the build is most reliable around 4x (see the speed
# sweep in tests): much slower drifts under the gentle sway, much faster overshoots.
BUILD_SPEED = 5.0

# The brace is a quick protective stab (hand straight down onto the capstone), so it
# runs at a fixed fast speed — it has to be pressing before the lurch lands. It is
# NOT the careful place speed the efficiency governor tunes for rebuilding.
BRACE_SPEED = 4.0


class GovernedArm:
    def __init__(self, *, governed: bool = True, capture: bool = False,
                 sway_amp: float = 0.04, sway_period: float = 2.2,
                 lurch_every: float = 10.0, lurch_amp: float = 0.24,
                 seed: int = 0, mem_path: str | None = None):
        self.governed = governed
        self.osc = ShipOscillator(sway_amp=sway_amp, sway_period=sway_period,
                                  lurch_every=lurch_every, lurch_amp=lurch_amp)
        self.rt = ShipRuntime(start="scatter", oscillator=self.osc, capture=capture)
        np.random.seed(seed)

        self.ns = VLAWithGovernor(PyramidVLA()) if governed else None
        self.eff = None
        if governed:
            path = mem_path or str(Path(tempfile.gettempdir()) / "shipyard_eff.json")
            Path(path).unlink(missing_ok=True)
            self.eff = EfficiencyGovernor(memory=MemoryBank(path), cfg=EffConfig,
                                          hard_fail=self._fell_off)
            self.eff.start_task([{"name": "build_cycle"}])
        self.speed = EffConfig.SPEED_BASELINE
        self.ns_log: list[dict] = []
        self.anticipator = None
        self.brace_strategy = "pin_cap"      # overwritten by learn_brace's discovery
        self.policy = None                   # the trained VLA, if driving the arm

    # ── primitives ────────────────────────────────────────────────────────────
    @staticmethod
    def _cad(lever: float) -> float:
        c = EffConfig
        frac = (lever - c.SPEED_BASELINE) / (c.SPEED_MAX - c.SPEED_BASELINE)
        return c.CAD_MIN + frac * (c.CAD_MAX - c.CAD_MIN)

    def reseat(self, name: str, speed: float) -> bool:
        self.rt.pick(name, speed=speed)
        return self.rt.place(name, speed=speed)

    def _to_build(self) -> list[str]:
        """Blocks not resting in their slot, base first — what to pick-and-place
        next (a hard lurch throws blocks off; the ones that fell need rebuilding)."""
        return [n for n in BUILD_ORDER if not self.rt.is_placed(n)]

    def _fell_off(self, state: dict) -> bool:
        return any(v[2] < 0.35 for v in state.values())

    def _pyramid_state(self) -> dict:
        """Every block's resting place in the deck frame — the thing a faster build
        must reproduce. If a fast placement knocks a block, its entry moves and the
        efficiency governor sees the disturbance and reverts that speed."""
        loc = self.rt.block_states_local()
        return {n: [float(loc[n][0]), float(loc[n][1]), float(loc[n][2])]
                for n in BUILD_ORDER}

    # ── phase 1: efficiency governor learns the build speed ───────────────────
    def learn_speed(self, episodes: int = 8) -> list[dict]:
        """The recurring job is building the pyramid from scattered blocks. The
        efficiency governor times each rebuild and ratchets the place-speed up while
        the tower still comes out intact (every block lands where it should); the
        first speed that starts knocking blocks is reverted, so it settles on the
        max SAFE speed. Calm seas here — it's the clean, repeatable task to optimise."""
        if not self.governed:
            return []
        rows = []
        sway_save, lurch_save = self.osc.sway_amp, self.osc.lurch_every
        self.osc.sway_amp, self.osc.lurch_every = 0.004, 0.0   # clean, calm rebuild
        for ep in range(episodes):
            cad = self._cad(self.eff.trial_speeds()[0])
            self.rt.reset(); self.rt.settle(0.2); self.rt.home()
            t0 = self.rt.sim_t()
            for name in BUILD_ORDER:                # full build, base first
                self.reseat(name, cad)
            duration = self.rt.sim_t() - t0
            report = self.eff.observe(duration, self._pyramid_state())
            rows.append({"episode": ep, "trial_speed": round(cad, 1),
                         "duration": round(duration, 1),
                         "integrity": round(self.rt.integrity(), 3),
                         "status": report["status"],
                         "speed": round(self._cad(self.eff.task.speeds[0]), 1)})
        self.osc.sway_amp, self.osc.lurch_every = sway_save, lurch_save
        self.speed = float(self._cad(self.eff.task.speeds[0]))
        return rows

    # ── phase 2: neuro-symbolic governor diagnoses the stall ──────────────────
    def diagnose(self, cycles: int = 14) -> dict:
        """Let the ship overwhelm a frozen arm so progress goes flat, and watch the
        neuro-symbolic governor diagnose *why* and kick the VLA."""
        if not self.governed:
            return {}
        from megantk.ns_governor import KnowledgeBase
        NSConfig.STALL_PATIENCE = 2
        NSConfig.GRD_EVAL_WINDOW = 2
        perc = self.ns.governor.perception
        for _ in range(5):              # warm the perception's rolling history
            self.rt.settle(0.9)
            perc.extract_facts(build_bundle(self.rt), KnowledgeBase(NSConfig))
        hyps: dict[str, int] = {}
        kicks = []
        for _ in range(cycles):
            self.rt.settle(0.9)
            _, st = self.ns.forward(build_bundle(self.rt), 0.4)   # flat -> stall
            if st["hypothesis"]:
                hyps[st["hypothesis"]] = hyps.get(st["hypothesis"], 0) + 1
            if st["governor_active"]:
                kl = st["kick_log"]
                kicks.append({"hypothesis": st["hypothesis"], "method": kl["peft_method"],
                              "layers": kl["layers_updated"], "rationale": st["rationale"]})
        self.ns_log = kicks
        return {"hypotheses": hyps, "kicks": kicks}

    # ── phase 3: learn the ship's rhythm (when do the hard lurches come?) ─────
    def learn_pattern(self, observe: int = 4) -> dict:
        """Watch the ship lurch a few times and learn its rhythm, so the brace can
        be in place *before* each future lurch (you can't dodge the first one you've
        never seen — megan-tk learns the beat, then deploys)."""
        if not self.governed:
            return {}
        self.anticipator = DisturbanceAnticipator(lead=4.0, guard=1.6, min_events=2)
        self.rt.reset(); self.rt.settle(0.3)
        seen = 0
        while len(self.anticipator.events) < observe and self.rt.sim_t() < observe * 30:
            self.rt.settle(1.0)
            for t in self.rt.lurch_log[seen:]:
                self.anticipator.observe_disturbance(t)
            seen = len(self.rt.lurch_log)
        a = self.anticipator
        return {"period": round(a.period, 2) if a.period else None,
                "phase": round(a.phase(), 2), "confidence": round(a.confidence(), 2),
                "observed": len(a.events)}

    # ── phase 3b: discover HOW to brace (try each, keep what holds) ───────────
    def learn_brace(self, trials: int = 1) -> dict:
        """The agent isn't told *how* to brace — it tries each steadying-hand
        strategy against a real lurch, records which keeps the tower standing, and
        commits to the winner (the anticipator's action repertoire). A capstone pin
        holds; steadying the wrong tier doesn't — and it finds that out itself."""
        if not self.governed:
            return {}
        if self.anticipator is None:
            self.anticipator = DisturbanceAnticipator(lead=4.0, guard=1.6, min_events=2)
        a = self.anticipator
        save_amp, save_every = self.osc.lurch_amp, self.osc.lurch_every
        rows = []
        for strat in self.rt.BRACE_STRATEGIES:
            a.register_action(strat)
            for _ in range(trials):
                held = self._brace_trial(strat)
                a.record_outcome(strat, saved=held > 0.85)
                rows.append({"strategy": strat, "held": round(held, 3)})
        self.osc.lurch_amp, self.osc.lurch_every = save_amp, save_every
        self.brace_strategy = a.best_action() or "pin_cap"
        return {"trials": rows, "best": self.brace_strategy,
                "rates": {s: round(a.actions[s].success_rate, 2)
                          for s in self.rt.BRACE_STRATEGIES}}

    def _brace_trial(self, strategy: str) -> float:
        """One calibration trial: stand a fresh tower, deploy ``strategy``, hit it
        with a real lurch, and report how much of the tower stayed standing."""
        rt = self.rt
        self.osc.lurch_every = 0.0                       # we trigger the lurch by hand
        rt.reset(); rt.settle(0.2)
        for s in plan.SLOTS:
            rt.set_in_slot(s.name)
        rt.settle(0.5); rt.home()
        rt.brace_engage(strategy=strategy, speed=BRACE_SPEED)
        rt.brace_hold(0.3)
        rt.force_lurch(side=1.0)                          # a real sharp roll, now
        t0 = rt.sim_t()
        while rt.sim_t() < t0 + self.osc.lurch_dur + 0.8:
            rt.brace_hold(0.3)
        held = rt.integrity()
        rt.brace_release(speed=BRACE_SPEED)
        return held

    # ── phase 4: the payoff — build & keep the pyramid standing on the ship ──
    def survival(self, governed: bool, duration: float = 60.0, overlay=None) -> dict:
        """Start from SCRAMBLED blocks and build the pyramid on the rocking ship.
        A **frozen** arm builds, but every hard lurch throws its progress off and it
        never gets ahead. A **governed** arm runs megan-tk: it learned the lurch
        rhythm, so just before each predicted hit it deploys the new BRACE action —
        a steadying hand that holds the stack down — then keeps building. The blocks
        are real free bodies that ride the sway and tumble when thrown."""
        if overlay is not None:
            self.rt._overlay = overlay
        self.rt.reset(); self.rt.settle(0.4); self.rt.home()   # blocks start scattered

        antic = self.anticipator if governed else None
        if governed and antic is None:
            antic = self.anticipator = DisturbanceAnticipator(lead=4.0, guard=1.6,
                                                              min_events=2)
        # if a trained VLA is loaded, IT decides every action (build/brace/wait,
        # which block, which brace) — no scripted planner.
        if governed and self.policy is not None:
            return self._survival_policy(antic, duration)
        speed = self.speed if governed else 1.0
        series, braces, seen, bracing = [], 0, 0, False

        def sample():
            series.append((round(self.rt.sim_t(), 2), round(self.rt.integrity(), 3)))

        # PHASE A — the arm builds the whole pyramid from the scrambled blocks for
        # real (real grasp, real placement). Calm seas while building (a tight stack
        # can't be built while the deck heaves), no lurches yet — same for both arms.
        full_sway, lurch_save = self.osc.sway_amp, self.osc.lurch_every
        self.osc.sway_amp, self.osc.lurch_every = 0.004, 0.0
        for name in BUILD_ORDER:
            self.reseat(name, speed=BUILD_SPEED)      # both arms build in calm seas
            sample()
        # ...then the seas pick up: the sway grows to its full height (the built
        # tower rides it), and the hard lurches begin.
        self.rt.ramp_sway(full_sway, 1.5)
        self.osc.lurch_every = lurch_save
        build_end = self.rt.sim_t()

        # PHASE B — the lurches come. Frozen just rebuilds (and loses ground each
        # hit); governed braces every predicted lurch, then keeps building.
        while self.rt.sim_t() < build_end + duration:
            if antic is not None:
                for t in self.rt.lurch_log[seen:]:
                    antic.observe_disturbance(t)
                seen = len(self.rt.lurch_log)

            if antic is not None and antic.should_protect(self.rt.sim_t()):
                if not bracing:
                    # the brace is a fast protective stab — get the hand down on the
                    # tower before the lurch lands, independent of the (residual-
                    # bounded) place speed used for careful rebuilding.
                    self.rt.brace_engage(strategy=self.brace_strategy,
                                         speed=BRACE_SPEED); bracing = True; braces += 1
                self.rt.brace_hold(0.5)
            else:
                if bracing:
                    self.rt.brace_release(speed=BRACE_SPEED); bracing = False
                todo = self._to_build()
                if todo:
                    self.reseat(todo[0], speed=speed)     # rebuild a thrown block
                else:
                    self.rt.settle(0.4)
            sample()
        if bracing:
            self.rt.brace_release()

        # headline metric = the survival phase (after the build), time-weighted
        surv = [(t, i) for t, i in series if t >= build_end]
        ts = np.array([t for t, _ in surv])
        ys = np.array([i for _, i in surv])
        dt = np.diff(ts, prepend=ts[0])
        total = float(dt.sum()) or 1.0
        return {
            "governed": governed,
            "speed": round(speed, 2),
            "series": series,
            "mean_integrity": round(float((ys * dt).sum() / total), 3),
            "uptime": round(float((dt[ys >= 0.8]).sum() / total), 3),
            "final": series[-1][1],
            "lurches": self.rt.lurches_fired,
            "braces": braces,
            "learned_period": round(antic.period, 2) if antic and antic.period else None,
        }

    def _survival_policy(self, antic, duration: float) -> dict:
        """Survival with the trained VLA in control: at every step it reads the scene
        and emits an action token (build/brace/wait + target + brace strategy), and
        we execute the matching skill. The build, the decision to brace before a
        lurch, which steadying hand to use, and the rebuild after — all the policy."""
        rt, pol = self.rt, self.policy
        series, braces, seen, bracing = [], 0, 0, False

        def sample():
            series.append((round(rt.sim_t(), 2), round(rt.integrity(), 3)))

        # PHASE A — assemble the pyramid bottom-up (the fixed stacking sequence; you
        # always build the tower first). Calm seas, no lurches, same as the baseline.
        full_sway, lurch_save = self.osc.sway_amp, self.osc.lurch_every
        self.osc.sway_amp, self.osc.lurch_every = 0.004, 0.0
        for name in BUILD_ORDER:
            self.reseat(name, speed=BUILD_SPEED)
            sample()
        self.rt.ramp_sway(full_sway, 1.5)
        self.osc.lurch_every = lurch_save
        build_end = rt.sim_t()

        # PHASE B — the storm. Now the VLA is in control: each step it reads the scene
        # and decides BRACE (which steadying hand), BUILD (re-seat a knocked block), or
        # WAIT — anticipating the lurch from the rhythm it learned and protecting the
        # tower itself, with no scripted rule telling it when.
        while rt.sim_t() < build_end + duration:
            for tt in rt.lurch_log[seen:]:
                antic.observe_disturbance(tt)
            seen = len(rt.lurch_log)

            act = pol.act(rt, antic)                       # ← the VLA's decision
            if act["mode"] == "brace":
                if not bracing:
                    rt.brace_engage(strategy=act["strategy"], speed=BRACE_SPEED)
                    bracing = True; braces += 1
                rt.brace_hold(0.5)
            else:
                if bracing:
                    rt.brace_release(speed=BRACE_SPEED); bracing = False
                if act["mode"] == "build" and (todo := self._to_build()):
                    self.reseat(todo[0], speed=self.speed)
                else:
                    rt.settle(0.4)
            sample()
        if bracing:
            rt.brace_release()

        surv = [(t, i) for t, i in series if build_end is not None and t >= build_end]
        if not surv:
            surv = series[-1:]
        ts = np.array([t for t, _ in surv]); ys = np.array([i for _, i in surv])
        dt = np.diff(ts, prepend=ts[0]); total = float(dt.sum()) or 1.0
        return {
            "governed": True, "speed": round(self.speed, 2), "series": series,
            "mean_integrity": round(float((ys * dt).sum() / total), 3),
            "uptime": round(float((dt[ys >= 0.8]).sum() / total), 3),
            "final": series[-1][1], "lurches": rt.lurches_fired, "braces": braces,
            "learned_period": round(antic.period, 2) if antic and antic.period else None,
            "policy_driven": True,
        }

    def close(self):
        self.rt.close()
