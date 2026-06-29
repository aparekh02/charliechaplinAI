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
    SPEED_STEP = 0.1
    DISTURB_TOL = 0.05
    REVERT_COOLDOWN = 2
    PRUNE_EVERY = 10_000
    CAD_MIN = 1.0
    CAD_MAX = 8.0


# build order: base (tier 0) first, then up — so the arm always builds bottom-up
BUILD_ORDER = [s.name for s in plan.SLOTS]


class GovernedArm:
    def __init__(self, *, governed: bool = True, capture: bool = False,
                 sway_amp: float = 0.022, sway_period: float = 2.4,
                 lurch_every: float = 6.0, lurch_amp: float = 0.13,
                 lurch_impulse: float = 1.15, seed: int = 0,
                 mem_path: str | None = None):
        self.governed = governed
        self.osc = ShipOscillator(sway_amp=sway_amp, sway_period=sway_period,
                                  lurch_every=lurch_every, lurch_amp=lurch_amp,
                                  lurch_impulse=lurch_impulse)
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

    def _upper_state(self) -> dict:
        bs = self.rt.block_states()
        dy = self.rt.deck_y()
        return {n: [float(bs[n][0]), float(bs[n][1] - dy), float(bs[n][2])]
                for n in plan.UPPER_NAMES}

    # ── phase 1: efficiency governor learns the build speed ───────────────────
    def learn_speed(self, episodes: int = 8) -> list[dict]:
        """The recurring job after a lurch is rebuilding the upper rows on the
        standing base. The efficiency governor speeds that up while it still comes
        out intact (the base is given, the arm restacks the 2x2 + capstone)."""
        if not self.governed:
            return []
        rows = []
        lurch_every = self.osc.lurch_every
        self.osc.lurch_every = 0.0      # the rebuild is the clean, repeatable task
        for ep in range(episodes):
            lever = self.eff.trial_speeds()[0]
            cad = self._cad(lever)
            self.rt.reset(); self.rt.settle(0.2)
            for n in plan.BASE_NAMES:           # base already standing (free bodies)
                self.rt.set_in_slot(n)
            self.rt.settle(0.3)
            self.rt.home()
            t0 = self.rt.sim_t()
            for name in plan.UPPER_NAMES:
                self.reseat(name, cad)
            duration = self.rt.sim_t() - t0
            report = self.eff.observe(duration, self._upper_state())
            rows.append({"episode": ep, "trial_speed": round(cad, 1),
                         "duration": round(duration, 1),
                         "integrity": round(self.rt.integrity(), 3),
                         "status": report["status"],
                         "speed": round(self._cad(self.eff.task.speeds[0]), 1)})
        self.osc.lurch_every = lurch_every
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

    # ── phase 3: the payoff — survival under the lurches at a given speed ──────
    def survival(self, speed: float, duration: float = 45.0, overlay=None,
                 reset: bool = True) -> dict:
        """Build/maintain the pyramid under sway + hard lurches at ``speed``."""
        if overlay is not None:
            self.rt._overlay = overlay
        if reset:
            self.rt.reset(); self.rt.settle(0.3); self.rt.home()
        series = []
        last_lurch = self.rt.lurches_fired
        while self.rt.sim_t() < duration:
            if self.rt.lurches_fired > last_lurch:          # the ship just lurched
                last_lurch = self.rt.lurches_fired
                self.rt.settle(1.2)                         # let thrown blocks settle
            todo = self._to_build()
            if todo:
                self.reseat(todo[0], speed=speed)          # rebuild a fallen block
            else:
                self.rt.settle(0.3)
            series.append((round(self.rt.sim_t(), 2), round(self.rt.integrity(), 3)))
        integ = [i for _, i in series]
        return {
            "speed": round(speed, 2),
            "series": series,
            "mean_integrity": round(float(np.mean(integ)), 3),
            "uptime": round(float(np.mean([i >= 0.8 for i in integ])), 3),
            "final": series[-1][1],
            "lurches": self.rt.lurches_fired,
        }

    def close(self):
        self.rt.close()
