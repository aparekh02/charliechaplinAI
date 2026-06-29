"""The live MuJoCo session for the rocking ship — IK, motion, grasp, and the deck.

Subclasses cadenza's ``_Runtime`` so the damped-least-squares IK, the top-down
pick/place motion and the table no-go clamp all come for free. What this adds:

- **The deck drives itself every physics step.** A mocap deck doesn't move on its
  own, so :meth:`tick` (passed as the ``render`` callback into every motion) sets
  ``data.mocap_pos`` from the :class:`ShipOscillator` as a function of sim time.
  That single hook means the ship sways during *every* arm motion — including
  headless runs, where cadenza would otherwise never call back.
- **IK that accounts for the moving base.** The arm is bolted to the deck, so the
  scratch sim used for IK is given the live mocap pose; otherwise the solver would
  plan as if the ship were centred and the hand would miss.
- **14 blocks, 14 welds, grasp-nearest** (like robogpt's StackRuntime, widened).
- **Pick / place primitives** plus optional **frame capture** for the demo video.

The lead-compensation that beats the sway is *not* here — the runtime just goes
where it's told. The policy (see :mod:`shipyard.vla`) decides the placement target;
the governor teaches it to aim where the slot *will* be.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import mujoco

from cadenza.arm import (_Runtime, _Q_DOWN, _GRIP_OPEN, _GRIP_CLOSED,
                         _HOVER, _LIFT, _speed_for)

from shipyard import pyramid_plan as plan
from shipyard.scene_builder import build_scene_xml
from shipyard.oscillator import ShipOscillator

# Max pinch->block distance for the closed gripper to grip it. Generous enough to
# absorb the small sway error between the arm reaching and the fingers closing.
_GRASP_REACH = 0.12
# Transit height: carry blocks well above the finished tower (capstone ~0.53) so
# the gripper descends onto each slot from straight above and never sweeps
# sideways across the placed blocks (which would knock them over).
_TRANSIT_Z = 0.70


class ShipRuntime(_Runtime):
    def __init__(self, start: str = "scatter",
                 oscillator: ShipOscillator | None = None, *,
                 hz: float = 50.0, capture: bool = False,
                 capture_every: int = 10, cam=(2.0, 140, -15),
                 width: int = 1280, height: int = 720,
                 block_friction: str | None = None):
        # cadenza's _Runtime loads from a path, so materialise the generated MJCF.
        kw = {} if block_friction is None else {"block_friction": block_friction,
                                                "table_friction": block_friction}
        xml = build_scene_xml(start, **kw)
        tmp = Path(tempfile.gettempdir()) / "shipyard_scene.xml"
        tmp.write_text(xml)
        super().__init__(str(tmp), hz)

        m = self.model
        self._mocap = int(m.body_mocapid[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "deck")])
        self._block_bids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                            for n in plan.SLOT_BY_NAME}
        self._blocks = [self._block_bids[s.name] for s in plan.SLOTS]
        self._welds = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                         f"grasp{i:02d}") for i in range(plan.N_BLOCKS)]
        self._seat_welds = {s.name: mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_EQUALITY, f"seat{i:02d}")
            for i, s in enumerate(plan.SLOTS)}
        self._active_weld = -1

        # free-joint qpos / qvel addresses. A placed block is welded to the deck
        # (rides the ship rigidly); a hard lurch frees every block so real physics
        # throws them about, then survivors near their slot re-lock.
        self._qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")]) for n in self._block_bids}
        self._vadr = {n: int(m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")]) for n in self._block_bids}
        self._start = start

        self.osc = oscillator or ShipOscillator()
        self._t0 = 0.0
        self._last_lurch_idx = -1
        self.lurches_fired = 0
        self._rng = np.random.default_rng(0)

        # frame capture
        self.capture = capture
        self._capture_every = max(1, capture_every)
        self._renderer = None
        self._cam = None
        self._width, self._height, self._cam_spec = width, height, cam
        self.frames: list[np.ndarray] = []
        self._frame_div = 0
        self._overlay = None       # set by the demo to stamp text on frames

    # ── the deck's heartbeat: called after every physics step ────────────────
    def tick(self) -> None:
        t = self.data.time - self._t0
        self.data.mocap_pos[self._mocap][1] = self.osc.position(t)
        self._maybe_lurch(t)
        if self.capture:
            self._frame_div += 1
            if self._frame_div % self._capture_every == 0:
                self.frames.append(self._grab())

    def _maybe_lurch(self, t: float) -> None:
        """At each hard lurch the deck snaps hard to one side (the oscillator's
        ``position`` does that) and the loose blocks get a real inertial kick — the
        higher and more exposed a block, the harder it's thrown. MuJoCo then
        resolves the actual tumbling, collisions and falls (caught by the deck
        rails). The heavy base mostly rides it out; the upper rows go flying — never
        just one block, and sometimes a base block too."""
        idx = self.osc.lurch_index(t)
        if idx is None or idx == self._last_lurch_idx:
            return
        self._last_lurch_idx = idx
        side = self.osc.lurch_side(idx)
        V = self.osc.lurch_impulse
        rng = self._rng
        tier_gain = (0.45, 1.0, 1.3)
        for n in self._block_bids:
            if self.data.xpos[self._block_bids[n]][2] < 0.35:
                continue                                   # already on the deck
            g = tier_gain[plan.SLOT_BY_NAME[n].tier] * V
            v = self._vadr[n]
            self.data.qvel[v:v + 3] = (rng.uniform(-0.25, 0.25) * g,
                                       -side * g * rng.uniform(0.7, 1.1),
                                       0.25 * g)
            self.data.qvel[v + 3:v + 6] = rng.uniform(-2.5, 2.5, 3) * g
        self.lurches_fired += 1

    def start_clock(self) -> None:
        self._t0 = float(self.data.time)

    def deck_y(self, ahead: float = 0.0) -> float:
        return self.osc.position(self.data.time - self._t0 + ahead)

    def deck_vel(self) -> float:
        return self.osc.velocity(self.data.time - self._t0)

    def sim_t(self) -> float:
        return self.data.time - self._t0

    # ── reset / settle ───────────────────────────────────────────────────────
    def reset(self) -> None:
        self.release()
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self._grip = _GRIP_OPEN
        self.data.ctrl[6] = self.data.ctrl[7] = _GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)
        self.start_clock()
        self._last_lurch_idx = -1
        self.lurches_fired = 0
        self._rng = np.random.default_rng(0)

    # ── placement state — blocks are free bodies, held in place by friction ────
    def landing_error(self, name: str) -> float:
        """Deck-frame distance of a block from its home slot."""
        slot = plan.SLOT_BY_NAME[name]
        p = self.data.xpos[self._block_bids[name]].copy()
        dy = self.data.mocap_pos[self._mocap][1]
        return float(np.hypot(p[0] - slot.x, p[1] - (slot.y + dy)))

    def is_placed(self, name: str) -> bool:
        """Geometrically resting in its slot (deck frame). No welds — a block is
        'placed' only as long as real physics keeps it there."""
        slot = plan.SLOT_BY_NAME[name]
        p = self.data.xpos[self._block_bids[name]]
        return (self.landing_error(name) <= plan.XY_TOL
                and abs(p[2] - slot.z) <= plan.Z_TOL)

    def at_rest(self, name: str) -> bool:
        v = self._vadr[name]
        return float(np.linalg.norm(self.data.qvel[v:v + 6])) < 0.5

    def set_in_slot(self, name: str) -> None:
        """Drop a free block straight into its slot (deck frame), at rest. Used to
        set up a 'base already standing' start without driving the arm — the block
        is still a free body, held only by friction afterwards."""
        slot = plan.SLOT_BY_NAME[name]
        adr, v = self._qadr[name], self._vadr[name]
        dy = self.data.mocap_pos[self._mocap][1]
        self.data.qpos[adr:adr + 3] = (slot.x, slot.y + dy, slot.z)
        self.data.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def settle(self, seconds: float) -> None:
        """Let the sim run (ship still sways) without commanding the arm."""
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.data.ctrl[:6] = self.data.qpos[:6]
            mujoco.mj_step(self.model, self.data)
            self.tick()

    # ── state the policy / metrics read ──────────────────────────────────────
    def block_states(self) -> dict[str, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        return {n: self.data.xpos[b].copy() for n, b in self._block_bids.items()}

    def integrity(self) -> float:
        return plan.integrity(self.block_states(),
                              float(self.data.mocap_pos[self._mocap][1]))

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[:6].copy()

    # ── grasp: the closing fingers grip and centre the block ─────────────────
    def grasp(self, name: str | None = None) -> bool:
        """Grip the block under the pinch. A real parallel-jaw gripper *centres* the
        object as its fingers close, so the block is aligned under the pinch (its
        pose corrected only by the few mm the jaws would pull it in) and welded to
        the palm. If no block is within reach, the grip takes nothing. The block is
        a free body again the instant it's released — this only models the hold.
        Returns whether a block was grasped."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        pinch = d.site_xpos[self._site].copy()
        cand = [(name, self._block_bids[name])] if name else \
            list(self._block_bids.items())
        best, best_d = None, 1e9
        for nm, b in cand:
            dist = float(np.linalg.norm(d.xpos[b] - pinch))
            if dist < best_d:
                best, best_d = nm, dist
        if best is None or best_d > _GRASP_REACH:
            return False
        idx = [s.name for s in plan.SLOTS].index(best)
        adr, v = self._qadr[best], self._vadr[best]
        d.qpos[adr:adr + 3] = pinch                  # centre the block in the jaws
        d.qpos[adr + 3:adr + 7] = (1, 0, 0, 0)
        d.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(m, d)
        body = self._blocks[idx]
        eq = self._welds[idx]
        p1, q1 = d.xpos[self._palm].copy(), d.xquat[self._palm].copy()
        p2, q2 = d.xpos[body].copy(), d.xquat[body].copy()
        q1i = np.zeros(4); mujoco.mju_negQuat(q1i, q1)
        prel = np.zeros(3); mujoco.mju_rotVecQuat(prel, p2 - p1, q1i)
        qrel = np.zeros(4); mujoco.mju_mulQuat(qrel, q1i, q2)
        m.eq_data[eq, :3] = 0.0
        m.eq_data[eq, 3:6] = prel
        m.eq_data[eq, 6:10] = qrel
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self._active_weld = eq
        return True

    def release(self) -> None:
        if self._active_weld >= 0:
            self.data.eq_active[self._active_weld] = 0
            self._active_weld = -1

    # ── IK on a moving base: give the scratch sim the live deck pose ──────────
    def _ik(self, target: np.ndarray, iters: int = 500, wrot: float = 0.4) -> np.ndarray:
        m = self.model
        ds = mujoco.MjData(m)
        ds.qpos[:] = self.data.qpos
        if m.nmocap:
            ds.mocap_pos[:] = self.data.mocap_pos
            ds.mocap_quat[:] = self.data.mocap_quat
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_forward(m, ds)
            perr = target - ds.site_xpos[self._site]
            qc = np.zeros(4); mujoco.mju_mat2Quat(qc, ds.site_xmat[self._site])
            qci = np.zeros(4); mujoco.mju_negQuat(qci, qc)
            qerr = np.zeros(4); mujoco.mju_mulQuat(qerr, _Q_DOWN, qci)
            if qerr[0] < 0:
                qerr = -qerr
            rerr = 2.0 * qerr[1:4]
            if np.linalg.norm(perr) < 5e-4 and np.linalg.norm(rerr) < 1e-2:
                break
            mujoco.mj_jacSite(m, ds, jacp, jacr, self._site)
            J = np.vstack([jacp[:, :6], wrot * jacr[:, :6]])
            err = np.concatenate([perr, wrot * rerr])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.08**2 * np.eye(6), err)
            ds.qpos[:6] = np.clip(ds.qpos[:6] + np.clip(dq, -0.1, 0.1),
                                  self._lo, self._hi)
        return ds.qpos[:6].copy()

    def _fingertip_z(self, qarm) -> float:
        s = self._scratch
        s.qpos[:] = self.data.qpos
        if self.model.nmocap:
            s.mocap_pos[:] = self.data.mocap_pos
            s.mocap_quat[:] = self.data.mocap_quat
        s.qpos[:6] = qarm
        mujoco.mj_forward(self.model, s)
        from cadenza.arm import _FINGERTIP_BELOW_PINCH
        return float(s.site_xpos[self._site][2]) - _FINGERTIP_BELOW_PINCH

    # ── pick / place (top-down), each motion ticking the deck ────────────────
    # cadenza's move holds the target for a fixed 120 settle steps after the ramp;
    # at the speeds the governor reaches that fixed cost dominates, so we use a much
    # shorter hold (the placement snaps the block home anyway, so we don't need the
    # servo to converge to the micron). This is what makes a fast re-seat actually
    # fast — and the efficiency governor's speed-up worth something.
    _CONVERGE_FAST = 24

    def _ramp_to(self, qarm, settle: int, speed: float) -> float:
        speed = float(np.clip(speed, 0.1, 8.0))
        q0 = self.data.qpos[:6].copy()
        travel = max(1, int(round(settle / speed)))
        self.data.ctrl[6] = self.data.ctrl[7] = self._grip
        for k in range(1, travel + 1):
            a = k / travel
            self.data.ctrl[:6] = (1.0 - a) * q0 + a * qarm
            mujoco.mj_step(self.model, self.data)
            self.tick()
        self.data.ctrl[:6] = qarm
        for _ in range(self._CONVERGE_FAST):
            mujoco.mj_step(self.model, self.data)
            self.tick()
        return float(np.linalg.norm(qarm - self.data.qpos[:6]))

    def move_to(self, target, settle: int = 420, speed: float = 1.0) -> float:
        return self._ramp_to(self._solve_above_surface(target), settle, speed)

    def set_grip(self, opening, settle: int = 90, speed: float = 1.0) -> None:
        super().set_grip(opening, settle=settle, render=self.tick, speed=speed)

    def home(self, speed: float = 1.0) -> None:
        if self.model.nkey > 0:
            self._ramp_to(self.model.key_qpos[0][:6], 420, speed)
        self.set_grip(_GRIP_OPEN, speed=speed)

    def pick(self, name: str, speed: float = 1.0) -> bool:
        """Pick up a block: open, drop the open gripper around it, close, lift.
        The grasp is a real pinch on the block where it actually is — no teleport,
        so the arm has to reach it. Returns whether the block came up."""
        blk = self.block_states()[name]
        loc = np.array([blk[0], blk[1], blk[2]])
        fast = min(4.0, speed * 1.4)
        self.set_grip(_GRIP_OPEN, settle=60, speed=fast)
        self.move_to([loc[0], loc[1], _TRANSIT_Z], speed=fast)   # above, from high
        self.move_to(loc + [0, 0, _HOVER], speed=fast)
        self.move_to(loc, speed=speed)
        got = self.grasp(name)
        self.set_grip(_GRIP_CLOSED, speed=speed)
        self.move_to([loc[0], loc[1], _TRANSIT_Z], speed=fast)   # lift clear of the tower
        return got

    def place(self, name: str, speed: float = 1.0, settle_steps: int = 12) -> bool:
        """Set a held block down on its slot and let go. No snapping, no welds —
        it rests on the tower by friction, and if it isn't supported it falls.
        Returns whether it ended up resting in its slot."""
        slot = plan.SLOT_BY_NAME[name]
        sp = lambda ph: _speed_for(speed, ph)
        # carry high above the slot, then descend straight down onto it (never
        # sweeping sideways over placed blocks)
        self.move_to([slot.x, slot.y + self.deck_y(), _TRANSIT_Z], speed=sp("carry"))
        self.move_to([slot.x, slot.y + self.deck_y(), slot.z + _HOVER], speed=sp("carry"))
        self.move_to([slot.x, slot.y + self.deck_y(), slot.z], speed=sp("lower"))
        for _ in range(settle_steps):                      # let it settle on contact
            self.data.ctrl[:6] = self.data.qpos[:6]
            mujoco.mj_step(self.model, self.data)
            self.tick()
        self.release()
        self.set_grip(_GRIP_OPEN, speed=sp("release"))
        self.move_to([slot.x, slot.y + self.deck_y(), _TRANSIT_Z], speed=sp("retract"))
        return self.is_placed(name)

    # ── rendering ────────────────────────────────────────────────────────────
    def _grab(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._height, self._width)
            cam = mujoco.MjvCamera()
            cam.distance, cam.azimuth, cam.elevation = self._cam_spec
            cam.lookat[:] = (0.45, 0.0, 0.45)
            self._cam = cam
        self._renderer.update_scene(self.data, self._cam)
        px = self._renderer.render()
        if self._overlay is not None:
            px = self._overlay(px, self)
        return px

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
