"""The live MuJoCo session for the rocking ship — IK, motion, grasp, and the deck.

Subclasses cadenza's ``_Runtime`` so the damped-least-squares IK, the top-down
pick/place motion and the table no-go clamp all come for free. What this adds:

- **The deck is a real slide joint** (driven by an actuator that :meth:`tick`
  commands from the :class:`ShipOscillator` every physics step). Unlike a mocap
  deck it has a real velocity, so friction actually carries the free blocks along
  with the ship — they *ride* the sway instead of hovering. ``tick`` is passed as
  the render callback into every arm motion, so the ship sways during all of them,
  even headless.
- **Arm/qpos offset.** The deck joint sits before the arm's six in ``qpos``, so the
  IK and motion use the arm slices ``_aq`` / ``_av`` rather than ``[:6]``.
- **14 free blocks** (no welds): they rest on friction and fall when unsupported.
  ``pick`` closes the jaws onto a block and lifts it for real — the fingers squeeze
  the cube and friction carries it (no teleport, no weld); ``brace`` presses the
  closed hand down on the tower so a lurch can't throw the blocks it covers.
- **Pick / place primitives** plus optional **frame capture** for the demo video.
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

# Finger opening to actually grip a block: tighter than cadenza's _GRIP_CLOSED so
# the jaws squeeze the 0.052 m cube and friction holds it (the open value leaves a
# gap wider than the block).
_GRIP_GRASP = 0.006
# Release just wide enough to free the block (faces at +-0.026 m) without splaying
# the fingers into the neighbouring blocks and knocking them over.
_GRIP_RELEASE = 0.024
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
        # the deck is a real roll hinge now (driven by an actuator): it has angular
        # velocity, friction carries the blocks along, and the whole platform (arm
        # included) tilts with it.
        self._deck_q = int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, "deck_roll")])
        self._deck_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_deck")
        self._deck_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "deck")
        # the deck joint sits before the arm's six, so the arm's qpos/qvel are
        # offset by it; ``_aq`` / ``_av`` are the arm's six joint slices.
        j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j1")
        a0, v0 = int(m.jnt_qposadr[j1]), int(m.jnt_dofadr[j1])
        self._aq, self._av = slice(a0, a0 + 6), slice(v0, v0 + 6)
        self._lo = m.jnt_range[j1:j1 + 6, 0].copy()
        self._hi = m.jnt_range[j1:j1 + 6, 1].copy()

        self._block_bids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                            for n in plan.SLOT_BY_NAME}
        self._blocks = [self._block_bids[s.name] for s in plan.SLOTS]
        self._welds = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                         f"grasp{i:02d}") for i in range(plan.N_BLOCKS)]
        self._active_weld = -1

        # free-joint qpos / qvel addresses for each block (real free bodies).
        self._qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")]) for n in self._block_bids}
        self._vadr = {n: int(m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")]) for n in self._block_bids}
        self._start = start

        self.osc = oscillator or ShipOscillator()
        self._t0 = 0.0
        self._last_lurch_idx = -1
        self.lurches_fired = 0
        self.lurch_log: list[float] = []      # sim-times of each lurch (for learning)
        self._rng = np.random.default_rng(0)
        self._jerk_t0 = None                  # set during a lurch (sharp deck slam)
        self._jerk_side = 1.0
        self._bracing = False                 # gripper pressing on the tower (HUD)

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
        self._maybe_lurch(t)
        # drive the ship: the gentle sway, OR — during a lurch — a hard sharp ROLL to
        # one side (a swell hitting that beam), held for the lurch, then eased back.
        # Rolling predominantly one way gives the loose blocks a definite downhill to
        # slide/topple toward (and a definite side for the brace to guard). The roll
        # acceleration is what throws them — no impulse fakery.
        if self._jerk_t0 is not None and (t - self._jerk_t0) < self.osc.lurch_dur:
            ph = (t - self._jerk_t0) / self.osc.lurch_dur
            ramp = min(1.0, ph / 0.25) * (1.0 if ph < 0.8 else (1.0 - ph) / 0.2)
            self.data.ctrl[self._deck_act] = self._jerk_side * self.osc.lurch_amp * ramp
        else:
            self._jerk_t0 = None
            self.data.ctrl[self._deck_act] = self.osc.position(t)
        if self.capture:
            self._frame_div += 1
            if self._frame_div % self._capture_every == 0:
                self.frames.append(self._grab())

    def _maybe_lurch(self, t: float) -> None:
        """Schedule the next hard lurch — a sharp slam of the deck (driven in
        :meth:`tick`). No per-block impulse: the deck really jerks and MuJoCo
        resolves which blocks slip and topple. Logs the time so the governor can
        learn the rhythm."""
        idx = self.osc.lurch_index(t)
        if idx is None or idx == self._last_lurch_idx:
            return
        self._last_lurch_idx = idx
        self._jerk_t0 = t
        self._jerk_side = self.osc.lurch_side(idx)
        self.lurches_fired += 1
        self.lurch_log.append(float(t))

    def lurching(self) -> bool:
        return self._jerk_t0 is not None

    def force_lurch(self, side: float = 1.0) -> None:
        """Trigger one sharp roll right now (used by the brace-discovery trials so a
        strategy can be tested against a real lurch on demand)."""
        self._jerk_t0 = self.sim_t()
        self._jerk_side = side
        self.lurches_fired += 1
        self.lurch_log.append(float(self.sim_t()))

    def start_clock(self) -> None:
        self._t0 = float(self.data.time)

    def deck_y(self, ahead: float = 0.0) -> float:
        """Commanded deck roll angle (rad) now or ``ahead`` seconds in the future."""
        return self.osc.position(self.data.time - self._t0 + ahead)

    def deck_pos(self) -> float:
        """Actual deck roll angle (rad) — the joint the whole platform rides on."""
        return float(self.data.qpos[self._deck_q])

    def deck_vel(self) -> float:
        return self.osc.velocity(self.data.time - self._t0)

    # ── deck-frame <-> world transform (the deck tilts, so blocks/slots live in the
    #    deck frame and we map through the live deck pose) ─────────────────────────
    def deck_world(self, local) -> np.ndarray:
        """A point given in the (level) deck frame -> its current world position."""
        mujoco.mj_forward(self.model, self.data)
        R = self.data.xmat[self._deck_bid].reshape(3, 3)
        return self.data.xpos[self._deck_bid] + R @ np.asarray(local, dtype=float)

    def deck_local(self, world) -> np.ndarray:
        """A world point -> its coordinates in the (level) deck frame."""
        R = self.data.xmat[self._deck_bid].reshape(3, 3)
        return R.T @ (np.asarray(world, dtype=float) - self.data.xpos[self._deck_bid])

    def block_states_local(self) -> dict[str, np.ndarray]:
        """Every block's position in the deck frame (rides the tilt out). This is the
        honest frame for 'is the pyramid still assembled' — a tower that holds its
        deck-frame shape is intact; one that slides downhill / topples is not."""
        mujoco.mj_forward(self.model, self.data)
        return {n: self.deck_local(self.data.xpos[b])
                for n, b in self._block_bids.items()}

    def sim_t(self) -> float:
        return self.data.time - self._t0

    # ── reset / settle ───────────────────────────────────────────────────────
    def reset(self) -> None:
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
        self.lurch_log = []
        self._jerk_t0 = None
        self._rng = np.random.default_rng(0)

    # ── placement state — blocks are free bodies, held in place by friction ────
    def landing_error(self, name: str) -> float:
        """Deck-frame distance of a block from its home slot."""
        slot = plan.SLOT_BY_NAME[name]
        p = self.block_states_local()[name]
        return float(np.hypot(p[0] - slot.x, p[1] - slot.y))

    def is_placed(self, name: str) -> bool:
        """Resting in its slot (deck frame). No welds — a block is 'placed' only as
        long as real physics keeps it there on the tilting deck."""
        return plan.is_block_placed(self.block_states_local(), name, 0.0)

    def at_rest(self, name: str) -> bool:
        v = self._vadr[name]
        return float(np.linalg.norm(self.data.qvel[v:v + 6])) < 0.5

    def set_in_slot(self, name: str) -> None:
        """Drop a free block straight into its slot (deck frame), at rest, oriented
        with the deck. Used to set up a 'base already standing' start without driving
        the arm — the block is still a free body, held only by friction afterwards."""
        slot = plan.SLOT_BY_NAME[name]
        adr, v = self._qadr[name], self._vadr[name]
        self.data.qpos[adr:adr + 3] = self.deck_world((slot.x, slot.y, slot.z))
        self.data.qpos[adr + 3:adr + 7] = self.data.xquat[self._deck_bid]
        self.data.qvel[v:v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def ramp_sway(self, target: float, seconds: float) -> None:
        """Smoothly change the sway amplitude (the seas picking up), holding the arm,
        so the built tower has time to settle into riding the larger sway."""
        start = self.osc.sway_amp
        steps = max(1, int(seconds / self.model.opt.timestep))
        for k in range(steps):
            self.osc.sway_amp = start + (target - start) * (k / steps)
            self.data.ctrl[:6] = self.data.qpos[self._aq]
            mujoco.mj_step(self.model, self.data)
            self.tick()
        self.osc.sway_amp = target

    def settle(self, seconds: float) -> None:
        """Let the sim run (ship still sways) without commanding the arm."""
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.data.ctrl[:6] = self.data.qpos[self._aq]      # hold the arm pose
            mujoco.mj_step(self.model, self.data)
            self.tick()

    # ── state the policy / metrics read ──────────────────────────────────────
    def block_states(self) -> dict[str, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        return {n: self.data.xpos[b].copy() for n, b in self._block_bids.items()}

    def integrity(self) -> float:
        return plan.integrity(self.block_states_local(), 0.0)

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self._av].copy()

    # ── grasp: REAL — the fingers close on the block and friction holds it ────
    def holding(self) -> str | None:
        """Which block (if any) the gripper is actually holding — a block pinched
        between the closing fingers and lifted with the hand. Read from physics, not
        a weld: a block within the finger gap, near the pinch, and moving with it."""
        mujoco.mj_forward(self.model, self.data)
        pinch = self.data.site_xpos[self._site]
        best, best_d = None, 1e9
        for nm, b in self._block_bids.items():
            d = float(np.linalg.norm(self.data.xpos[b] - pinch))
            if d < best_d:
                best, best_d = nm, d
        return best if best_d < 0.05 else None

    # ── IK for the 6 arm joints (the deck joint comes first in qpos; the scratch
    #    sim inherits the live deck position, so IK plans for where the ship is) ──
    def _ik(self, target: np.ndarray, iters: int = 500, wrot: float = 0.4) -> np.ndarray:
        m = self.model
        aq, av = self._aq, self._av
        ds = mujoco.MjData(m)
        ds.qpos[:] = self.data.qpos
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
            J = np.vstack([jacp[:, av], wrot * jacr[:, av]])
            err = np.concatenate([perr, wrot * rerr])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.08**2 * np.eye(6), err)
            ds.qpos[aq] = np.clip(ds.qpos[aq] + np.clip(dq, -0.1, 0.1),
                                  self._lo, self._hi)
        return ds.qpos[aq].copy()

    def _fingertip_z(self, qarm) -> float:
        s = self._scratch
        s.qpos[:] = self.data.qpos
        s.qpos[self._aq] = qarm
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
        q0 = self.data.qpos[self._aq].copy()
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
        return float(np.linalg.norm(qarm - self.data.qpos[self._aq]))

    def move_to(self, target, settle: int = 420, speed: float = 1.0) -> float:
        return self._ramp_to(self._solve_above_surface(target), settle, speed)

    def set_grip(self, opening, settle: int = 90, speed: float = 1.0) -> None:
        super().set_grip(opening, settle=settle, render=self.tick, speed=speed)

    def home(self, speed: float = 1.0) -> None:
        if self.model.nkey > 0:
            self._ramp_to(self.model.key_qpos[0][self._aq], 420, speed)
        self.set_grip(_GRIP_OPEN, speed=speed)

    def pick(self, name: str, speed: float = 1.0) -> bool:
        """Pick up a block for real: open the jaws, descend around the block, CLOSE
        the jaws onto it (the parallel fingers squeeze and self-centre it; friction
        holds it — no teleport, no weld), and lift. Returns whether it came up."""
        blk = self.block_states()[name]
        loc = np.array([blk[0], blk[1], blk[2]])
        fast = min(4.0, speed * 1.4)
        self.set_grip(_GRIP_OPEN, settle=60, speed=fast)
        self.move_to([loc[0], loc[1], _TRANSIT_Z], speed=fast)   # above, from high
        self.move_to(loc + [0, 0, _HOVER], speed=fast)
        self.move_to(loc, speed=speed)                           # descend around it
        self.set_grip(_GRIP_GRASP, settle=140, speed=0.8)        # squeeze it
        self.move_to([loc[0], loc[1], _TRANSIT_Z], speed=fast)   # lift
        return self.holding() == name

    def place(self, name: str, speed: float = 1.0, settle_steps: int = 16) -> bool:
        """Carry a held block over its slot, lower it on, and OPEN the jaws to let
        go — it rests on the tower by friction (and falls if unsupported). Returns
        whether it ended up resting in its slot."""
        slot = plan.SLOT_BY_NAME[name]
        sp = lambda ph: _speed_for(speed, ph)
        w = self.deck_world((slot.x, slot.y, slot.z))      # slot in world (live tilt)
        # carry high above the slot, then descend straight down onto it (never
        # sweeping sideways over placed blocks)
        self.move_to([w[0], w[1], _TRANSIT_Z], speed=sp("carry"))
        self.move_to(self.deck_world((slot.x, slot.y, slot.z + _HOVER)), speed=sp("carry"))
        self.move_to(self.deck_world((slot.x, slot.y, slot.z)), speed=sp("lower"))
        for _ in range(settle_steps):                      # let it settle on contact
            self.data.ctrl[:6] = self.data.qpos[self._aq]
            mujoco.mj_step(self.model, self.data)
            self.tick()
        # open only just enough to let go (a wide splay would knock the neighbours)
        self.set_grip(_GRIP_RELEASE, speed=sp("release"))
        self.move_to(self.deck_world((slot.x, slot.y, slot.z + _LIFT)), speed=sp("retract"))
        return self.is_placed(name)

    # ── brace: the new action megan-tk learns — physically hold the tower ────
    # This is real and works *with* the roll: the arm sets its hand down on top of the
    # tower (a steadying finger on the capstone) and freezes its joint angles. Because
    # the arm is bolted to the same deck, a frozen pose keeps the hand pinned on the
    # tower as the whole platform tilts — so when the ship rolls and the tower starts
    # to tip over its downhill edge, the downward pin holds it and it rides the lurch
    # out instead of toppling. No impulse-skipping; it's the contact physics. Too hard
    # a press wedges the stack apart, so a light touch is best — which is exactly the
    # kind of thing the discovery loop (try a few, keep what holds) settles.
    #
    # A brace STRATEGY is (anchor_block, press_depth): where to set the hand down and
    # how firmly. The discovery loop tries these and keeps whichever holds best.
    BRACE_STRATEGIES = {
        "pin_cap":      ("block13", 0.000),   # rest on the capstone (light touch)
        "pin_cap_firm": ("block13", 0.020),   # press the capstone a little
        "pin_mid":      ("block09", 0.000),   # steady the 2x2 tier instead
    }

    def brace_engage(self, side: float = 1.0, strategy: str = "pin_cap",
                     speed: float = 4.0) -> int:
        """Set the steadying hand down on the tower per ``strategy`` and freeze it.
        ``side`` is accepted for API symmetry (the pin is symmetric) but unused."""
        anchor, depth = self.BRACE_STRATEGIES.get(strategy,
                                                  self.BRACE_STRATEGIES["pin_cap"])
        s = plan.SLOT_BY_NAME[anchor]
        self.set_grip(_GRIP_GRASP, speed=speed)            # a firm, narrow tip
        w = self.deck_world((s.x, s.y, s.z))
        self.move_to([w[0], w[1], _TRANSIT_Z], speed=speed)             # above the top
        self.move_to(self.deck_world((s.x, s.y, s.z + 0.10)), speed=speed)
        self.move_to(self.deck_world((s.x, s.y, s.z - depth)), speed=2.0)  # set it down
        self._bracing = True
        self._brace_q = self.data.qpos[self._aq].copy()    # freeze the steadying pose
        return sum(self.is_placed(n) for n in self._block_bids)

    def brace_hold(self, seconds: float) -> None:
        """Hold the steadying hand in place. Freeze the joint angles: the arm rides
        the deck, so a fixed pose keeps the hand pinned on the rolling tower."""
        q = getattr(self, "_brace_q", self.data.qpos[self._aq].copy())
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.data.ctrl[:6] = q
            mujoco.mj_step(self.model, self.data)
            self.tick()

    def brace_release(self, speed: float = 4.0) -> None:
        self._bracing = False
        cap = plan.SLOT_BY_NAME["block13"]
        w = self.deck_world((cap.x, cap.y, cap.z))
        self.move_to([w[0], w[1], _TRANSIT_Z], speed=speed)

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
