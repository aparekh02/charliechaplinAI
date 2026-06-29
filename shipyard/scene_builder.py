"""Generate the rocking-ship + pyramid MuJoCo scene as MJCF text.

Derived from cadenza's arm scene (and robogpt's stack scene): same 6-axis arm,
table, pedestal, gripper, collision groups and actuators. Two things change:

1. **The deck is a ``mocap`` body** that the arm, pedestal and table all hang off.
   A mocap body is kinematically driven (we set ``data.mocap_pos`` every physics
   step) rather than simulated, so the "ship" sways exactly on the rhythm we ask
   for — no actuator to tune, no contact dynamics on the deck itself. Crucially a
   mocap body adds *no joint*, so the arm's six joints stay the first six in the
   model and cadenza's IK (which slices ``jnt_range[:6]``) works unchanged.

2. **Fourteen free blocks** start pre-built into the pyramid (positions from
   :mod:`shipyard.pyramid_plan`). They couple to the table only through friction,
   so when the deck accelerates under them they skid — slowly at first, then the
   upper rows topple. Block/table friction is low on purpose: the tower stands on
   a level deck but creeps once the ship starts rocking.

Use :func:`build_scene_xml` to get the MJCF string (the runtime writes it to a
temp file and loads it), or run this module to dump it to ``assets/``.
"""

from __future__ import annotations

from pathlib import Path

from shipyard import pyramid_plan as plan

# Block/table contact friction (slide, torsion, roll). Low enough that a small
# deck acceleration overcomes it (blocks skid), high enough that the pyramid
# stands on a still deck. Tuned empirically.
# Real friction: high static grip so a well-built tower stands on the gently
# swaying deck, but the blocks are free bodies — a hard lurch slides/topples them
# for real, and the gripper (or a carried block) can nudge them.
BLOCK_FRICTION = "0.7 0.15 0.004"
TABLE_FRICTION = "0.7 0.15 0.004"

# Deck (mocap) origin in world; the arm/table positions below are relative to it.
DECK_POS = (0.0, 0.0, 0.0)


def _materials() -> str:
    mats = []
    for i, (r, g, b, a) in enumerate(plan.block_colors()):
        mats.append(f'    <material name="block{i:02d}" rgba="{r:.3f} {g:.3f} '
                    f'{b:.3f} {a:.1f}"/>')
    return "\n".join(mats)


def _starts(start, scatter_seed, settle_gap):
    """Initial block centres, in SLOTS order, for the chosen start state:

    - ``"built"``   — the full pyramid (tiny per-tier gap so touching faces settle
      cleanly instead of interpenetrating).
    - ``"knocked"`` — the 3x3 base in place, the upper 5 lying flat on the table
      (the demo's start: the waves already toppled the top, the arm rebuilds it).
    - ``"scatter"`` — all 14 loose on the table (uses ``scatter_seed``).
    """
    if start == "scatter":
        return [tuple(p) for p in plan.scatter_positions(scatter_seed or 0)]
    knocked = plan.knocked_scatter()
    out = []
    for s in plan.SLOTS:
        if start == "knocked" and s.name in knocked:
            out.append(knocked[s.name])
        else:
            out.append((s.x, s.y, s.z + settle_gap * s.tier))
    return out


def _blocks_and_welds(start, scatter_seed, block_friction, settle_gap):
    """Block bodies (chosen start state) plus one weld per block."""
    starts = _starts(start, scatter_seed, settle_gap)
    bodies, welds = [], []
    for i, (x, y, z) in enumerate(starts):
        name = f"block{i:02d}"
        bodies.append(
            f'    <body name="{name}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
            f'      <freejoint name="{name}_free"/>\n'
            f'      <geom name="{name}_geom" type="box" '
            f'size="{plan.BLOCK_HALF} {plan.BLOCK_HALF} {plan.BLOCK_HALF}" '
            f'material="{name}" mass="0.05" friction="{block_friction}" '
            f'condim="3" contype="2" conaffinity="7"/>\n'
            f'    </body>')
        # the gripper grip: an inactive weld activated when the closed fingers are
        # on a block (models a real pinch grasp). Released to set the block down —
        # it then rests on the tower by friction alone, real free-body physics.
        welds.append(f'    <weld name="grasp{i:02d}" body1="palm" '
                     f'body2="{name}" active="false"/>')
    return "\n".join(bodies), "\n".join(welds)


def _keyframe(start, scatter_seed, settle_gap) -> str:
    """Full keyframe: arm home + open grip + every block's free-joint pose + the
    deck mocap pose. qpos = 6 arm + 2 grip + 14*(xyz + quat)."""
    starts = _starts(start, scatter_seed, settle_gap)
    arm = "0 0.6 0.9 0 0.6 0"
    grip = "0.04 0.04"
    blocks = "  ".join(f"{x:.4f} {y:.4f} {z:.4f} 1 0 0 0" for x, y, z in starts)
    return (
        f'    <key name="home"\n'
        f'         qpos="{arm} {grip}  {blocks}"\n'
        f'         ctrl="{arm} {grip}"\n'
        f'         mpos="0 0 0" mquat="1 0 0 0"/>')


def build_scene_xml(start: str = "scatter", *, scatter_seed: int | None = 0,
                    block_friction: str = BLOCK_FRICTION,
                    table_friction: str = TABLE_FRICTION,
                    settle_gap: float = 0.0008) -> str:
    """Return the full MJCF for the rocking-ship pyramid scene.

    ``start`` selects the initial block layout (see :func:`_starts`): ``"knocked"``
    (default — base built, upper 5 to be stacked by the arm), ``"built"`` (full
    pyramid), or ``"scatter"``. ``block_friction`` / ``table_friction`` /
    ``settle_gap`` are exposed for tuning the skid-vs-stand balance.
    """
    materials = _materials()
    bodies, welds = _blocks_and_welds(start, scatter_seed, block_friction, settle_gap)
    keyframe = _keyframe(start, scatter_seed, settle_gap)
    dx, dy, dz = DECK_POS
    return f"""<mujoco model="charliechaplin_ship_pyramid">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002" iterations="50"/>

  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.4 0.4 0.4" specular="0.15 0.15 0.15"/>
    <rgba haze="0.16 0.28 0.40 1"/>
    <global azimuth="140" elevation="-18" offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="sea" builtin="checker" rgb1="0.10 0.25 0.40"
             rgb2="0.06 0.16 0.28" width="512" height="512"/>
    <material name="sea" texture="sea" texrepeat="12 12" reflectance="0.3"/>
    <material name="metal" rgba="0.55 0.58 0.62 1"/>
    <material name="joint" rgba="0.95 0.6 0.12 1"/>
    <material name="link"  rgba="0.25 0.45 0.78 1"/>
    <material name="grip"  rgba="0.13 0.14 0.17 1"/>
    <material name="deck"  rgba="0.66 0.48 0.30 1"/>
    <material name="hull"  rgba="0.34 0.23 0.14 1"/>
    <material name="table" rgba="0.50 0.37 0.24 1"/>
{materials}
  </asset>

  <default>
    <joint damping="3" armature="0.05" frictionloss="0.1"/>
    <position kp="600" dampratio="1" forcerange="-200 200"/>
    <geom contype="0" conaffinity="0" condim="3" friction="1 0.05 0.001"/>
    <!-- arm links: collide with the world (table/deck/walls) so the arm can't
         pass through them, but NOT with the blocks (only the gripper touches
         those). contype bit3=8, conaffinity bit0=1. -->
    <default class="arm">
      <geom contype="8" conaffinity="1"/>
    </default>
    <default class="finger">
      <joint type="slide" damping="8" armature="0.01" range="0 0.045"/>
      <position kp="400" dampratio="1" forcerange="-80 80"/>
      <geom material="grip" friction="2.5 0.2 0.02" condim="4"
            contype="4" conaffinity="2"/>
    </default>
  </default>

  <worldbody>
    <light name="top" pos="0.3 0 3.0" dir="0 0 -1" diffuse="0.8 0.8 0.8"
           castshadow="true"/>
    <!-- the sea, well below the raft so it never z-fights or shows through -->
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.35" material="sea"
          contype="0" conaffinity="0"/>

    <!-- THE SHIP: one solid raft, kinematically driven (mocap). The arm, pedestal
         and table are bolted to it, so the whole thing moves as one plank when the
         deck sways or lurches. The deck top is a real collision surface, so blocks
         knocked off the tower land and tumble on the deck (not through it). The
         free blocks are NOT children — they only follow the deck through friction,
         which is what lets a hard lurch throw them off. -->
    <body name="deck" mocap="true" pos="{dx} {dy} {dz}">
      <geom name="deck_top" type="box" size="0.66 0.62 0.03" pos="0.26 0 -0.03"
            material="deck" contype="1" conaffinity="10" friction="0.6 0.1 0.01"
            condim="3"/>
      <geom name="hull" type="box" size="0.56 0.5 0.10" pos="0.26 0 -0.16"
            material="hull" contype="0" conaffinity="0"/>
      <!-- gunwales: solid barriers around the deck edge so blocks knocked off the
           tower stay on the ship instead of flying into the sea (WORLD group). -->
      <geom name="rail_y1" type="box" size="0.66 0.02 0.16" pos="0.26 0.6 0.13"
            material="hull" contype="1" conaffinity="10" friction="0.5 0.1 0.01"/>
      <geom name="rail_y0" type="box" size="0.66 0.02 0.16" pos="0.26 -0.6 0.13"
            material="hull" contype="1" conaffinity="10" friction="0.5 0.1 0.01"/>
      <geom name="rail_x1" type="box" size="0.02 0.62 0.16" pos="0.9 0 0.13"
            material="hull" contype="1" conaffinity="10" friction="0.5 0.1 0.01"/>
      <geom name="rail_x0" type="box" size="0.02 0.62 0.16" pos="-0.38 0 0.13"
            material="hull" contype="1" conaffinity="10" friction="0.5 0.1 0.01"/>

      <geom name="pedestal" type="cylinder" size="0.09 0.1" pos="0 0 0.1"
            material="metal" contype="0" conaffinity="0"/>

      <body name="table" pos="0.5 0 0">
        <geom name="table_top" type="box" size="0.22 0.32 0.02" pos="0 0 0.38"
              material="table" contype="1" conaffinity="10"
              friction="{table_friction}" condim="4"
              solref="0.004 1" solimp="0.98 0.99 0.001"/>
        <geom name="leg_a" type="box" size="0.02 0.02 0.19" pos="0.18 0.28 0.19" material="table"/>
        <geom name="leg_b" type="box" size="0.02 0.02 0.19" pos="0.18 -0.28 0.19" material="table"/>
        <geom name="leg_c" type="box" size="0.02 0.02 0.19" pos="-0.18 0.28 0.19" material="table"/>
        <geom name="leg_d" type="box" size="0.02 0.02 0.19" pos="-0.18 -0.28 0.19" material="table"/>
      </body>

      <body name="shoulder" pos="0 0 0.2" gravcomp="1" childclass="arm">
        <joint name="j1" axis="0 0 1" range="-2.9 2.9"/>
        <geom type="cylinder" size="0.07 0.05" pos="0 0 0.03" material="joint"/>
        <body name="upper_arm" pos="0 0 0.06" gravcomp="1">
          <joint name="j2" axis="0 1 0" range="-2.0 2.0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.34" size="0.05" material="link"/>
          <geom type="sphere" size="0.06" pos="0 0 0" material="joint"/>
          <body name="forearm" pos="0 0 0.34" gravcomp="1">
            <joint name="j3" axis="0 1 0" range="-2.7 2.7"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.28" size="0.042" material="link"/>
            <geom type="sphere" size="0.052" pos="0 0 0" material="joint"/>
            <body name="wrist_roll" pos="0 0 0.28" gravcomp="1">
              <joint name="j4" axis="0 0 1" range="-3.0 3.0"/>
              <geom type="cylinder" size="0.04 0.04" pos="0 0 0.02" material="joint"/>
              <body name="wrist_pitch" pos="0 0 0.05" gravcomp="1">
                <joint name="j5" axis="0 1 0" range="-2.0 2.0"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.038" material="link"/>
                <body name="palm" pos="0 0 0.06" gravcomp="1">
                  <joint name="j6" axis="0 0 1" range="-3.0 3.0"/>
                  <!-- compact end-effector: the grasp is a weld, so the tip only
                       needs to reach the block, not clamp it. Keeping it small
                       (about one block wide) lets the arm set blocks into the
                       tight 2x2 / capstone without the palm knocking neighbours. -->
                  <!-- The gripper DOES collide with the blocks and the table
                       (HAND group: contype bit2=4, conaffinity bits0,1=3). So it
                       can't pass through a block — it can nudge one, and a block
                       it's carrying can push others. Real contact. -->
                  <geom type="box" size="0.020 0.018 0.010" pos="0 0 0.006" material="grip"
                        contype="4" conaffinity="2"/>
                  <site name="pinch" pos="0 0 0.062" size="0.006" rgba="0 1 0 0.6"/>
                  <camera name="grip_cam" pos="0.06 0 0.0" xyaxes="0 1 0 1 0 0" fovy="60"/>
                  <body name="left_finger" pos="0 0.011 0.018">
                    <joint name="grip_left" class="finger" axis="0 1 0"/>
                    <geom type="box" size="0.006 0.007 0.024" pos="0 0 0.024"
                          contype="4" conaffinity="2"/>
                  </body>
                  <body name="right_finger" pos="0 -0.011 0.018">
                    <joint name="grip_right" class="finger" axis="0 -1 0"/>
                    <geom type="box" size="0.006 0.007 0.024" pos="0 0 0.024"
                          contype="4" conaffinity="2"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- the 14 blocks (free bodies; pre-built into the pyramid) -->
{bodies}
  </worldbody>

  <equality>
{welds}
  </equality>

  <actuator>
    <position name="a_j1" joint="j1"/>
    <position name="a_j2" joint="j2"/>
    <position name="a_j3" joint="j3"/>
    <position name="a_j4" joint="j4"/>
    <position name="a_j5" joint="j5"/>
    <position name="a_j6" joint="j6"/>
    <position name="a_grip_left"  class="finger" joint="grip_left"/>
    <position name="a_grip_right" class="finger" joint="grip_right"/>
  </actuator>

  <keyframe>
{keyframe}
  </keyframe>
</mujoco>
"""


def write_scene(path: str | Path, start: str = "knocked") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_scene_xml(start))
    return path


if __name__ == "__main__":
    out = write_scene(Path(__file__).resolve().parent / "assets" / "ship_pyramid.xml")
    print(f"wrote {out}")
