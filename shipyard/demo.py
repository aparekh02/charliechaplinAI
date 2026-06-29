"""The charliechaplinAI demo: building pyramids on a moving ship with megan-tk.

Three acts, then a side-by-side payoff rendered as one narrative video:

1. **Learn to work fast** — the efficiency governor rebuilds the pyramid from
   scattered blocks over and over, ratcheting the build speed up while the tower
   still comes out intact. Rebuild time falls sharply.
2. **Diagnose the ship** — a frozen arm is overwhelmed by the rocking; progress
   goes flat; the neuro-symbolic governor diagnoses ``OSCILLATION`` and fires a
   FourierFT PEFT kick at the VLA's attention layer.
3. **Survive the lurches** — a continuous run: the deck sways and every so often
   lurches hard, throwing several blocks off the tower. The **regular VLA** (slow,
   no governor) can't rebuild fast enough and the pyramid stays in pieces; the
   **megan-tk** arm, at the speed it learned, rebuilds between lurches and keeps
   the pyramid standing.

Outputs (to ``--out``): ``results.json``, ``integrity.png`` (the two survival
curves), and ``demo.mp4`` (intro -> regular VLA -> with megan-tk -> outro).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shipyard.vla import GovernedArm
from shipyard.overlay import make_overlay


def _hd(): return 1280, 720


def _print_header(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


# ── title / outro cards (cv2 text on a deep gradient) ────────────────────────
def _card(lines, seconds=2.6, fps=30, accent=(120, 200, 255)):
    import cv2
    w, h = _hd()
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):                                   # vertical gradient
        t = y / h
        img[y, :] = (int(24 + 16 * t), int(26 + 18 * t), int(34 + 26 * t))
    cy = h // 2 - 30 * (len(lines) - 1)
    for i, (text, scale, col, thick) in enumerate(lines):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)[0][0]
        cv2.putText(img, text, (w // 2 - tw // 2, cy), cv2.FONT_HERSHEY_DUPLEX,
                    scale, col, thick, cv2.LINE_AA)
        cy += int(64 * scale)
    cv2.rectangle(img, (w // 2 - 120, cy + 6), (w // 2 + 120, cy + 12), accent, -1)
    return [img.copy() for _ in range(int(seconds * fps))]


def run(out_dir: Path, *, episodes: int, duration: float, render: bool,
        seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    LURCH_EVERY, IMPULSE = 15.0, 1.0

    # ── Act 1 + 2: the governor learns, on a real torch VLA ──────────────────
    _print_header("ACT 1  efficiency governor learns to build faster")
    arm = GovernedArm(governed=True, seed=seed, lurch_every=LURCH_EVERY,
                      lurch_impulse=IMPULSE)
    learn = arm.learn_speed(episodes=episodes)
    for r in learn:
        print(f"  ep{r['episode']:02d}  trial {r['trial_speed']:.1f}x  "
              f"rebuild {r['duration']:5.1f}s  integrity {r['integrity']:.2f}  "
              f"-> {r['status']:7s}  (speed now {r['speed']:.1f}x)")
    learned_speed = arm.speed
    print(f"  learned build speed: {learned_speed:.1f}x  "
          f"(rebuild {learn[0]['duration']:.0f}s -> {learn[-1]['duration']:.0f}s)")

    _print_header("ACT 2  neuro-symbolic governor diagnoses the rocking")
    diag = arm.diagnose()
    print(f"  diagnoses: {diag['hypotheses']}")
    for k in diag["kicks"][:5]:
        print(f"  KICK  {k['hypothesis']:11s}  {k['method']:7s} -> {k['layers']}")
    arm.close()
    results.update(learn=learn, learned_speed=round(learned_speed, 2),
                   diagnosis={"hypotheses": diag["hypotheses"], "kicks": diag["kicks"]})

    # ── Act 3: survival — regular VLA vs megan-tk ────────────────────────────
    _print_header("ACT 3  survival on the rocking ship: regular VLA vs megan-tk")
    cap = dict(capture=render, width=_hd()[0], height=_hd()[1], capture_every=16)

    reg = GovernedArm(governed=False, seed=seed + 1, lurch_every=LURCH_EVERY,
                      lurch_impulse=IMPULSE)
    reg.rt.capture = render
    reg.rt._width, reg.rt._height, reg.rt._capture_every = _hd()[0], _hd()[1], 16
    reg_res = reg.survival(speed=1.0, duration=duration,
                           overlay=make_overlay("Regular VLA", "no governor  1.0x")
                           if render else None)
    reg_frames = list(reg.rt.frames); reg.close()
    print(f"  REGULAR VLA  (1.0x) : mean integrity {reg_res['mean_integrity']:.2f}  "
          f"final {reg_res['final']:.2f}  (lurches {reg_res['lurches']})")

    gov = GovernedArm(governed=False, seed=seed + 1, lurch_every=LURCH_EVERY,
                      lurch_impulse=IMPULSE)
    gov.rt.capture = render
    gov.rt._width, gov.rt._height, gov.rt._capture_every = _hd()[0], _hd()[1], 16
    sub = f"learned {learned_speed:.1f}x  OSCILLATION->FourierFT"
    gov_res = gov.survival(speed=learned_speed, duration=duration,
                           overlay=make_overlay("With megan-tk", sub) if render else None)
    gov_frames = list(gov.rt.frames); gov.close()
    print(f"  WITH megan-tk ({learned_speed:.1f}x): mean integrity "
          f"{gov_res['mean_integrity']:.2f}  final {gov_res['final']:.2f}  "
          f"(lurches {gov_res['lurches']})")
    gain = gov_res["mean_integrity"] - reg_res["mean_integrity"]
    print(f"\n  ==> megan-tk keeps the pyramid {gain*100:+.0f} integrity-points "
          f"higher on average")

    results["survival_regular"] = {k: v for k, v in reg_res.items() if k != "series"}
    results["survival_megantk"] = {k: v for k, v in gov_res.items() if k != "series"}

    # ── artifacts ────────────────────────────────────────────────────────────
    _plot(out_dir / "integrity.png", reg_res, gov_res, learned_speed)
    if render and gov_frames:
        white = (245, 245, 250)
        story = (
            _card([("Building Pyramids on a Moving Ship", 1.5, white, 3),
                   ("a 6-axis arm  -  real MuJoCo physics", 0.9, (180, 200, 220), 2),
                   ("kept standing by megan-tk", 0.9, (120, 200, 255), 2)], 3.0)
            + _card([("A regular VLA model", 1.5, white, 3),
                     ("knows how to stack - but can't keep up", 0.85,
                      (200, 200, 210), 2),
                     ("with the rocking ship", 0.85, (200, 200, 210), 2)], 2.6)
            + reg_frames
            + _card([("With megan-tk", 1.5, white, 3),
                     ("diagnoses the OSCILLATION, builds faster", 0.85,
                      (160, 220, 160), 2),
                     (f"learned {learned_speed:.1f}x  -  pyramid stays up", 0.85,
                      (120, 200, 255), 2)], 2.6)
            + gov_frames
            + _card([(f"regular VLA   {reg_res['mean_integrity']*100:.0f}%  "
                      f"vs   megan-tk  {gov_res['mean_integrity']*100:.0f}%", 1.2,
                      white, 3),
                     ("average pyramid integrity on the rocking ship", 0.85,
                      (180, 200, 220), 2)], 3.2))
        _write_mp4(out_dir / "demo.mp4", story)
        results["video"] = str(out_dir / "demo.mp4")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {out_dir/'results.json'}, {out_dir/'integrity.png'}"
          + (f", {out_dir/'demo.mp4'}" if results.get("video") else ""))
    return results


def _plot(path, reg, gov, speed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for res, lab, c in ((reg, "regular VLA (1.0x)", "#d6453b"),
                        (gov, f"megan-tk ({speed:.1f}x)", "#2e7d32")):
        t = [p[0] for p in res["series"]]
        y = [p[1] for p in res["series"]]
        ax.plot(t, y, label=lab, color=c, lw=2)
    ax.axhline(0.43, ls=":", c="#888", lw=1)
    ax.text(1, 0.45, "base only (upper rows knocked off)", fontsize=8, color="#888")
    ax.set_xlabel("time on the rocking ship (s)")
    ax.set_ylabel("pyramid integrity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Keeping the pyramid standing through the lurches")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _write_mp4(path, frames, fps: int = 30):
    import imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                            macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.out), episodes=args.episodes, duration=args.duration,
        render=not args.no_render, seed=args.seed)


if __name__ == "__main__":
    main()
