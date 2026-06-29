"""Heads-up display drawn onto rendered frames (title, clock, live integrity, and
whether the ship is lurching). Pure cv2 drawing over the numpy RGB frame, scaled
to the frame size so it reads at any resolution."""

from __future__ import annotations

import cv2
import numpy as np


def make_overlay(title: str, subtitle: str = ""):
    """Return an overlay(frame, rt) that draws the HUD for a survival run."""
    def overlay(px: np.ndarray, rt) -> np.ndarray:
        img = np.ascontiguousarray(px[:, :, ::-1])     # RGB -> BGR for cv2
        h, w = img.shape[:2]
        s = h / 720.0                                  # scale fonts to resolution
        integ = rt.integrity()
        col = ((90, 210, 90) if integ >= 0.8 else
               (70, 180, 240) if integ >= 0.5 else (70, 90, 235))

        # top title bar
        cv2.rectangle(img, (0, 0), (w, int(58 * s)), (28, 30, 38), -1)
        cv2.putText(img, title, (int(20 * s), int(40 * s)), cv2.FONT_HERSHEY_DUPLEX,
                    1.1 * s, (245, 245, 250), max(1, int(2 * s)), cv2.LINE_AA)
        if subtitle:
            tw = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                                 max(1, int(2 * s)))[0][0]
            cv2.putText(img, subtitle, (w - tw - int(20 * s), int(39 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (150, 205, 255),
                        max(1, int(2 * s)), cv2.LINE_AA)

        # bottom integrity bar
        y = h - int(46 * s)
        cv2.putText(img, f"pyramid integrity {integ*100:3.0f}%", (int(20 * s), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85 * s, (240, 240, 245),
                    max(1, int(2 * s)), cv2.LINE_AA)
        bx, by, bw, bh = int(20 * s), h - int(30 * s), w - int(40 * s), int(16 * s)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (55, 58, 68), -1)
        cv2.rectangle(img, (bx, by), (bx + int(bw * integ), by + bh), col, -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (220, 220, 230),
                      max(1, int(1 * s)))

        # clock + lurch counter, and a flash when the ship lurches
        info = f"t={rt.sim_t():4.1f}s   lurches={rt.lurches_fired}"
        iw = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                             max(1, int(2 * s)))[0][0]
        cv2.putText(img, info, (w - iw - int(20 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * s, (215, 215, 225), max(1, int(2 * s)), cv2.LINE_AA)
        if rt.osc.is_lurching(rt.sim_t()):
            txt = "LURCH!"
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.6 * s,
                                 max(1, int(3 * s)))[0][0]
            cv2.putText(img, txt, (w // 2 - tw // 2, int(110 * s)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6 * s, (60, 80, 240),
                        max(1, int(3 * s)), cv2.LINE_AA)
        return img[:, :, ::-1]                          # back to RGB
    return overlay
