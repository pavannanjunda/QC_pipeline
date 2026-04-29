"""
pipeline/hard_qc.py
--------------------
Hard QC (PRD validation) — any single failure immediately rejects the video.

Checks:
  1. Duration       → must be >= MIN_DURATION_SEC
  2. Resolution     → must be >= MIN_WIDTH x MIN_HEIGHT (1080p)
  3. FPS            → must be >= MIN_FPS (30)
  4. File integrity → OpenCV could open it (checked in preprocessing)
  5. Motion         → at least some frame-to-frame difference detected

Returns:
  {
    "hard_qc_status": "PASS" | "FAIL",
    "reasons": [...]          # populated only on FAIL
  }
"""

from __future__ import annotations

import numpy as np

import config
from utils.logger import logger


# ── Motion detection helpers ──────────────────────────────────────────────────

def _compute_motion_scores(frames: list[np.ndarray]) -> list[float]:
    """
    Compute mean absolute difference between consecutive grey frames.
    Returns a list of scores (one per consecutive pair).
    """
    if len(frames) < 2:
        return [0.0]

    scores: list[float] = []
    import cv2

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray - prev_gray))
        scores.append(float(diff))
        prev_gray = gray
    return scores


# ── Main checker ──────────────────────────────────────────────────────────────

MOTION_THRESHOLD = 2.0   # mean pixel diff below this → static video


def run_hard_qc(
    frames: list[np.ndarray],
    meta: dict,
) -> dict:
    """
    Run all hard-QC checks.

    Args:
        frames  : sampled frames from preprocessing
        meta    : dict with fps, width, height, duration_sec

    Returns:
        dict with hard_qc_status ("PASS" | "FAIL") and reasons list.
    """
    reasons: list[str] = []

    # 1. Duration
    if meta["duration_sec"] < config.MIN_DURATION_SEC:
        reasons.append(
            f"Effective duration too short: "
            f"{meta['duration_sec']:.1f}s < {config.MIN_DURATION_SEC}s"
        )

    # 2. Resolution
    if meta["width"] < config.MIN_WIDTH or meta["height"] < config.MIN_HEIGHT:
        reasons.append(
            f"Resolution too low: {meta['width']}x{meta['height']} "
            f"(minimum {config.MIN_WIDTH}x{config.MIN_HEIGHT})"
        )

    # 3. FPS
    if meta["fps"] < config.MIN_FPS:
        reasons.append(
            f"Frame rate too low: {meta['fps']:.1f} FPS "
            f"(minimum {config.MIN_FPS} FPS)"
        )

    # 4. Enough frames to analyse
    if len(frames) < 2:
        reasons.append("Could not extract enough frames — file may be corrupted.")

    # 5. Motion detection
    if len(frames) >= 2:
        motion_scores = _compute_motion_scores(frames)
        mean_motion = float(np.mean(motion_scores))
        if mean_motion < MOTION_THRESHOLD:
            reasons.append(
                f"No significant motion detected (mean frame diff = {mean_motion:.2f}). "
                "Video appears to be static."
            )

    status = "FAIL" if reasons else "PASS"
    if status == "FAIL":
        logger.warning(f"Hard QC FAILED: {reasons}")
    else:
        logger.debug("Hard QC PASSED.")

    return {"hard_qc_status": status, "reasons": reasons}
