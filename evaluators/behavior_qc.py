"""
evaluators/behavior_qc.py
--------------------------
Human Behaviour Score — evaluates naturalness and correctness of human actions.

Methods:
  A. Naturalness      → optical flow smoothness (variance)
  B. Speed check      → action duration vs expected
  C. Temporal consistency → motion pattern regularity

Note: Full action recognition (Video Swin / I3D) requires a GPU and
Kinetics pre-trained weights. We include the integration skeleton and
use optical-flow-based heuristics as the primary signal so the pipeline
runs on CPU as well.

Output: behavior_score ∈ [0, 1], fail_reasons list
"""

from __future__ import annotations

import numpy as np

import config
from utils.logger import logger


# ── A. Optical flow smoothness ────────────────────────────────────────────────

def _optical_flow_smoothness(frames: list[np.ndarray]) -> float:
    """
    Compute Farneback optical flow between consecutive frames.
    Returns a smoothness score in [0, 1]:
      high variance → jerky/unnatural movement → low score
      low variance  → smooth movement          → high score
    """
    import cv2

    if len(frames) < 2:
        return 0.5

    magnitudes: list[float] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))
        prev_gray = gray

    if not magnitudes:
        return 0.5

    mean_mag = float(np.mean(magnitudes))
    var_mag = float(np.var(magnitudes))

    # Normalise variance: high var → unnatural. Cap variance effect at 50.
    smoothness = float(np.clip(1.0 - var_mag / 50.0, 0.0, 1.0))
    return smoothness


# ── B. Speed check ────────────────────────────────────────────────────────────

def _speed_score(frames: list[np.ndarray], duration_sec: float) -> tuple[float, list[str]]:
    """
    Compare mean optical-flow speed with a reference "normal" egocentric speed.
    Returns (score, reasons).
    """
    import cv2

    reasons: list[str] = []
    if len(frames) < 2 or duration_sec <= 0:
        return 0.5, []

    # Count "active" (high motion) frames
    MOTION_TH = 3.0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    active_count = 0
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = float(np.mean(np.abs(gray - prev_gray)))
        if diff > MOTION_TH:
            active_count += 1
        prev_gray = gray

    active_ratio = active_count / (len(frames) - 1)

    # Heuristic: for typical task videos (5–120 s), expect 30–90% active frames
    if active_ratio < 0.20:
        reasons.append("Action too slow or too little movement recorded.")
        score = 0.3
    elif active_ratio > 0.98:
        reasons.append("Unnaturally constant motion — possible replay/loop artefact.")
        score = 0.5
    else:
        # Linear mapping: 0.20 → 0.4, 0.60 → 1.0, > 0.95 penalised above
        score = float(np.clip((active_ratio - 0.20) / 0.60 + 0.40, 0.0, 1.0))

    return score, reasons


# ── C. Temporal consistency ───────────────────────────────────────────────────

def _temporal_consistency(frames: list[np.ndarray]) -> float:
    """
    Check that the motion pattern across the video is reasonably consistent
    (not abrupt spikes or dead stops).
    Returns score in [0, 1].
    """
    import cv2

    if len(frames) < 4:
        return 0.7

    diffs: list[float] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(float(np.mean(np.abs(gray - prev_gray))))
        prev_gray = gray

    if not diffs:
        return 0.7

    # Large standard deviation relative to mean → inconsistent
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs))
    cv = std_d / (mean_d + 1e-6)  # coefficient of variation

    # cv < 0.5 → consistent, cv > 2.0 → very inconsistent
    score = float(np.clip(1.0 - (cv - 0.5) / 1.5, 0.0, 1.0))
    return score


# ── Main entry ────────────────────────────────────────────────────────────────

def evaluate_behavior(
    frames: list[np.ndarray],
    meta: dict,
) -> dict:
    """
    Returns:
      {
        "behavior_score": float,
        "fail_reasons": list[str],
        "detail": {...}
      }
    """
    fail_reasons: list[str] = []

    # A. Smoothness
    smoothness = _optical_flow_smoothness(frames)
    if smoothness < 0.30:
        fail_reasons.append("Unnatural human movement — excessive jerkiness detected.")

    # B. Speed
    speed_sc, speed_reasons = _speed_score(frames, meta.get("duration_sec", 0))
    fail_reasons.extend(speed_reasons)

    # C. Temporal consistency
    consistency = _temporal_consistency(frames)
    if consistency < 0.30:
        fail_reasons.append("Temporally inconsistent motion pattern — abrupt cuts or loops suspected.")

    behavior_score = float(
        0.40 * smoothness +
        0.35 * speed_sc +
        0.25 * consistency
    )

    logger.info(
        f"Behavior QC → smoothness={smoothness:.2f}, speed={speed_sc:.2f}, "
        f"consistency={consistency:.2f} → behavior_score={behavior_score:.3f}"
    )

    return {
        "behavior_score": round(behavior_score, 4),
        "fail_reasons": fail_reasons,
        "detail": {
            "smoothness": round(smoothness, 4),
            "speed_score": round(speed_sc, 4),
            "temporal_consistency": round(consistency, 4),
        },
    }
