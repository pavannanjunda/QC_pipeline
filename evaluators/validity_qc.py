"""
evaluators/validity_qc.py
--------------------------
Recording Validity Score — checks that the recording is a genuine
first-person egocentric capture with visible hands and sufficient activity.

Methods:
  A. Hand detection        → MediaPipe Hands
  B. Hand position         → proximity to frame centre
  C. Effective duration    → fraction of frames with motion
  D. Live activity         → motion variance signal

Output: validity_score ∈ [0, 1], fail_reasons list
"""

from __future__ import annotations

import numpy as np
import cv2

import config
from utils.logger import logger

# ── Lazy MediaPipe loader ─────────────────────────────────────────────────────
_mp_hands = None
_hands_detector = None


def _get_hands():
    global _mp_hands, _hands_detector
    if _hands_detector is None:
        try:
            import mediapipe as mp
            from mediapipe.python.solutions import hands as mp_hands
            logger.info("Loading MediaPipe Hands …")
            _mp_hands = mp_hands
            _hands_detector = _mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
            logger.success("MediaPipe Hands loaded.")
        except ImportError as e:
            logger.error(f"MediaPipe import failed: {e}")
            raise e
    return _mp_hands, _hands_detector


# ── A & B. Hand detection + position ─────────────────────────────────────────

def _analyze_hands(frames: list[np.ndarray]) -> tuple[float, float, list[str]]:
    """
    Returns:
      hand_presence_ratio  : fraction of sampled frames where ≥1 hand detected
      hand_position_score  : 1.0 if hands near centre, lower if near edges
      fail_reasons         : list of strings
    """
    try:
        mp_h, detector = _get_hands()
    except Exception:
        return 0.5, 0.5, ["MediaPipe hand detection not available."]
        
    fail_reasons: list[str] = []

    # Sample at most 20 frames for speed
    step = max(1, len(frames) // 20)
    sampled = frames[::step]

    hand_detected_count = 0
    centre_scores: list[float] = []

    for frame in sampled:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        if result.multi_hand_landmarks:
            hand_detected_count += 1
            h, w = frame.shape[:2]
            for hand_lms in result.multi_hand_landmarks:
                xs = [lm.x for lm in hand_lms.landmark]
                ys = [lm.y for lm in hand_lms.landmark]
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))

                # Distance from centre (0.5, 0.5) normalised by 0.5
                dist_from_centre = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
                # Score: 1.0 at centre, 0.0 at corner
                centre_scores.append(float(np.clip(1.0 - dist_from_centre / 0.707, 0.0, 1.0)))

    hand_presence = hand_detected_count / len(sampled) if sampled else 0.0
    hand_pos_score = float(np.mean(centre_scores)) if centre_scores else 0.0

    if hand_presence < 0.30:
        fail_reasons.append(
            f"Hands frequently out of frame — detected in only "
            f"{hand_presence*100:.0f}% of sampled frames."
        )
    if centre_scores and hand_pos_score < 0.40:
        fail_reasons.append("Hands consistently near frame edges — poor egocentric framing.")

    return hand_presence, hand_pos_score, fail_reasons


# ── C. Effective duration (motion frames) ─────────────────────────────────────

def _effective_duration_ratio(frames: list[np.ndarray]) -> float:
    """Fraction of frame pairs with meaningful motion (> threshold)."""
    if len(frames) < 2:
        return 0.0

    MOTION_TH = 3.0
    active = 0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if np.mean(np.abs(gray - prev_gray)) > MOTION_TH:
            active += 1
        prev_gray = gray
    return active / (len(frames) - 1)


# ── D. Live activity (motion variance) ────────────────────────────────────────

def _live_activity_score(frames: list[np.ndarray]) -> float:
    """
    Motion variance across the whole video as a proxy for liveness.
    Returns score in [0, 1].
    """
    if len(frames) < 2:
        return 0.0

    diffs: list[float] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(float(np.mean(np.abs(gray - prev_gray))))
        prev_gray = gray

    mean_diff = float(np.mean(diffs))
    # Typical natural activity → mean_diff ~5–25; scale accordingly
    score = float(np.clip(mean_diff / 20.0, 0.0, 1.0))
    return score


# ── Main entry ────────────────────────────────────────────────────────────────

def evaluate_validity(frames: list[np.ndarray], meta: dict) -> dict:
    """
    Returns:
      {
        "validity_score": float,
        "fail_reasons": list[str],
        "detail": {...}
      }
    """
    fail_reasons: list[str] = []

    # A & B — hands
    try:
        hand_presence, hand_pos, hand_reasons = _analyze_hands(frames)
        fail_reasons.extend(hand_reasons)
    except Exception as e:
        logger.error(f"MediaPipe hand analysis failed: {e}")
        hand_presence, hand_pos = 0.5, 0.5

    # C — effective duration
    eff_ratio = _effective_duration_ratio(frames)
    if eff_ratio < 0.20:
        fail_reasons.append("Effective recording duration too low — most of the video is static.")

    # D — live activity
    live_score = _live_activity_score(frames)
    if live_score < 0.10:
        fail_reasons.append("No significant live activity detected.")

    validity_score = float(
        0.30 * hand_presence +
        0.20 * hand_pos +
        0.25 * eff_ratio +
        0.25 * live_score
    )

    logger.info(
        f"Validity QC → hands={hand_presence:.2f}, hand_pos={hand_pos:.2f}, "
        f"eff_ratio={eff_ratio:.2f}, live={live_score:.2f} → validity={validity_score:.3f}"
    )

    return {
        "validity_score": round(validity_score, 4),
        "fail_reasons": fail_reasons,
        "detail": {
            "hand_presence_ratio": round(hand_presence, 4),
            "hand_position_score": round(hand_pos, 4),
            "effective_duration_ratio": round(eff_ratio, 4),
            "live_activity_score": round(live_score, 4),
        },
    }
