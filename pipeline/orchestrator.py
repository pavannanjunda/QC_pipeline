"""
pipeline/orchestrator.py
-------------------------
Main pipeline orchestrator.
"""

from __future__ import annotations

import os
from typing import Optional

import config
from utils.logger import logger
from db.mongo import get_task_metadata
from pipeline.preprocessing import extract_frames_and_metadata, prepare_video_file
from pipeline.hard_qc import run_hard_qc
from evaluators.task_qc import evaluate_task
from evaluators.behavior_qc import evaluate_behavior
from evaluators.validity_qc import evaluate_validity
from evaluators.semantic_qc import evaluate_semantic


# ── Scoring ───────────────────────────────────────────────────────────────────

def _compute_final_score(
    task_score: float,
    behavior_score: float,
    validity_score: float,
    semantic_score: float,
) -> float:
    return round(
        0.4 * task_score +
        0.3 * behavior_score +
        0.2 * validity_score +
        0.1 * semantic_score,
        4,
    )


def _decide_status(final_score: float) -> str:
    if final_score >= config.PASS_THRESHOLD:
        return "PASS"
    elif final_score >= config.WARNING_THRESHOLD:
        return "WARNING"
    return "FAIL"


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(session: dict) -> dict:
    """
    Process one session document end-to-end.
    """
    session_id = session.get("_id")
    video_uuid = session.get("video_uuid", str(session_id))
    raw_path: Optional[str] = session.get("nas_file_path") or session.get("storage_dir")

    # Pull task metadata
    task_id = session.get("task_id")
    task_description: str = session.get("metadata", {}).get("task_description", "")
    task_keywords: list[str] = session.get("metadata", {}).get("task_keywords", [])
    
    if not task_description and task_id:
        task_meta = get_task_metadata(task_id)
        task_description = task_meta.get("description", "")
        task_keywords = task_meta.get("keywords", [])

    logger.info(f"[{video_uuid}] ─── Pipeline START ───")

    if not raw_path:
        return _hard_fail(video_uuid, "No video file path in session document.")

    cleanup_fn = None
    try:
        # Resolve absolute path and handle ZIP extraction
        video_path, cleanup_fn = prepare_video_file(raw_path, session_id=str(session_id))
        frames, meta = extract_frames_and_metadata(video_path)
        
    except FileNotFoundError as e:
        return _hard_fail(video_uuid, str(e))
    except Exception as e:
        return _hard_fail(video_uuid, f"Preprocessing failed: {e}")
    finally:
        if cleanup_fn:
            cleanup_fn()

    # ── STEP 3: Hard QC ───────────────────────────────────────────────────────
    hard_qc = run_hard_qc(frames, meta)
    if hard_qc["hard_qc_status"] == "FAIL":
        return {
            "video_name": video_uuid, "hard_qc_status": "FAIL",
            "task_score": 0.0, "behavior_score": 0.0,
            "validity_score": 0.0, "semantic_score": 0.0,
            "final_score": 0.0, "status": "FAIL",
            "fail_reasons": hard_qc["reasons"], "meta": meta,
        }

    # ── STEP 4: Core QC ───────────────────────────────────────────────────────
    task_result     = evaluate_task(frames, task_description, task_keywords)
    behavior_result = evaluate_behavior(frames, meta)
    validity_result = evaluate_validity(frames, meta)
    semantic_result = evaluate_semantic(frames, task_description)

    # ── STEP 5: Scoring ───────────────────────────────────────────────────────
    task_score     = task_result["task_score"]
    behavior_score = behavior_result["behavior_score"]
    validity_score = validity_result["validity_score"]
    semantic_score = semantic_result["semantic_score"]

    final_score = _compute_final_score(task_score, behavior_score, validity_score, semantic_score)
    status = _decide_status(final_score)

    all_reasons: list[str] = (
        task_result["fail_reasons"] + behavior_result["fail_reasons"] +
        validity_result["fail_reasons"] + semantic_result["fail_reasons"]
    )

    return {
        "video_name": video_uuid, "hard_qc_status": "PASS",
        "task_score": task_score, "behavior_score": behavior_score,
        "validity_score": validity_score, "semantic_score": semantic_score,
        "final_score": final_score, "status": status,
        "fail_reasons": all_reasons, "meta": meta,
        "detail": {
            "task": task_result.get("detail", {}),
            "behavior": behavior_result.get("detail", {}),
            "validity": validity_result.get("detail", {}),
            "semantic": semantic_result.get("detail", {}),
        },
    }


def _hard_fail(video_uuid: str, reason: str) -> dict:
    logger.error(f"[{video_uuid}] Hard fail: {reason}")
    return {
        "video_name": video_uuid, "hard_qc_status": "FAIL",
        "task_score": 0.0, "behavior_score": 0.0,
        "validity_score": 0.0, "semantic_score": 0.0,
        "final_score": 0.0, "status": "FAIL",
        "fail_reasons": [reason], "meta": {},
    }
