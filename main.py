"""
main.py
-------
Entry point for the QC Evaluation Pipeline.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback

from tqdm import tqdm

import config
from db.mongo import iter_pending_sessions, write_qc_result
from pipeline.orchestrator import run_pipeline
from utils.logger import logger


def process_batch(batch_size: int) -> int:
    """
    Fetch and process one batch of pending sessions.
    Returns the number of sessions processed.
    """
    sessions = list(iter_pending_sessions(batch_size=batch_size))

    if not sessions:
        logger.info("No pending sessions found.")
        return 0

    logger.info(f"Found {len(sessions)} session(s) to process.")
    processed = 0

    for session in tqdm(sessions, desc="QC Processing", unit="video"):
        session_id = session["_id"]
        video_uuid = session.get("video_uuid", str(session_id))
        device_brand = session.get("device_brand", "unknown")

        logger.info(
            f"Processing session [{video_uuid}] | "
            f"device={device_brand} | task_id={session.get('task_id')}"
        )

        try:
            result = run_pipeline(session)
            write_qc_result(session_id, result)

            # Pretty-print summary
            logger.info(
                f"\n{'─'*60}\n"
                f"  VIDEO      : {result['video_name']}\n"
                f"  Task       : {result['task_score']:.3f}\n"
                f"  Behavior   : {result['behavior_score']:.3f}\n"
                f"  Validity   : {result['validity_score']:.3f}\n"
                f"  Semantic   : {result['semantic_score']:.3f}\n"
                f"  FINAL      : {result['final_score']:.3f}  →  {result['status']}\n"
                f"  Reasons    : {result.get('fail_reasons', [])}\n"
                f"{'─'*60}"
            )
            processed += 1

        except Exception as exc:
            logger.error(
                f"Unhandled error for session [{video_uuid}]: {exc}\n"
                + traceback.format_exc()
            )
            # Write a FAIL result so the session isn't stuck in limbo
            fallback = {
                "video_name": video_uuid,
                "task_score": 0.0,
                "behavior_score": 0.0,
                "validity_score": 0.0,
                "semantic_score": 0.0,
                "final_score": 0.0,
                "status": "FAIL",
                "fail_reasons": [f"Pipeline error: {exc}"],
            }
            try:
                write_qc_result(session_id, fallback)
            except Exception:
                pass  # best-effort

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="QC Evaluation Pipeline for Egocentric Videos"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously, polling MongoDB every --interval seconds.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds when running as daemon (default: 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Number of sessions to process in one batch (default: {config.BATCH_SIZE}).",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  QC Evaluation Pipeline  —  starting up")
    logger.info(f"  Device  : {config.DEVICE}")
    logger.info(f"  Batch   : {args.batch_size or 'unlimited'}")
    logger.info(f"  DB      : {config.MONGO_DB_NAME}")
    logger.info(f"  NAS Root: {config.VIDEO_BASE_DIR}")
    logger.info("=" * 60)

    if args.daemon:
        logger.info(f"Daemon mode ON — polling every {args.interval}s")
        while True:
            try:
                n = process_batch(args.batch_size)
                logger.info(f"Batch complete — processed {n} session(s). Sleeping …")
            except KeyboardInterrupt:
                logger.info("Daemon stopped by user.")
                break
            except Exception as e:
                logger.error(f"Batch-level error: {e}")
            time.sleep(args.interval)
    else:
        n = process_batch(args.batch_size)
        logger.info(f"Done — processed {n} session(s).")


if __name__ == "__main__":
    main()
