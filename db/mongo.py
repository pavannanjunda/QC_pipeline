"""
db/mongo.py
-----------
All MongoDB interactions for the QC pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

import config
from utils.logger import logger


# ─── Connection ───────────────────────────────────────────────────────────────

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        logger.info("Connecting to MongoDB Atlas …")
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=10_000)
        # Force connection check
        _client.admin.command("ping")
        logger.success("MongoDB connected.")
    return _client


def get_collection() -> Collection:
    return get_client()[config.MONGO_DB_NAME]["sessions"]


# ─── Fetch ────────────────────────────────────────────────────────────────────

def iter_pending_sessions(batch_size: int = 0) -> Iterator[dict]:
    """
    Yield session documents that:
      • have been successfully uploaded (upload_status = "Uploaded")
      • have not yet been QC-scored (qc_score is None, NaN, or 0)
    """
    col = get_collection()
    
    # Core query for pending QC
    # We allow 0/0.0 and we temporarily ignore annotation_status to allow re-testing
    query = {
        "upload_status": "Uploaded",
        "qc_score": {"$in": [None, float("nan"), 0, 0.0, 0]},
    }
    
    # Handle the case where the field is missing entirely
    query_missing = {
        "upload_status": "Uploaded",
        "qc_score": {"$exists": False},
    }

    cursor = col.find({"$or": [query, query_missing]})
    if batch_size > 0:
        cursor = cursor.limit(batch_size)

    for doc in cursor:
        yield doc


# ─── Write results ────────────────────────────────────────────────────────────

def write_qc_result(session_id: str, result: dict) -> None:
    """
    Persist the QC result back to the sessions document.
    """
    col = get_collection()

    status = result.get("status", "FAIL")
    final_score = result.get("final_score", 0.0)
    fail_reasons = result.get("fail_reasons", [])

    new_annotation_status = "Rejected" if status == "FAIL" else "QA-Pending"

    update = {
        "$set": {
            "qc_score": round(final_score, 4),
            "rejection_reason": "; ".join(fail_reasons) if fail_reasons else None,
            "annotation_status": new_annotation_status,
            "metadata.qc_detail": {
                **result,
                "evaluated_at": datetime.utcnow().isoformat(),
            },
        }
    }

    res = col.update_one({"_id": session_id}, update)
    if res.matched_count == 0:
        logger.warning(f"No document matched _id={session_id} during QC write-back.")
    else:
        logger.info(
            f"[{session_id}] QC written → score={final_score:.3f}, "
            f"status={status}, annotation_status={new_annotation_status}"
        )


def bulk_write_qc_results(results: list[tuple[str, dict]]) -> None:
    """Batch update multiple session results in one round-trip."""
    if not results:
        return
    col = get_collection()
    ops = []
    for session_id, result in results:
        status = result.get("status", "FAIL")
        final_score = result.get("final_score", 0.0)
        fail_reasons = result.get("fail_reasons", [])
        new_annotation_status = "Rejected" if status == "FAIL" else "QA-Pending"
        ops.append(
            UpdateOne(
                {"_id": session_id},
                {
                    "$set": {
                        "qc_score": round(final_score, 4),
                        "rejection_reason": "; ".join(fail_reasons) or None,
                        "annotation_status": new_annotation_status,
                        "metadata.qc_detail": {
                            **result,
                            "evaluated_at": datetime.utcnow().isoformat(),
                        },
                    }
                },
            )
        )
    col.bulk_write(ops, ordered=False)
    logger.info(f"Bulk wrote {len(ops)} QC results to MongoDB.")


# ─── Task Metadata ────────────────────────────────────────────────────────────

def get_task_metadata(task_id: str) -> dict:
    """Fetch task details from tasks collection."""
    if not task_id:
        return {}
    db = get_client()[config.MONGO_DB_NAME]
    task = db["tasks"].find_one({"_id": task_id})
    if not task:
        task = db["tasks"].find_one({"task_code": task_id})
        
    if task:
        return {
            "title": task.get("task_title", ""),
            "description": task.get("task_description", ""),
            "keywords": task.get("task_tags", [])
        }
    return {}
