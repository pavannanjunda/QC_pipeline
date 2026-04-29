"""
api.py
------
FastAPI server to provide QC pipeline statistics.
"""

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict

load_dotenv()

app = FastAPI(title="QC Pipeline Analytics API")

# DB Connection
try:
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["test"]
    sessions_col = db["sessions"]
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

@app.get("/stats/duration")
async def get_duration_stats():
    """
    Returns total video duration aggregated by Day, Week, and Month.
    """
    try:
        # 1. Total Duration (All time)
        pipeline_total = [
            {"$match": {"metadata.qc_detail.meta.duration_sec": {"$exists": True}}},
            {"$group": {"_id": None, "total_sec": {"$sum": "$metadata.qc_detail.meta.duration_sec"}}}
        ]
        total_res = list(sessions_col.aggregate(pipeline_total))
        total_sec = total_res[0]["total_sec"] if total_res else 0

        # 2. Daily Duration (Last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        pipeline_daily = [
            {
                "$match": {
                    "createdAt": {"$gte": seven_days_ago},
                    "metadata.qc_detail.meta.duration_sec": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
                    "duration_sec": {"$sum": "$metadata.qc_detail.meta.duration_sec"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        daily_res = {item["_id"]: round(item["duration_sec"] / 3600, 2) for item in sessions_col.aggregate(pipeline_daily)}

        # 3. Weekly Duration (Last 4 weeks)
        four_weeks_ago = datetime.utcnow() - timedelta(weeks=4)
        pipeline_weekly = [
            {
                "$match": {
                    "createdAt": {"$gte": four_weeks_ago},
                    "metadata.qc_detail.meta.duration_sec": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-W%U", "date": "$createdAt"}},
                    "duration_sec": {"$sum": "$metadata.qc_detail.meta.duration_sec"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        weekly_res = {item["_id"]: round(item["duration_sec"] / 3600, 2) for item in sessions_col.aggregate(pipeline_weekly)}

        # 4. Monthly Duration (Last 6 months)
        six_months_ago = datetime.utcnow() - timedelta(days=180)
        pipeline_monthly = [
            {
                "$match": {
                    "createdAt": {"$gte": six_months_ago},
                    "metadata.qc_detail.meta.duration_sec": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m", "date": "$createdAt"}},
                    "duration_sec": {"$sum": "$metadata.qc_detail.meta.duration_sec"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        monthly_res = {item["_id"]: round(item["duration_sec"] / 3600, 2) for item in sessions_col.aggregate(pipeline_monthly)}

        return {
            "total_hours": round(total_sec / 3600, 2),
            "daily_hours": daily_res,
            "weekly_hours": weekly_res,
            "monthly_hours": monthly_res,
            "unit": "hours"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "QC Pipeline API is online", "endpoints": ["/stats/duration"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
