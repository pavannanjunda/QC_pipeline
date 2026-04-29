"""
config.py
---------
Centralised configuration loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI: str = os.getenv("MONGO_URI", "")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "test")

# ── Preprocessing ─────────────────────────────────────────────────────────────
FRAME_SAMPLE_FPS: int = int(os.getenv("FRAME_SAMPLE_FPS", 2))
VIDEO_BASE_DIR: str = os.getenv("VIDEO_BASE_DIR", "/mnt/nas/xp-capture/xp-capture-staging/Validating")
TEMP_DIR: str = os.getenv("TEMP_DIR", "/home/simpel/eval_pipeline/temp_extract")

# ── Hard QC thresholds ────────────────────────────────────────────────────────
MIN_DURATION_SEC: float = float(os.getenv("MIN_DURATION_SEC", 5))
MIN_WIDTH: int = int(os.getenv("MIN_WIDTH", 1920))
MIN_HEIGHT: int = int(os.getenv("MIN_HEIGHT", 1080))
MIN_FPS: float = float(os.getenv("MIN_FPS", 30))

# ── Scoring ───────────────────────────────────────────────────────────────────
PASS_THRESHOLD: float = float(os.getenv("PASS_THRESHOLD", 0.75))
WARNING_THRESHOLD: float = float(os.getenv("WARNING_THRESHOLD", 0.50))

# ── Runtime ───────────────────────────────────────────────────────────────────
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 0))
DEVICE: str = os.getenv("DEVICE", "cpu")
