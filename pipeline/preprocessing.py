"""
pipeline/preprocessing.py
--------------------------
Frame extraction, video metadata retrieval, and ZIP handling.
"""

from __future__ import annotations

import os
import zipfile
import shutil
from typing import Any, Optional, Callable

import cv2
import numpy as np

import config
from utils.logger import logger


def prepare_video_file(raw_path: str, session_id: str = "") -> tuple[str, Optional[Callable]]:
    """
    Resolves the video path. 
    If the database path fails, it tries to find a matching ZIP in VIDEO_BASE_DIR.
    """
    # 1. Try resolving exactly as stored in DB
    abs_path = os.path.join(config.VIDEO_BASE_DIR, raw_path.lstrip("/"))
    
    if not os.path.exists(abs_path):
        logger.warning(f"DB path not found: {abs_path}. Searching for fallback...")
        
        # 2. Fallback: Search for a ZIP file matching the session_id or its timestamp
        # Session IDs look like "Session_1777113622_0ZVYA"
        # Files look like "session_1777113622_iPhone.zip"
        found_zip = None
        
        # Extract the numeric timestamp part if possible
        parts = session_id.split('_')
        ts = parts[1] if len(parts) > 1 else session_id
        
        search_pattern = ts.lower()
        
        for file in os.listdir(config.VIDEO_BASE_DIR):
            if file.lower().endswith(".zip") and search_pattern in file.lower():
                found_zip = os.path.join(config.VIDEO_BASE_DIR, file)
                logger.success(f"Found matching ZIP via fallback: {found_zip}")
                break
        
        if not found_zip:
            raise FileNotFoundError(f"Could not find video or ZIP for session {session_id} in {config.VIDEO_BASE_DIR}")
        
        abs_path = found_zip

    # 3. Handle ZIP Extraction
    if abs_path.lower().endswith(".zip"):
        extract_path = os.path.join(config.TEMP_DIR, os.path.basename(abs_path).replace(".", "_"))
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path, exist_ok=True)
        
        logger.info(f"Extracting: {abs_path}")
        try:
            with zipfile.ZipFile(abs_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            raise RuntimeError(f"Failed to unzip {abs_path}: {e}")
        
        # Find largest video file, skipping macOS metadata
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        found_video = None
        largest_size = 0
        
        for root, _, files in os.walk(extract_path):
            if "__MACOSX" in root: continue
            for file in files:
                if file.startswith("._"): continue
                if file.lower().endswith(video_extensions):
                    fp = os.path.join(root, file)
                    sz = os.path.getsize(fp)
                    if sz > largest_size:
                        largest_size = sz
                        found_video = fp
        
        if not found_video:
            shutil.rmtree(extract_path)
            raise FileNotFoundError(f"No valid video found inside ZIP: {abs_path}")
        
        logger.info(f"Using video: {os.path.basename(found_video)} ({largest_size/1e6:.1f} MB)")
        return found_video, lambda: shutil.rmtree(extract_path)

    return abs_path, None


def extract_frames_and_metadata(video_path: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Open the video file, read its metadata, and sample frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec: float = (total_frames / native_fps) if native_fps > 0 else 0.0

    meta: dict[str, Any] = {
        "fps": native_fps, "width": width, "height": height,
        "duration_sec": duration_sec, "total_frames": total_frames,
    }

    # Frame sampling
    sample_every_n: int = max(1, int(round(native_fps / config.FRAME_SAMPLE_FPS)))
    frames: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_every_n == 0:
            frames.append(frame)
        frame_idx += 1
    cap.release()
    
    return frames, meta
