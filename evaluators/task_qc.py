"""
evaluators/task_qc.py
---------------------
Task Completion Score — evaluates whether the recorded session matches
the intended task.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch
from PIL import Image

import config
from utils.logger import logger

# ── Lazy model globals ────────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None
_blip_processor = None
_blip_model = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        logger.info("Loading CLIP model (via Transformers) …")
        model_id = "openai/clip-vit-base-patch32"
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(config.DEVICE)
        logger.success("CLIP loaded.")
    return _clip_model, _clip_processor


def _load_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        logger.info("Loading BLIP model …")
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(config.DEVICE)
        _blip_model.eval()
        logger.success("BLIP loaded.")
    return _blip_processor, _blip_model


def _get_features(features_obj):
    """Helper to extract the tensor from CLIP model output."""
    if hasattr(features_obj, "pooler_output"):
        return features_obj.pooler_output
    return features_obj


# ── A. CLIP similarity ────────────────────────────────────────────────────────

def _clip_similarity(frames: list[np.ndarray], task_description: str) -> float:
    """Return mean cosine similarity between frames and the task text."""
    import cv2

    model, processor = _load_clip()
    device = config.DEVICE

    sims: list[float] = []
    # Sample at most 16 frames to keep it fast
    step = max(1, len(frames) // 16)
    
    # Process text once
    text_inputs = processor(text=[task_description], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = _get_features(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    for frame in frames[::step]:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = _get_features(model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        sim = (image_features @ text_features.T).item()
        sims.append(sim)

    return float(np.mean(sims)) if sims else 0.0


# ── B. BLIP captioning + keyword coverage ────────────────────────────────────

def _blip_caption_coverage(frames: list[np.ndarray], task_keywords: list[str]) -> float:
    """
    Caption a subset of frames with BLIP.
    Score = fraction of task keywords found across all captions.
    """
    import cv2

    if not task_keywords:
        return 0.5  # neutral when no keywords given

    processor, model = _load_blip()
    device = config.DEVICE

    captions: list[str] = []
    step = max(1, len(frames) // 8)
    for frame in frames[::step]:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(pil, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True).lower()
        captions.append(caption)
        logger.debug(f"BLIP caption: {caption}")

    all_text = " ".join(captions)
    matched = sum(1 for kw in task_keywords if kw.lower() in all_text)
    return matched / len(task_keywords)


# ── C. Action coverage ────────────────────────────────────────────────────────

def _action_coverage(frames: list[np.ndarray]) -> float:
    """
    Fraction of frame pairs that have meaningful motion.
    """
    import cv2

    if len(frames) < 2:
        return 0.0

    MOTION_TH = 5.0
    active = 0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.mean(np.abs(gray - prev_gray))
        if diff > MOTION_TH:
            active += 1
        prev_gray = gray
    return active / (len(frames) - 1)


# ── Main entry ────────────────────────────────────────────────────────────────

def evaluate_task(
    frames: list[np.ndarray],
    task_description: str = "",
    task_keywords: Optional[list[str]] = None,
) -> dict:
    """
    Main task evaluation entry point.
    """
    task_keywords = task_keywords or []
    fail_reasons: list[str] = []

    # A. CLIP
    clip_sim_norm = 0.5
    if task_description:
        try:
            clip_sim = _clip_similarity(frames, task_description)
            # Normalise to 0–1 with a generous linear map
            clip_sim_norm = float(np.clip((clip_sim + 0.3) / 0.65, 0.0, 1.0))
        except Exception as e:
            logger.error(f"CLIP evaluation failed: {e}")
            clip_sim_norm = 0.5
    
    # B. BLIP keyword coverage
    try:
        keyword_cov = _blip_caption_coverage(frames, task_keywords)
    except Exception as e:
        logger.error(f"BLIP captioning failed: {e}")
        keyword_cov = 0.5

    # C. Action coverage
    action_cov = _action_coverage(frames)

    # Combine (weighted average)
    task_score = float(
        0.45 * clip_sim_norm +
        0.35 * keyword_cov +
        0.20 * action_cov
    )

    if task_score < 0.40:
        fail_reasons.append("Task matching failed — visual content does not match task description.")
    if action_cov < 0.30:
        fail_reasons.append("Insufficient action coverage — video is too static.")

    logger.info(
        f"Task QC → clip_sim={clip_sim_norm:.2f}, keyword_cov={keyword_cov:.2f}, "
        f"action_cov={action_cov:.2f} → task_score={task_score:.3f}"
    )

    return {
        "task_score": round(task_score, 4),
        "fail_reasons": fail_reasons,
        "detail": {
            "clip_similarity_norm": round(clip_sim_norm, 4),
            "keyword_coverage": round(keyword_cov, 4),
            "action_coverage": round(action_cov, 4),
        },
    }
