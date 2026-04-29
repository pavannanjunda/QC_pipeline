"""
evaluators/semantic_qc.py
--------------------------
Semantic Evaluation Score — checks that the video content is logically
coherent and aligned with the task description.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import config
from utils.logger import logger

# ── Lazy globals ──────────────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None


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


def _get_features(model, features_obj):
    """Helper to extract the tensor from CLIP model output."""
    if hasattr(features_obj, "pooler_output"):
        return features_obj.pooler_output
    return features_obj


# ── A. Task alignment ─────────────────────────────────────────────────────────

def _task_alignment_score(frames: list[np.ndarray], task_description: str) -> float:
    """CLIP-based alignment between sampled frames and task description."""
    import cv2

    if not task_description:
        return 0.5

    model, processor = _load_clip()
    device = config.DEVICE

    text_inputs = processor(text=[task_description], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = _get_features(model, model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    step = max(1, len(frames) // 12)
    sims: list[float] = []
    for frame in frames[::step]:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = _get_features(model, model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        sim = (image_features @ text_features.T).item()
        sims.append(sim)

    raw_sim = float(np.mean(sims)) if sims else 0.0
    return float(np.clip((raw_sim + 0.3) / 0.65, 0.0, 1.0))


# ── B. Logical coherence ──────────────────────────────────────────────────────

def _logical_coherence_score(frames: list[np.ndarray]) -> float:
    """
    Check temporal smoothness of CLIP embeddings.
    """
    import cv2

    model, processor = _load_clip()
    device = config.DEVICE

    step = max(1, len(frames) // 12)
    sampled = frames[::step]

    if len(sampled) < 2:
        return 0.7

    feats: list[torch.Tensor] = []
    for frame in sampled:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = _get_features(model, model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        feats.append(image_features)

    consecutive_sims: list[float] = []
    for i in range(len(feats) - 1):
        sim = (feats[i] @ feats[i + 1].T).item()
        consecutive_sims.append(sim)

    mean_coherence = float(np.mean(consecutive_sims))
    score = float(np.clip((mean_coherence - 0.70) / 0.25, 0.0, 1.0))
    return score


# ── C. Irrelevance penalty ────────────────────────────────────────────────────

def _irrelevance_penalty(frames: list[np.ndarray], task_description: str) -> float:
    """
    Identify frames that are very low similarity to the task description.
    """
    import cv2

    if not task_description:
        return 0.8

    model, processor = _load_clip()
    device = config.DEVICE

    text_inputs = processor(text=[task_description], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = _get_features(model, model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    step = max(1, len(frames) // 12)
    IRRELEVANCE_TH = -0.05

    irrelevant = 0
    total = 0
    for frame in frames[::step]:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = _get_features(model, model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        sim = (image_features @ text_features.T).item()
        if sim < IRRELEVANCE_TH:
            irrelevant += 1
        total += 1

    if total == 0:
        return 0.8

    irrelevance_ratio = irrelevant / total
    return float(np.clip(1.0 - irrelevance_ratio * 1.5, 0.0, 1.0))


# ── Main entry ────────────────────────────────────────────────────────────────

def evaluate_semantic(frames: list[np.ndarray], task_description: str = "") -> dict:
    fail_reasons: list[str] = []

    # A. Task alignment
    try:
        alignment = _task_alignment_score(frames, task_description)
    except Exception as e:
        logger.error(f"Semantic alignment failed: {e}")
        alignment = 0.5

    # B. Logical coherence
    try:
        coherence = _logical_coherence_score(frames)
    except Exception as e:
        logger.error(f"Coherence check failed: {e}")
        coherence = 0.5

    # C. Irrelevance penalty
    try:
        relevance = _irrelevance_penalty(frames, task_description)
    except Exception as e:
        logger.error(f"Irrelevance check failed: {e}")
        relevance = 0.8

    semantic_score = float(0.45 * alignment + 0.30 * coherence + 0.25 * relevance)

    if alignment < 0.35:
        fail_reasons.append("Semantic mismatch — video content does not match task description.")
    if coherence < 0.30:
        fail_reasons.append("Logical sequence broken — inconsistent scene flow.")
    if relevance < 0.40:
        fail_reasons.append("Irrelevant actions detected.")

    logger.info(f"Semantic QC → alignment={alignment:.2f}, coherence={coherence:.2f}, relevance={relevance:.2f} → semantic_score={semantic_score:.3f}")

    return {
        "semantic_score": round(semantic_score, 4),
        "fail_reasons": fail_reasons,
        "detail": {
            "task_alignment": round(alignment, 4),
            "logical_coherence": round(coherence, 4),
            "relevance_score": round(relevance, 4),
        },
    }
