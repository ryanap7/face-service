"""
Passive anti-spoofing with MiniFASNet (Silent-Face-Anti-Spoofing,
Apache-2.0): two models look at the face with different margins around it
and together say whether it is a live person or a photo or screen.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from face_service import config

INPUT = 80
REAL = 1


def crop_box(src_w: int, src_h: int, box: Tuple[float, float, float, float], scale: float):
    """The official crop: the face box grown by `scale`, kept in the image."""
    x, y, box_w, box_h = box
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = box_w / 2 + x, box_h / 2 + y
    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    return int(left), int(top), int(right), int(bottom)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def judge(image: np.ndarray, box, sessions: List[Tuple[object, float]]) -> Dict:
    h, w = image.shape[:2]
    total = np.zeros((1, 3), dtype=np.float32)
    for session, scale in sessions:
        left, top, right, bottom = crop_box(w, h, tuple(float(v) for v in box), scale)
        patch = cv2.resize(image[top:bottom + 1, left:right + 1], (INPUT, INPUT))
        # As trained: BGR pixel values, not divided by 255.
        tensor = patch.transpose(2, 0, 1)[None].astype(np.float32)
        total += _softmax(session.run(None, {'input': tensor})[0])
    label = int(total.argmax())
    confidence = float(total[0, label] / len(sessions))
    real = float(total[0, REAL] / len(sessions))
    if label == REAL and confidence >= config.LIVENESS_REAL:
        verdict = 'REAL'
    elif label != REAL and confidence >= config.LIVENESS_SPOOF:
        verdict = 'SPOOF'
    else:
        verdict = 'UNSURE'
    return {'verdict': verdict, 'real': round(real, 4)}
