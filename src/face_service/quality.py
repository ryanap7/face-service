"""Whether a detected face is good enough to enrol or to recognise."""
from typing import Dict, List

import cv2
import numpy as np

from face_service import config

# Nose height between eyes and mouth on a level face.
LEVEL_NOSE = 0.55


def pose(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Rough head turn from the five YuNet landmarks (right eye, left eye,
    nose, right and left mouth corner): how far the nose sits off the
    eyes' centre, in eye distances, and how high between eyes and mouth.
    """
    eye_r, eye_l, nose, mouth_r, mouth_l = landmarks
    eyes = (eye_r + eye_l) / 2
    mouth = (mouth_r + mouth_l) / 2
    eye_dist = max(float(np.linalg.norm(eye_l - eye_r)), 1.0)
    span = max(float(mouth[1] - eyes[1]), 1.0)
    return {
        'yaw': round(float(nose[0] - eyes[0]) / eye_dist, 3),
        'pitch': round(float(nose[1] - eyes[1]) / span - LEVEL_NOSE, 3),
    }


def assess(face_box: np.ndarray, landmarks: np.ndarray, aligned: np.ndarray) -> Dict:
    """Measures and the issues that keep a face from being used."""
    grey = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    size = int(min(face_box[2], face_box[3]))
    brightness = float(grey.mean())
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    turn = pose(landmarks)

    issues: List[str] = []
    if size < config.MIN_FACE_PX:
        issues.append('FACE_TOO_SMALL')
    if brightness < config.MIN_BRIGHTNESS:
        issues.append('TOO_DARK')
    if brightness > config.MAX_BRIGHTNESS:
        issues.append('TOO_BRIGHT')
    if sharpness < config.MIN_SHARPNESS:
        issues.append('BLURRY')
    if abs(turn['yaw']) > config.MAX_YAW or abs(turn['pitch']) > config.MAX_PITCH:
        issues.append('NOT_FRONTAL')
    return {
        'ok': not issues,
        'issues': issues,
        'faceSize': size,
        'brightness': round(brightness, 1),
        'sharpness': round(sharpness, 1),
        **turn,
    }
