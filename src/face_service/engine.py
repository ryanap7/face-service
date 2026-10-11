"""One set of models; a pool of them serves requests side by side."""
import queue
from typing import Callable, Dict, Optional, TypeVar

import cv2
import numpy as np
import onnxruntime as ort

from face_service import config, liveness, quality

T = TypeVar('T')


class FaceEngine:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            str(config.MODELS_DIR / config.DETECTOR), '', (320, 320),
            score_threshold=config.DETECTION_SCORE, nms_threshold=0.3, top_k=20,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(config.MODELS_DIR / config.RECOGNIZER), '')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.liveness = [
            (ort.InferenceSession(str(config.MODELS_DIR / name), options, providers=['CPUExecutionProvider']), scale)
            for name, scale in config.LIVENESS
        ]

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces if faces is not None else np.zeros((0, 15), dtype=np.float32)

    def analyze(self, image: np.ndarray, with_liveness: bool = True) -> Dict:
        faces = self.detect(image)
        result: Dict = {
            'model': config.MODEL_VERSION,
            'faces': int(len(faces)),
            'face': None,
            'quality': {'ok': False, 'issues': ['NO_FACE']},
            'liveness': None,
            'embedding': None,
        }
        if not len(faces):
            return result

        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(-areas)
        face = faces[order[0]]
        landmarks = face[4:14].reshape(5, 2)
        aligned = self.recognizer.alignCrop(image, face)
        assessed = quality.assess(face[:4], landmarks, aligned)
        if len(faces) > 1 and areas[order[1]] >= config.SECOND_FACE_RATIO * areas[order[0]]:
            assessed['issues'].insert(0, 'MULTIPLE_FACES')
            assessed['ok'] = False

        result['face'] = {
            'box': [round(float(v), 1) for v in face[:4]],
            'score': round(float(face[14]), 4),
            'landmarks': [[round(float(x), 1), round(float(y), 1)] for x, y in landmarks],
        }
        result['quality'] = assessed
        if with_liveness:
            result['liveness'] = liveness.judge(image, face[:4], self.liveness)
        if assessed['ok']:
            feature = self.recognizer.feature(aligned).flatten()
            feature = feature / max(float(np.linalg.norm(feature)), 1e-9)
            result['embedding'] = [round(float(v), 6) for v in feature]
        return result


class EnginePool:
    """Engines are not thread-safe; each request borrows one."""

    def __init__(self, size: int, factory: Callable[[], object] = FaceEngine):
        self._engines: 'queue.Queue' = queue.Queue()
        for _ in range(size):
            self._engines.put(factory())

    def run(self, work: Callable[[object], T], timeout: Optional[float] = 10.0) -> T:
        engine = self._engines.get(timeout=timeout)
        try:
            return work(engine)
        finally:
            self._engines.put(engine)


def decode(data: bytes) -> Optional[np.ndarray]:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    longest = max(image.shape[:2])
    if longest > config.MAX_SIDE:
        factor = config.MAX_SIDE / longest
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return image
