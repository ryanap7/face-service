import os
from pathlib import Path

MODELS_DIR = Path(os.getenv('FACE_MODELS_DIR', Path(__file__).resolve().parents[2] / 'models'))

DETECTOR = 'face_detection_yunet_2023mar.onnx'
RECOGNIZER = 'face_recognition_sface_2021dec.onnx'
# (file, how far around the face the model looks)
LIVENESS = [('2.7_80x80_MiniFASNetV2.onnx', 2.7), ('4_0_0_80x80_MiniFASNetV1SE.onnx', 4.0)]
# Embeddings are only comparable within one version.
MODEL_VERSION = 'yunet-2023mar+sface-2021dec'

API_KEY = os.getenv('FACE_API_KEY', '')
POOL_SIZE = int(os.getenv('FACE_POOL_SIZE', '4'))
MAX_IMAGE_BYTES = 5 * 1024 * 1024
# Larger frames are scaled down first; a face fills plenty of pixels at this size.
MAX_SIDE = 1280

DETECTION_SCORE = 0.8
# A second face this large next to the first means someone else is in frame.
SECOND_FACE_RATIO = 0.5
MIN_FACE_PX = 90
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 200
# Variance of the Laplacian on the 112 px aligned face; low means blurred.
MIN_SHARPNESS = 25.0
# Nose offset from the eyes' centre, in eye distances; beyond this the
# face is turned too far for a reliable match.
MAX_YAW = 0.35
MAX_PITCH = 0.45
LIVENESS_REAL = 0.8
LIVENESS_SPOOF = 0.8
