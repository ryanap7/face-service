"""
The real models on public sample photos from the OpenCV repository
(tests/.cache, fetched with scripts/fetch_test_images.sh); skipped when
the photos or the downloaded model are missing.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from face_service import config

CACHE = Path(__file__).parent / '.cache'
pytestmark = pytest.mark.skipif(
    not (CACHE / 'lena.jpg').exists() or not (config.MODELS_DIR / config.RECOGNIZER).exists(),
    reason='sample photos or recognizer model missing',
)


@pytest.fixture(scope='module')
def engine():
    from face_service.engine import FaceEngine
    return FaceEngine()


def test_one_live_usable_face_with_a_unit_embedding(engine):
    result = engine.analyze(cv2.imread(str(CACHE / 'lena.jpg')))
    assert result['faces'] == 1 and result['quality']['ok']
    assert result['liveness']['verdict'] == 'REAL'
    assert len(result['embedding']) == 128
    assert abs(np.linalg.norm(result['embedding']) - 1) < 1e-3


def test_the_same_face_matches_and_another_does_not(engine):
    lena = cv2.imread(str(CACHE / 'lena.jpg'))
    same = engine.analyze(cv2.flip(lena, 1))['embedding']
    base = engine.analyze(lena)['embedding']
    messi = cv2.imread(str(CACHE / 'messi5.jpg'))
    other = engine.analyze(cv2.resize(messi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))['embedding']
    assert np.dot(base, same) > 0.6
    assert np.dot(base, other) < 0.3


def test_a_face_too_small_gives_no_embedding(engine):
    result = engine.analyze(cv2.imread(str(CACHE / 'messi5.jpg')))
    assert 'FACE_TOO_SMALL' in result['quality']['issues'] and result['embedding'] is None
