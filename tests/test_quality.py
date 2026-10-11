import numpy as np

from face_service import quality

# A level face: eyes, nose under their centre, mouth corners.
LEVEL = np.array([[40, 50], [80, 50], [60, 72], [45, 90], [75, 90]], dtype=np.float32)


def aligned(value=120, noise=40):
    rng = np.random.default_rng(0)
    img = np.clip(value + rng.normal(0, noise, (112, 112, 3)), 0, 255)
    return img.astype(np.uint8)


def test_a_level_face_has_no_turn():
    turn = quality.pose(LEVEL)
    assert abs(turn['yaw']) < 0.05 and abs(turn['pitch']) < 0.05


def test_a_face_turned_aside_is_not_frontal():
    turned = LEVEL.copy()
    turned[2, 0] = 78  # nose near the left eye
    result = quality.assess(np.array([0, 0, 150, 150]), turned, aligned())
    assert 'NOT_FRONTAL' in result['issues'] and not result['ok']


def test_small_dark_and_blurred_faces_are_named():
    flat = np.full((112, 112, 3), 30, dtype=np.uint8)
    result = quality.assess(np.array([0, 0, 40, 40]), LEVEL, flat)
    assert result['issues'] == ['FACE_TOO_SMALL', 'TOO_DARK', 'BLURRY']


def test_a_good_face_passes():
    result = quality.assess(np.array([0, 0, 150, 150]), LEVEL, aligned())
    assert result['ok'] and result['issues'] == []
