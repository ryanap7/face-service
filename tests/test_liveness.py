import numpy as np

from face_service import liveness


def test_the_crop_grows_the_face_box_and_stays_in_the_image():
    # Same arithmetic as Silent-Face-Anti-Spoofing's CropImage._get_new_box.
    assert liveness.crop_box(640, 480, (300, 200, 100, 100), 2.7) == (215, 115, 485, 385)
    left, top, right, bottom = liveness.crop_box(640, 480, (0, 0, 100, 100), 4.0)
    assert left == 0 and top == 0 and right <= 639 and bottom <= 479


class Session:
    def __init__(self, logits):
        self.logits = np.array([logits], dtype=np.float32)

    def run(self, _outputs, _inputs):
        return [self.logits]


def test_two_models_agreeing_on_a_live_face_say_real():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    sessions = [(Session([0, 9, 0]), 2.7), (Session([0, 8, 1]), 4.0)]
    result = liveness.judge(image, (300, 200, 100, 100), sessions)
    assert result['verdict'] == 'REAL' and result['real'] > 0.9


def test_disagreement_is_unsure_and_a_photo_is_spoof():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (300, 200, 100, 100)
    unsure = [(Session([0, 9, 0]), 2.7), (Session([9, 0, 0]), 4.0)]
    assert liveness.judge(image, box, unsure)['verdict'] == 'UNSURE'
    spoof = [(Session([9, 0, 0]), 2.7), (Session([0, 0, 9]), 4.0)]
    assert liveness.judge(image, box, spoof)['verdict'] in ('SPOOF', 'UNSURE')
    photo = [(Session([9, 0, 0]), 2.7), (Session([9, 0, 0]), 4.0)]
    assert liveness.judge(image, box, photo)['verdict'] == 'SPOOF'
