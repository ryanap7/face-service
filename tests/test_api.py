from queue import Empty

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from face_service import api, config


class FakeEngine:
    def analyze(self, image, with_liveness=True):
        return {'faces': 1, 'shape': list(image.shape), 'liveness': with_liveness}


class FakePool:
    def __init__(self, busy=False):
        self.busy = busy

    def run(self, work):
        if self.busy:
            raise Empty
        return work(FakeEngine())


def jpeg(w=320, h=240):
    ok, data = cv2.imencode('.jpg', np.full((h, w, 3), 128, dtype=np.uint8))
    return data.tobytes()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, 'API_KEY', 'secret')
    with TestClient(api.create_app(lambda: FakePool())) as c:
        yield c


def test_health_names_the_model(client):
    assert client.get('/health').json()['model'] == config.MODEL_VERSION


def test_an_image_is_analysed_with_the_api_key(client):
    res = client.post('/v1/analyze?liveness=false', content=jpeg(), headers={'X-Api-Key': 'secret'})
    assert res.status_code == 200
    assert res.json() == {'faces': 1, 'shape': [240, 320, 3], 'liveness': False}


def test_large_frames_are_scaled_down(client):
    res = client.post('/v1/analyze', content=jpeg(2560, 1440), headers={'X-Api-Key': 'secret'})
    assert res.json()['shape'] == [720, 1280, 3]


@pytest.mark.parametrize('body, headers, status', [
    (b'x', {}, 401),
    (b'', {'X-Api-Key': 'secret'}, 400),
    (b'not an image', {'X-Api-Key': 'secret'}, 400),
    (b'0' * (config.MAX_IMAGE_BYTES + 1), {'X-Api-Key': 'secret'}, 413),
])
def test_bad_requests_are_refused(client, body, headers, status):
    assert client.post('/v1/analyze', content=body, headers=headers).status_code == status


def test_a_busy_service_says_so(monkeypatch):
    monkeypatch.setattr(config, 'API_KEY', '')
    with TestClient(api.create_app(lambda: FakePool(busy=True))) as c:
        assert c.post('/v1/analyze', content=jpeg()).status_code == 503
