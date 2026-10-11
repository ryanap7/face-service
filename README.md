# SIMTELPAS Face Service

Face detection, quality checks, liveness (anti-spoofing) and face embeddings
for the cooperative cashier: an inmate stands in front of the cashier's camera,
is recognised among the inmates of that prison, then confirms with their PIN.

The service is stateless. It turns one image into facts about the largest face
in it; enrolment, matching, thresholds and audit live in the backend. Images
are never stored.

## How the cashier flow uses it

1. The cashier page finds faces in the browser (MediaPipe) at camera speed, so
   the box and hints ("dekatkan wajah", "kurang terang") need no server.
2. When a face is steady and sharp, the page sends that frame (1–2 a second) to
   the backend, which forwards it here.
3. This service returns quality, liveness and the embedding. The backend
   compares the embedding with the enrolled inmates of the cashier's prison
   (cosine similarity) and shows the best match; the inmate confirms with PIN.

Enrolment uses the same endpoint on the three frames taken in the face photo
session; the backend stores their embeddings with `model`.

## API

Internal only: the backend calls it on the private network; the port is closed
to everything else on the host.

### `POST /v1/analyze?liveness=true`

Body: the raw image (`image/jpeg`, PNG or WebP), at most 5 MB. Larger frames
are scaled to 1280 px on the long side. Header `X-Api-Key` when `FACE_API_KEY`
is set.

```json
{
  "model": "yunet-2023mar+sface-2021dec",
  "faces": 1,
  "face": { "box": [207.8, 182.5, 145.9, 206.9], "score": 0.909, "landmarks": [[271.6, 269.4], "…"] },
  "quality": {
    "ok": true, "issues": [],
    "faceSize": 145, "brightness": 116.7, "sharpness": 1247.1, "yaw": 0.168, "pitch": 0.003
  },
  "liveness": { "verdict": "REAL", "real": 0.9966 },
  "embedding": [0.0213, "… 128 numbers, unit length"]
}
```

- `quality.issues`: `NO_FACE`, `MULTIPLE_FACES`, `FACE_TOO_SMALL`, `TOO_DARK`,
  `TOO_BRIGHT`, `BLURRY`, `NOT_FRONTAL`. `embedding` is only given when
  `quality.ok`.
- `liveness.verdict`: `REAL`, `SPOOF` (a printed photo or a screen) or `UNSURE`.
- Errors: 400 empty or unreadable image, 401 wrong key, 413 too large, 503 all
  engines busy (retry).

### `GET /health`

`{"status": "ok", "model": "…", "pool": 4}`

## Matching (in the backend)

Embeddings are compared by dot product (they are unit length). On OpenCV's
benchmark the same person scores above about 0.36 with SFace; our own
threshold is set from shadow-mode data at the cashier before it decides
anything. Sample photos: the same face 0.89–0.91, two different people 0.13.
Embeddings from different `model` values are not comparable: when the model
changes, enrolments are recomputed from the stored session frames.

## Models and licences

| File | Purpose | Source | Licence |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | detection and 5 landmarks | OpenCV Zoo | MIT |
| `face_recognition_sface_2021dec.onnx` | 128-d embedding (downloaded at build, SHA-256 checked) | OpenCV Zoo | Apache-2.0 |
| `2.7_80x80_MiniFASNetV2.onnx`, `4_0_0_80x80_MiniFASNetV1SE.onnx` | liveness | Silent-Face-Anti-Spoofing (minivision-ai), exported to ONNX from the published weights; outputs match PyTorch to 1.5e-6 | Apache-2.0 (`models/LICENSE-MiniFASNet-Apache-2.0`) |

All are licensed for commercial use. InsightFace/ArcFace weights are not used:
they are for non-commercial research unless licensed.

## Running

```sh
cp .env.example .env          # set FACE_API_KEY
docker compose up -d --build
curl localhost:8100/health
```

About 15 ms per frame on one CPU core; `FACE_POOL_SIZE` engines work in
parallel. No GPU needed, so the GPU stays with transcription.

## Development

```sh
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/fetch_models.py
scripts/fetch_test_images.sh   # public OpenCV sample photos for the model tests
.venv/bin/python -m pytest
```
