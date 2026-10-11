"""
Downloads the model files kept out of git (too large), each checked
against its known SHA-256. Safe to run again.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / 'models'
FILES = {
    'face_recognition_sface_2021dec.onnx': (
        'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
        '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for name, (url, expected) in FILES.items():
        target = MODELS / name
        if target.exists() and sha256(target) == expected:
            print(f'ok {name}')
            continue
        print(f'downloading {name}')
        partial = target.with_suffix('.part')
        urllib.request.urlretrieve(url, partial)
        if sha256(partial) != expected:
            partial.unlink()
            print(f'checksum mismatch for {name}', file=sys.stderr)
            return 1
        partial.rename(target)
        print(f'ok {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
