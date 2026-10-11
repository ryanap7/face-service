#!/bin/sh
# Public sample photos from the OpenCV repository, for tests/test_models.py.
set -e
cd "$(dirname "$0")/../tests" && mkdir -p .cache && cd .cache
for name in lena.jpg messi5.jpg; do
  [ -f "$name" ] || curl -fsSL -o "$name" "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/$name"
done
