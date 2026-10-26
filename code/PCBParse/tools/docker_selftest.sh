#!/bin/bash
# Run on the 4070Ti machine (WSL Ubuntu with Docker) BEFORE uploading submission.zip.
#   bash docker_selftest.sh /path/to/edarace-agent-base_ubuntu24.tar /path/to/submission.zip /path/to/open_GT
set -e
BASE_TAR=$1; ZIP=$2; GT=$3
W=$(mktemp -d); echo "work dir $W"
docker image inspect edarace-agent-base:ubuntu24 >/dev/null 2>&1 || docker load -i "$BASE_TAR"
echo "== base image environment"
docker run --rm --entrypoint bash edarace-agent-base:ubuntu24 -c \
  "python3 --version; nproc; free -g | head -2; python3 -c 'import importlib, importlib.util;[print(m, getattr(importlib.import_module(m),\"__version__\",\"ok\")) if importlib.util.find_spec(m) else print(m,\"MISSING\") for m in [\"numpy\",\"scipy\",\"cv2\",\"onnxruntime\",\"PIL\",\"torch\",\"ultralytics\"]]'" || true
mkdir -p $W/sub $W/input $W/output && cd $W/sub && unzip -q "$ZIP"
for d in "$GT"/*/; do c=$(basename "$d"); cp "$d/$c.png" $W/input/; done
docker build -t pcbparse:test . 2>&1 | tail -5
time docker run --rm -v $W/input:/input:ro -v $W/output:/output:rw pcbparse:test -input /input -output /output
ls $W/output/final_results | head
echo "outputs in $W/output/final_results"
