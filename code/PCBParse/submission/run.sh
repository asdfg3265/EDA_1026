#!/bin/bash
# Entry point required by the platform: /submission/run.sh -input <dir> -output <dir>
INPUT_DIR=""; OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -input|--input) INPUT_DIR="$2"; shift 2 ;;
    -output|--output) OUTPUT_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done
INPUT_DIR="${INPUT_DIR:-/input}"; OUTPUT_DIR="${OUTPUT_DIR:-/output}"
echo "[PCBParse] input=$INPUT_DIR output=$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/final_results"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 -c "import numpy, scipy, cv2, onnxruntime; print('[PCBParse] deps ok numpy', numpy.__version__, 'cv2', cv2.__version__, 'ort', onnxruntime.__version__)"
python3 /submission/agent.py -input "$INPUT_DIR" -output "$OUTPUT_DIR"
EXIT_CODE=$?
echo "[PCBParse] exit code $EXIT_CODE"
exit $EXIT_CODE
