FROM edarace-agent-base:ubuntu24
WORKDIR /submission
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# Install only what the base image lacks; try a domestic mirror first, then PyPI.
RUN python3 -c "import numpy, scipy, cv2, onnxruntime, PIL" 2>/dev/null || \
    pip3 install --break-system-packages --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        numpy scipy opencv-python-headless onnxruntime pillow || \
    pip3 install --break-system-packages --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple \
        numpy scipy opencv-python-headless onnxruntime pillow || \
    pip3 install --break-system-packages --no-cache-dir numpy scipy opencv-python-headless onnxruntime pillow
RUN python3 -c "import numpy, scipy, cv2, onnxruntime, PIL; print('deps ok')"
COPY agent.py run.sh /submission/
COPY pcbparse /submission/pcbparse
COPY models /submission/models
COPY data /submission/data
RUN chmod +x /submission/run.sh && python3 -c "import sys; sys.path.insert(0,'/submission'); from pcbparse.ocr import get_rec; get_rec(); print('ocr ok')"
ENTRYPOINT ["/submission/run.sh"]
