# Lightweight runtime image (CPU is sufficient; ~2-6 s per schematic).
# docker build -t pcbparse .
# docker run --rm -v /path/to/images:/data/input -v /path/to/out:/data/output pcbparse
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pcbparse ./pcbparse
COPY models ./models
COPY data ./data
COPY run.py .
ENTRYPOINT ["python", "run.py"]
CMD ["/data/input", "/data/output"]
