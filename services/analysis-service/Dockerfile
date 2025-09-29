# Note: This image installs heavy scientific deps. First build may take a while.
FROM python:3.11-slim

# System deps for scientific stack and healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
        gfortran \
        libopenblas-dev \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Install Python deps (includes numpy/pandas/scikit-learn/tensorflow/etc.)
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8003

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]

