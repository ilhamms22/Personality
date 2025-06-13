# --- Tahap 1: Build Dependensi ---
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

# Instal dependensi sistem yang dibutuhkan untuk build (seperti build-essential)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpq-dev \
    libsndfile1 \
    libsoxr-dev \
    libhdf5-dev \
    libopencv-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements.txt dan instal dependensi Python ke venv
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# --- Tahap 2: Runtime (Image Final) ---
FROM python:3.11-slim-bullseye

WORKDIR /app

# Salin venv yang sudah jadi dari tahap builder
COPY --from=builder /opt/venv /opt/venv

# Salin kode aplikasi Anda
COPY . /app/.

# Atur PATH untuk venv
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH
ENV PATH="/opt/venv/bin:$PATH" # Pastikan venv ada di PATH

# Hapus cache pip di tahap akhir jika masih ada
RUN rm -rf /root/.cache/pip

# Jika Anda punya CMD atau ENTRYPOINT, letakkan di sini
# Contoh: CMD ["python", "app.py"]
