# Contoh: Gunakan base image Python yang sesuai
# Sesuaikan 'python:3.9-slim-buster' dengan versi Python dan OS dasar yang Anda gunakan
# Misalnya: FROM python:3.10-slim-bullseye atau FROM python:3.8-slim-buster
FROM python:3.11-slim-bullseye

# Atur direktori kerja di dalam container
WORKDIR /app

# Atur variabel lingkungan PATH untuk venv
ENV NIXPACKS_PATH=/opt/venv/bin:$NIXPACKS_PATH

# Langkah baru: Instal build dependencies dan system libraries
# Ini harus dilakukan SEBELUM pip install
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

# Salin semua file dari konteks build ke dalam direktori /app di container
COPY . /app/.

# Buat virtual environment, aktifkan, dan instal semua dependensi Python
# Baris ini tetap sama seperti sebelumnya
RUN --mount=type=cache,id=s/a726cafc-25c1-447f-b656-9d9f7f7ff7e9-/root/cache/pip,target=/root/.cache/pip python -m venv --copies /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install -r requirements.txt

# Jika Anda memiliki CMD atau ENTRYPOINT di Dockerfile Anda, letakkan di sini
# CMD ["python", "app.py"]
