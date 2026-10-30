FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libegl1 libgles2 libportaudio2 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-log-init --uid 99 --gid 100 --no-create-home awareness
COPY deploy/audio/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from mediapipe.tasks.python.core.mediapipe_c_bindings import load_raw_library; load_raw_library()"
COPY bentebot_audio_worker/ ./bentebot_audio_worker/
USER 99:100
EXPOSE 18769
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18769/health', timeout=3)"
CMD ["python", "-m", "bentebot_audio_worker"]
