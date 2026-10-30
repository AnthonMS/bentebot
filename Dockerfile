FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libopus0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --no-log-init --uid 99 --gid 100 --no-create-home --home-dir /app bentebot \
    && mkdir -p /app/data && chown 99:100 /app/data && chmod 2775 /app/data
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY bentebot/ ./bentebot/
COPY main.py ./
USER 99:100
VOLUME ["/app/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=3)"
CMD ["python", "-m", "bentebot"]
