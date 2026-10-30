FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# The native extension currently needs the unreleased 2.2 client API. Pin the
# reviewed development client; keep it completely outside the bot environment.
RUN pip install --no-cache-dir \
    'discord.py-self @ https://github.com/dolfies/discord.py-self/archive/2ba64a9a997e151a9c259984e0a179b1fdf4aff4.tar.gz' \
    davey==0.1.7 av==16.1.0 Pillow==12.0.0 \
    && pip install --no-cache-dir --no-deps discord-native-voice==0.1.1
WORKDIR /app
COPY bentebot_viewer /app/bentebot_viewer
RUN python -m bentebot_viewer.check
USER 99:100
ENTRYPOINT ["python", "-m", "bentebot_viewer"]
