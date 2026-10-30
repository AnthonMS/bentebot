# Experimental Pascal-compatible CTranslate2 runtime, separate from the bot image.
# PyTorch runs VAD on CPU; faster-whisper independently selects CUDA through CTranslate2.
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Current cuDNN releases no longer support Pascal. 9.7's official support matrix does.
# CPU torch avoids relying on a current CUDA PyTorch wheel containing sm_61 kernels.
RUN pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
       nvidia-cublas-cu12==12.6.4.1 nvidia-cudnn-cu12==9.7.1.26 \
       ctranslate2==4.6.0 faster-whisper==1.2.1 \
       "whisperlivekit @ https://github.com/QuentinFuxa/WhisperLiveKit/archive/eb92c115ee7abc1e161b0b2159da31785b0f5566.zip"
EXPOSE 8000
COPY tune_backend.py /opt/bentebot-speech/tune_backend.py
RUN python /opt/bentebot-speech/tune_backend.py
# faster-whisper 1.2.1 uses an av.open argument removed in PyAV 19.
# Keep this late so existing expensive CUDA/Torch build layers can be reused.
RUN pip install --no-cache-dir av==16.1.0 \
    && python -c "import io,wave; from faster_whisper.audio import decode_audio; b=io.BytesIO(); w=wave.open(b,'wb'); w.setparams((1,2,16000,0,'NONE','not compressed')); w.writeframes(bytes(8000)); w.close(); b.seek(0); assert decode_audio(b).shape == (4000,)"
ENTRYPOINT ["wlk", "--host", "0.0.0.0"]
CMD ["--backend", "faster-whisper", "--backend-policy", "localagreement", "--model", "base", "--language", "auto", "--pcm-input"]
