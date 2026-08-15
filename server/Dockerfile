FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-whisper.txt .
# Install the open-source speech engine (heavy; comment out to skip).
RUN pip install --no-cache-dir -r requirements-whisper.txt || true

COPY correction.py main.py ./

ENV QURANI_JANNATI_WHISPER_MODEL=small
ENV QURANI_JANNATI_WHISPER_DEVICE=auto
ENV QURANI_JANNATI_WHISPER_COMPUTE=auto
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
