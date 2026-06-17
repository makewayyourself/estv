# Backend container for the Gemini Live Translator.
# Works on Render, Railway, Fly.io, Cloud Run, etc. (anything that supports
# WebSockets and injects a $PORT).
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source (backend + static frontend + PDF fonts).
COPY main.py .
COPY services ./services
COPY static ./static
COPY assets ./assets

# Most platforms set $PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
