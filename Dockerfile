# Stage: runtime image (single stage, no build tools needed)
FROM python:3.11-slim

# Show logs immediately; disable __pycache__ writes would be noise in CI.
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy only requirements first so dependency layer is cached independently
# of application code changes (layer caching optimization).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code AFTER dependencies: code changes reuse the
# cached dependency layer; model is copied explicitly (not COPY . .).
COPY src/ ./src/
COPY modelo.keras ./modelo.keras

# Run as an unprivileged user (defense in depth: no root in container).
RUN useradd --create-home appuser
USER appuser

EXPOSE 10000

# Port is configurable via PORT env var (defaults to 10000).
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.getenv('PORT','10000')+'/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
