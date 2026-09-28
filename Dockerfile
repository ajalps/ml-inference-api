# Credit Default Risk Inference API — container image
FROM python:3.11-slim

WORKDIR /app

# Install pinned dependencies first so this layer is cached across rebuilds
# that only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained model artifact.
COPY app/ ./app/
COPY model/ ./model/

EXPOSE 8000

# Basic container health check hitting the API's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
