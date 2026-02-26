FROM python:3.12-slim

RUN useradd --create-home appuser

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi==0.129.0 \
    uvicorn[standard]==0.41.0 \
    python-multipart==0.0.22 \
    mlflow==3.6.0 \
    sentence-transformers==5.1.2 \
    timm==1.0.22 \
    Pillow==12.0.0 \
    pydantic-settings==2.12.0 \
    python-dotenv==1.2.1 \
    boto3>=1.37.2

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "multimodal_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
