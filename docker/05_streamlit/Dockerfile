FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.52.0 \
    plotly==5.24.0 \
    psycopg2-binary==2.9.10 \
    boto3==1.35.0 \
    pandas==2.2.0 \
    pillow==11.0.0 \
    mlflow==3.6.0 \
    requests==2.32.3

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/multimodal_ai/streamlit/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
