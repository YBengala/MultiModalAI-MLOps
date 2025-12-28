FROM python:3.13.7-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system mlflow==3.6.0 psycopg2-binary boto3

EXPOSE 5000
CMD ["mlflow", "server"]
