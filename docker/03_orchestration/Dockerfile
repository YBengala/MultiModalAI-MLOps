FROM apache/airflow:slim-2.11.0-python3.12

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

USER root
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
USER airflow
RUN uv pip install --no-cache-dir \
    apache-airflow-providers-amazon \
    apache-airflow-providers-postgres \
    apache-airflow-providers-fab \
    mlflow==3.6.0 \
    psycopg2-binary
