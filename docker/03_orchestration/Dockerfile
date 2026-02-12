FROM apache/airflow:slim-2.11.0-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-amazon \
    apache-airflow-providers-postgres \
    mlflow==3.6.0 \
    psycopg2-binary
