FROM apache/airflow:slim-2.11.0-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git wget gnupg \
    && wget -qO - https://dvc.org/deb/iterative.asc | gpg --dearmor > /etc/apt/trusted.gpg.d/iterative.gpg \
    && wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list \
    && apt-get update && apt-get install -y dvc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Git safe directory for bind-mounted repo
RUN git config --global --add safe.directory /opt/airflow/repo

USER airflow


RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    apache-airflow-providers-amazon \
    apache-airflow-providers-postgres \
    mlflow==3.6.0 \
    psycopg2-binary \
    pyarrow \
    sentence-transformers \
    timm \
    Pillow \
    pydantic-settings \
    python-dotenv \
    boto3
