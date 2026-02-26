FROM apache/airflow:slim-2.11.0-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git pipx \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install "dvc[s3]"


# Git safe directory for bind-mounted repo
RUN git config --global --add safe.directory /opt/airflow/repo

USER airflow


RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    apache-airflow-providers-amazon==9.9.0 \
    apache-airflow-providers-postgres==6.5.3 \
    mlflow==3.6.0 \
    psycopg2-binary==2.9.10 \
    pyarrow==22.0.0 \
    sentence-transformers==5.1.2 \
    timm==1.0.22 \
    Pillow==12.0.0 \
    pydantic-settings==2.12.0 \
    python-dotenv==1.2.1 \
    boto3>=1.37.2 \
    optuna==4.6.0 \
    scikit-learn==1.7.2 \
    joblib==1.5.2
