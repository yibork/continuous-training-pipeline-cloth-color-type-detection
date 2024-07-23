# Start from the latest Airflow image
FROM apache/airflow:latest

USER root
RUN groupadd -r airflow || true

# Update the system and install necessary software
RUN apt-get update \
    && apt-get install -y --no-install-recommends vim curl git rsync unzip \
    && apt-get autoremove -y \
    && apt-get clean

COPY gdrive-user-credentials.json /opt/airflow/gdrive-user-credentials.json
COPY uploads /opt/airflow/uploads
COPY uploads.dvc /opt/airflow/uploads.dvc
COPY test_dataset /opt/airflow/test_dataset

# Set permissions for the uploads directory and uploads.dvc
RUN chown -R airflow:airflow /opt/airflow/uploads \
    && chmod 755 /opt/airflow/uploads \
    && chown airflow:airflow /opt/airflow/uploads.dvc \
    && chmod 644 /opt/airflow/uploads.dvc \
    && chown -R airflow:airflow /opt/airflow/test_dataset \
    && chmod -R 755 /opt/airflow/test_dataset

# Switch to airflow user
USER airflow

# Update pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install specific versions of TensorFlow, TensorFlow.js, MLflow, DVC, and other required packages
RUN pip install tensorflow mlflow dvc-gdrive dvc numpy 

# Set environment variable for DVC to find GDrive credentials
ENV GDRIVE_USER_CREDENTIALS_DATA=/opt/airflow/gdrive-user-credentials.json
