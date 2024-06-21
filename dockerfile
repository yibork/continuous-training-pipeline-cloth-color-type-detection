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
COPY imagery /opt/airflow/imagery
COPY test-dataset /opt/airflow/test-dataset
COPY models /opt/airflow/models
COPY imagery.dvc /opt/airflow/imagery.dvc

# Set permissions for the imagery directory and imagery.dvc
RUN chown -R airflow:airflow /opt/airflow/imagery \
    && chmod 755 /opt/airflow/imagery \
    && chown -R airflow:airflow /opt/airflow/models \
    && chmod 755 /opt/airflow/models \
    && chown airflow:airflow /opt/airflow/imagery.dvc \
    && chmod 644 /opt/airflow/imagery.dvc

# Install PyTorch, torchvision, and torchaudio for CPU usage
USER airflow


# Install MLflow and DVC with Google Drive support
RUN pip install mlflow dvc-gdrive dvc

# Set environment variable for DVC to find GDrive credentials
ENV GDRIVE_USER_CREDENTIALS_DATA=/opt/airflow/gdrive-user-credentials.json
