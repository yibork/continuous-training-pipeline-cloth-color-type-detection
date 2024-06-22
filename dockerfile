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

# Set permissions for the imagery directory and imagery.dvc
RUN chown -R airflow:airflow /opt/airflow/uploads \
    && chmod 755 /opt/airflow/uploads \
    && chown airflow:airflow /opt/airflow/uploads.dvc \
    && chmod 644 /opt/airflow/uploads.dvc

# Install PyTorch, torchvision, and torchaudio for CPU usage
USER airflow

# Install specific version of TensorFlow along with MLflow, DVC, and other required packages
RUN pip install --upgrade pip && \
    pip install mlflow dvc-gdrive dvc tensorflow==2.12 scikit-learn numpy

# Set environment variable for DVC to find GDrive credentials
ENV GDRIVE_USER_CREDENTIALS_DATA=/opt/airflow/gdrive-user-credentials.json
