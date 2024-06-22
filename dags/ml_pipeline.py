from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import confusion_matrix
from mlflow.tracking import MlflowClient
import os
import json

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'continual_image_model_training',
    default_args=default_args,
    description='A DAG for continually fine-tuning a machine learning model on image data and logging to MLflow',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Define the path to your model and data directory
data_dir = '/opt/airflow/uploads'
model_path = '/tmp/model.h5'

def ingest_data(**kwargs):
    return data_dir

def preprocess_data(**kwargs):
    ti = kwargs['ti']
    data_dir = ti.xcom_pull(task_ids='ingest_data')

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalizing the images and setting validation split
    train_data_flow = datagen.flow_from_directory(
        data_dir, 
        target_size=(128, 128), 
        batch_size=32, 
        class_mode='categorical', 
        subset='training'
    )
    val_data_flow = datagen.flow_from_directory(
        data_dir, 
        target_size=(128, 128), 
        batch_size=32, 
        class_mode='categorical', 
        subset='validation'
    )
    
    # Get the class indices for mapping labels
    class_indices = train_data_flow.class_indices
    
    # Save class indices to disk
    class_indices_path = '/tmp/class_indices.json'
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)

    return class_indices_path

def train_model(**kwargs):
    ti = kwargs['ti']
    class_indices_path = ti.xcom_pull(task_ids='preprocess_data')
    
    # Load class indices from disk
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data_flow = datagen.flow_from_directory(
        data_dir, 
        target_size=(128, 128), 
        batch_size=32, 
        class_mode='categorical', 
        subset='training'
    )
    val_data_flow = datagen.flow_from_directory(
        data_dir, 
        target_size=(128, 128), 
        batch_size=32, 
        class_mode='categorical', 
        subset='validation'
    )

    # Ensure there are images in both the training and validation sets
    if train_data_flow.samples == 0:
        raise ValueError("No training images found.")
    if val_data_flow.samples == 0:
        print("No validation images found. Skipping validation.")
        val_data_flow = None

    # Define the model architecture
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(len(class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    if val_data_flow is not None:
        model.fit(train_data_flow, epochs=10, validation_data=val_data_flow)
        # Evaluate the model
        loss, accuracy = model.evaluate(val_data_flow)

        # Predict on the validation data
        y_pred = model.predict(val_data_flow)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = val_data_flow.classes
        cm = confusion_matrix(y_true, y_pred_classes)
    else:
        model.fit(train_data_flow, epochs=10)
        # Evaluate the model
        loss, accuracy = model.evaluate(train_data_flow)

        # Predict on the training data (since there's no validation data)
        y_pred = model.predict(train_data_flow)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = train_data_flow.classes
        cm = confusion_matrix(y_true, y_pred_classes)

    # Save the model as .h5
    model.save(model_path, save_format='h5')
    if os.path.exists(model_path):
        print(f"Model saved successfully at {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found after saving.")

    # Save confusion matrix and metrics
    np.savetxt('/tmp/confusion_matrix.csv', cm, delimiter=',')
    np.save('/tmp/metrics.npy', np.array([accuracy, loss]))

    # Log model and metrics to MLflow
    mlflow.set_tracking_uri("http://172.22.208.1:5000")
    with mlflow.start_run() as run:
        mlflow.log_artifact(model_path, artifact_path="model.h5")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
        mlflow.log_artifact("/tmp/confusion_matrix.csv")
        run_id = run.info.run_id
        print(f"Model and metrics logged in run: {run_id}")

    model_name = "mobile_net"
    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)

    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model.h5",
        run_id=run_id
    )

    print(f"Model registered as version {model_version.version} of {model_name}")

def log_to_mlflow(**kwargs):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found when attempting to load.")

    # Load metrics and confusion matrix
    accuracy, loss = np.load('/tmp/metrics.npy')
    cm = np.loadtxt('/tmp/confusion_matrix.csv', delimiter=',')

    # Set the tracking URI to the specified server
    mlflow.set_tracking_uri("http://172.22.208.1:5000")

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the model
        mlflow.tensorflow.log_model(model, artifact_path="model")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        # Log confusion matrix as an artifact
        mlflow.log_artifact("/tmp/confusion_matrix.csv")

        run_id = run.info.run_id
        print(f"Model and metrics logged in run: {run_id}")

    model_name = "mobile_net"
    client = MlflowClient()

    # Check if the model is already registered
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)

    # Register the model
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )

    print(f"Model registered as version {model_version.version} of {model_name}")

# Define the tasks
ingest_data_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

log_to_mlflow_task = PythonOperator(
    task_id='log_to_mlflow',
    python_callable=log_to_mlflow,
    provide_context=True,
    dag=dag,
)

# Define the task dependencies
ingest_data_task >> preprocess_data_task >> train_model_task >> log_to_mlflow_task
