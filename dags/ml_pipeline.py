from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import layers, models
import numpy as np
import os
import json
import mlflow
import mlflow.tensorflow
from collections import defaultdict
import matplotlib.pyplot as plt

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

data_dir = '/opt/airflow/uploads'
model_path = '/tmp/model.h5'

def extract_label_from_filename(filename):
    return filename.split('.')[0]

def load_data(data_dir, max_images_per_class=1):
    images = []
    labels = []
    class_count = defaultdict(int)

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                label = extract_label_from_filename(file)
                if class_count[label] < max_images_per_class:
                    image_path = os.path.join(root, file)
                    img = keras_image.load_img(image_path, target_size=(224, 224))
                    img = keras_image.img_to_array(img)
                    images.append(img)
                    labels.append(label)
                    class_count[label] += 1
                if len(class_count) >= 14 and all(count >= max_images_per_class for count in class_count.values()):
                    break
        if len(class_count) >= 14 and all(count >= max_images_per_class for count in class_count.values()):
            break

    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Number of images per class: {dict(class_count)}")
    print(f"Total images loaded: {len(images)}")
    return images, labels


def create_simple_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_class_labels(label_to_index, filename):
    with open(filename, 'w') as f:
        for label, index in label_to_index.items():
            f.write(f"{index}: {label}\n")
    print(f"Class labels saved to {filename}")

def ingest_data(**kwargs):
    return data_dir

def preprocess_data(**kwargs):
    ti = kwargs['ti']
    data_dir = ti.xcom_pull(task_ids='ingest_data')

    train_dir = os.path.join(data_dir, "training")
    test_dir = os.path.join(data_dir, "testing")

    train_images, train_labels = load_data(train_dir, max_images_per_class=1)
    test_images, test_labels = load_data(test_dir, max_images_per_class=1)

    label_to_index = {label: index for index, label in enumerate(np.unique(train_labels))}
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_labels = np.array([label_to_index[label] for label in train_labels])
    test_labels = np.array([label_to_index[label] for label in test_labels])

    class_labels_path = '/tmp/class_labels.txt'
    save_class_labels(label_to_index, class_labels_path)

    return {
        'train_images': train_images.tolist(),
        'train_labels': train_labels.tolist(),
        'test_images': test_images.tolist(),
        'test_labels': test_labels.tolist(),
        'class_labels_path': class_labels_path,
        'input_shape': (224, 224, 3),
        'num_classes': len(label_to_index),
        'index_to_label': index_to_label
    }

def train_model(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='preprocess_data')

    train_images = np.array(data['train_images'])
    train_labels = np.array(data['train_labels'])
    test_images = np.array(data['test_images'])
    test_labels = np.array(data['test_labels'])
    input_shape = data['input_shape']
    num_classes = data['num_classes']
    index_to_label = data['index_to_label']
    class_labels_path = data['class_labels_path']

    train_images = train_images.reshape(-1, 224, 224, 3)
    test_images = test_images.reshape(-1, 224, 224, 3)

    model = create_simple_cnn_model(input_shape, num_classes)
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), steps_per_epoch=10)

    model.save(model_path)

    train_loss, train_accuracy = model.evaluate(train_images, train_labels)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    metrics_dir = '/tmp/metrics'
    os.makedirs(metrics_dir, exist_ok=True)

    with open(os.path.join(metrics_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(os.path.join(metrics_dir, 'accuracy_plot.png'))
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(metrics_dir, 'loss_plot.png'))
    plt.show()

    mlflow.set_tracking_uri("http://192.168.11.101:5000/")
    with mlflow.start_run() as run:
        mlflow.log_artifact(model_path, artifact_path="model.h5")
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_artifact(class_labels_path)
        mlflow.log_artifact(os.path.join(metrics_dir, 'history.json'))
        mlflow.log_artifact(os.path.join(metrics_dir, 'accuracy_plot.png'))
        mlflow.log_artifact(os.path.join(metrics_dir, 'loss_plot.png'))
        run_id = run.info.run_id
        print(f"Model and metrics logged in run: {run_id}")

    model_name = "simple_cnn"
    client = mlflow.MlflowClient()

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

def evaluate_model(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='preprocess_data')

    test_images = np.array(data['test_images'])
    test_labels = np.array(data['test_labels'])
    index_to_label = data['index_to_label']

    model = tf.keras.models.load_model(model_path)

    test_images = test_images.reshape(-1, 224, 224, 3)

    loss, accuracy = model.evaluate(test_images, test_labels)

    np.save('/tmp/metrics.npy', np.array([accuracy, loss]))

    mlflow.set_tracking_uri("http://192.168.11.101:5000/")
    with mlflow.start_run() as run:
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_loss", loss)
        run_id = run.info.run_id
        print(f"Model evaluation metrics logged in run: {run_id}")

    return accuracy

def log_to_mlflow(**kwargs):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found when attempting to load.")

    accuracy, loss = np.load('/tmp/metrics.npy')

    mlflow.set_tracking_uri("http://192.168.11.101:5000/")

    with mlflow.start_run() as run:
        mlflow.log_artifact(model_path, artifact_path="model.h5")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        run_id = run.info.run_id
        print(f"Model and metrics logged in run: {run_id}")

    model_name = "mobile_net"
    client = mlflow.MlflowClient()

    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)

    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model.h5",
        run_id=run_id,
    )
    ti = kwargs['ti']

    test_accuracy = ti.xcom_pull(task_ids='evaluate_model')
    if test_accuracy > 0.85:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
        )
        print(f"Model version {model_version.version} of {model_name} transitioned to Production")
    else:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )
        print(f"Model version {model_version.version} of {model_name} transitioned to Staging")
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
evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)
log_to_mlflow_task = PythonOperator(
    task_id='log_to_mlflow',
    python_callable=log_to_mlflow,
    provide_context=True,
    dag=dag,
)
ingest_data_task >> preprocess_data_task >> train_model_task >> evaluate_task >> log_to_mlflow_task
