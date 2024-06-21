import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
import yaml
from tensorflow.keras.applications import MobileNetV2

def load_data(data_path):
    data = []
    labels = []
    class_names = []
    for file in os.listdir(data_path):
        if file.endswith('.npy'):
            data.append(np.load(os.path.join(data_path, file)))
            label = file.split('.')[0]  # Assuming file names are of the form class.filename.npy
            if label not in class_names:
                class_names.append(label)
            labels.append(class_names.index(label))
    return np.array(data), np.array(labels), class_names

def build_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Reshape((3, 3, 1280)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    data_path = os.path.join(data_dir, params['train']['data_path'])
    model_path = os.path.join(data_dir, params['train']['model_path'])

    X, y, class_names = load_data(data_path)

    input_shape = (96, 96, 3)
    num_classes = len(class_names)

    model = build_model(input_shape, num_classes)

    mlflow.start_run()
    model.fit(X, y, epochs=10)
    mlflow.keras.log_model(model, "model")
    mlflow.end_run()

    model.save(model_path)

    # Save class names for inference
    with open(os.path.join(model_path, 'class_names.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
