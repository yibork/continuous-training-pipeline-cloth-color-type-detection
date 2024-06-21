import os
import cv2
import numpy as np
import yaml

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (96, 96))
    image = image.astype('float32') / 255.0
    np.save(output_path, image)

def preprocess_data(data_dir):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    raw_data_path = os.path.join(data_dir, params['preprocess']['raw_data_path'])
    processed_data_path = os.path.join(data_dir, params['preprocess']['processed_data_path'])
    
    os.makedirs(processed_data_path, exist_ok=True)

    for image_file in os.listdir(raw_data_path):
        image_path = os.path.join(raw_data_path, image_file)
        output_path = os.path.join(processed_data_path, f"{os.path.splitext(image_file)[0]}.npy")
        preprocess_image(image_path, output_path)
