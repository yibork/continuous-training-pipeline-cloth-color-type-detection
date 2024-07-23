from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments, Schema, MultiClassActualLabel  # Import necessary types
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os
import numpy as np
import pandas as pd

# Initialize the Arize client
arize_client = Client(space_key='0fc63b5', api_key='ed7be69db4262ce34e2')
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def get_image_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    embedding = model.predict(img_data)
    return embedding.flatten()

# Directory of your dataset
dataset_dir = 'uploads/'

# List to hold the data
data_list = []

# Loop through the dataset directory
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            embedding = get_image_embedding(file_path)
            data_list.append({'image_id': file, 'embedding': embedding, 'label': label})

# Convert to DataFrame
data_df = pd.DataFrame(data_list)

schema = Schema(
    prediction_id_column_name="image_id",
    actual_label_column_name="label",
    feature_column_names=["embedding"],
    actual_score_column_name="actual_score",
    prediction_score_column_name="prediction_score"
)

response = arize_client.log(
    model_id='cv_model',
    model_version='v1',
    model_type=ModelTypes.MULTI_CLASS,  # Use the ModelTypes enum for multi-class classification
    environment=Environments.TRAINING,  # Use the Environments enum
    dataframe=data_df,
    schema=schema
)

print("Training data logged successfully.")
