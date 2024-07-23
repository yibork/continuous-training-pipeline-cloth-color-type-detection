import os
import pandas as pd
from PIL import Image
import numpy as np
import whylogs as why
import io

# Set WhyLabs environment variables
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-UUM7vy"
os.environ["WHYLABS_API_KEY"] = "eWwChjatfw.lC3S17S94ilprEs22TY5jJAT6H4dN2mfAKpfZn1hAHEx5r4EvPZ06:org-UUM7vy"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-2"

dataset_dir = "./uploads"
data = []

# Function to extract image features (e.g., mean color)
def extract_image_features(image):
    # Convert image to numpy array
    image_array = np.array(image)
    # Calculate mean color
    mean_color = image_array.mean(axis=(0, 1))
    return mean_color

# Iterate over the files in the dataset directory
for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file.endswith((".png", ".jpg", ".jpeg")):  # Process image files
            print(f"Processing file: {file_path}")
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mean_color = extract_image_features(img)
                    data.append({
                        "file_path": file_path,
                        "width": width,
                        "height": height,
                        "mean_color_r": mean_color[0],
                        "mean_color_g": mean_color[1],
                        "mean_color_b": mean_color[2],
                        "label": os.path.basename(subdir)
                    })
            except Exception as e:
                print(f"Failed to process image {file_path}: {e}")
df = pd.DataFrame(data)
results = why.log(df)
# Upload results to WhyLabs
results.writer("whylabs").write()
