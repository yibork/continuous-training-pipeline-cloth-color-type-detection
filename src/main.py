from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
from pathlib import Path
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import logging
from moviepy.editor import AudioFileClip
import mlflow
import random
from mlflow.tracking import MlflowClient
from zipfile import ZipFile

app = FastAPI()

# CORS middleware configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to store uploaded images and audio temporarily
UPLOAD_DIR = Path("uploads")
AUDIO_DIR = Path("audio")

# Set your MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Define the probabilities
PRODUCTION_PROB = 0
STAGING_PROB = 1

# Define model stages
PRODUCTION_STAGE = "Production"
STAGING_STAGE = "Staging"

def get_model(stage):
    client = MlflowClient()
    model_name = "mobile_net"
    
    # Get the latest version of the model in the specified stage
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise Exception(f"No models found in stage: {stage}")

    model_version = versions[0]
    model_uri = model_version.source
    model_path = f"models/{model_name}/{stage}"

    # Download the model
    os.makedirs(model_path, exist_ok=True)
    mlflow.artifacts.download_artifacts(model_uri, dst_path=model_path)

    return model_path

def load_model_from_stage(stage):
    model_path = get_model(stage)
    model_json_path = os.path.join(model_path, "model.json")
    model_bin_path = os.path.join(model_path, "group1-shard1of1.bin")

    # Check if the model files exist
    if not os.path.exists(model_json_path) or not os.path.exists(model_bin_path):
        raise FileNotFoundError("Model files not found. Ensure both model.json and group1-shard1of1.bin are present.")

    return model_json_path, model_bin_path

# Load the initial model
model_stage = PRODUCTION_STAGE if random.random() < PRODUCTION_PROB else STAGING_STAGE
model_json_path, model_bin_path = load_model_from_stage(model_stage)
logging.info(f"Model loaded from {model_stage} stage with paths: {model_json_path}, {model_bin_path}")

SAMPLE_RATE = 16000
N_MFCC = 13

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_audio(input_path: Path, output_path: Path):
    try:
        audio = AudioFileClip(str(input_path))
        audio.write_audiofile(str(output_path), codec='libmp3lame')
        audio.close()
    except Exception as e:
        logging.error(f"Error processing audio: {e}")

def preprocess_audio(file_path):
    try:
        logging.info(f"Starting audio preprocessing for file: {file_path}")
        
        # Check if file exists and format is recognized
        if not file_path.is_file():
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            audio, sr = sf.read(file_path, dtype='float32')
            logging.info(f"Audio loaded. Sample rate: {sr}, Audio shape: {audio.shape}")
        except RuntimeError as e:
            logging.error(f"Error opening '{file_path}': {e}")
            raise
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
            logging.info(f"Resampled audio to {SAMPLE_RATE} Hz")
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        logging.info(f"MFCC shape before flattening: {mfcc.shape}")
        
        # Flatten MFCC features
        mfcc = mfcc.flatten()
        logging.info(f"MFCC shape after flattening: {mfcc.shape}")
        
        # Define target length for padding/truncating
        target_length = 3960
        
        # Pad or truncate MFCC to target length
        if len(mfcc) > target_length:
            mfcc = mfcc[:target_length]
            logging.info("MFCC truncated to target length")
        else:
            mfcc = np.pad(mfcc, (0, target_length - len(mfcc)), 'constant')
            logging.info("MFCC padded to target length")
        
        logging.info(f"Final MFCC shape: {mfcc.shape}")
        
        # Add batch dimension
        mfcc = mfcc[np.newaxis, ...]
        logging.info(f"MFCC shape after adding batch dimension: {mfcc.shape}")
        
        return mfcc
    except Exception as e:
        logging.error(f"Error in preprocess_audio: {e}")
        raise
import model 

def predict_keyword(file_path):
    try:
        processed_audio = preprocess_audio(file_path)
        prediction = model.predict(processed_audio)
        not_correct_prob = prediction[0][1]
        
        if not_correct_prob > 0.5:
            predicted_class = 'not correct'
        else:
            predicted_class = 'correct'
        
        logging.info(f"Prediction: {prediction}, Predicted class: {predicted_class}")
        return predicted_class
    except Exception as e:
        logging.error(f"Error in predict_keyword: {e}")
        raise

@app.post("/upload_feedback/")
async def upload_feedback(image: UploadFile = File(...), audio: UploadFile = File(None)):
    try:
            # Ensure the uploads and audio directories exist
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate unique IDs for the files
        unique_id = uuid.uuid4().hex

        # Extract class name from the image filename
        image_filename = Path(image.filename).stem
        class_name = image_filename.split('.')[0]

        predicted_class = None
        audio_path = None

        if audio is not None:
            audio_path = AUDIO_DIR / f"{unique_id}.mp3"
            with audio_path.open("wb") as buffer:
                shutil.copyfileobj(audio.file, buffer)
            
            logging.info(f"Audio file saved at: {audio_path}")
            predicted_class = predict_keyword(audio_path)
            if predicted_class=="not correct":
                class_name = "Unknown"
        else:
            class_name = "Unknown"
        class_dir = UPLOAD_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image_extension = Path(image.filename).suffix
        new_image_filename = f"{class_name}.{unique_id}{image_extension}"
        image_path = class_dir / new_image_filename
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Add the uploaded image to DVC
        os.system(f"dvc add {image_path}")
        
        # Push the uploaded image to DVC
        os.system("dvc push")

        return JSONResponse(
            content={
                "message": "Feedback uploaded successfully.",
                "image_path": str(image_path),
                "audio_path": str(audio_path) if audio_path else "No audio uploaded.",
                "predicted_class": predicted_class if predicted_class else "No audio uploaded."
            },
            status_code=200
        )
    except Exception as e:
        logging.error(f"Error in upload_feedback: {e}")
        return JSONResponse(
            content={"message": f"Failed to upload feedback: {str(e)}"},
            status_code=500
        )

@app.get("/download-model")
async def download_model():
    try:
        # Choose model stage based on probabilities
        stage = PRODUCTION_STAGE if random.random() < PRODUCTION_PROB else STAGING_STAGE
        
        model_json_path, model_bin_path = load_model_from_stage(stage)
        model_zip_path = f"{model_json_path}.zip"

        # Create a zip file containing both model.json and group1-shard1of1.bin
        with ZipFile(model_zip_path, 'w') as zipf:
            zipf.write(model_json_path, arcname="model.json")
            zipf.write(model_bin_path, arcname="group1-shard1of1.bin")

        return FileResponse(model_zip_path, media_type='application/zip', filename="model.zip")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
