from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import shutil
import os

app = FastAPI()

# Path to store uploaded images temporarily
UPLOAD_DIR = "uploads"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Ensure the uploads directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save the uploaded file to the uploads directory
    file_path = Path(UPLOAD_DIR) / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Image uploaded successfully."}
