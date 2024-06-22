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

    # Add the uploaded file to DVC
    os.system(f"dvc add {file_path}")

    # Push the uploaded file to DVC
    os.system("dvc push")

    # Commit and push the uploads.dvc file to GitHub
    # os.system("git add uploads.dvc")
    # os.system("git commit -m 'Add uploads.dvc'")
    # os.system("git push")

    return {"message": "Image uploaded and pushed to DVC successfully."}
