import os
import mlflow

def log_and_save(model_dir, promotion_decision, model_path, dvc_file_path, registered_model_name):
    mlflow.log_artifact(model_dir)
    mlflow.log_param('promotion_decision', promotion_decision)
    
    # Assuming that the promotion decision determines if we push the model to DVC
    if promotion_decision:
        print("Pushing model to DVC...")
    mlflow.register_model(model_dir, registered_model_name)
