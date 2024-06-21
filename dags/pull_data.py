import os

def pull_data_from_dvc():
    """Pull data from DVC storage."""
    os.system('dvc pull')
