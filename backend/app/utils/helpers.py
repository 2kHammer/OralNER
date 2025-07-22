import secrets
import string
from datetime import datetime
import os, shutil

def get_current_datetime():
    """
    Returns the current datetime in the format "%Y.%m.%d-%H.%M.%S"

    Returns
    (str)
    """
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

def delete_checkpoints_folder(path):
    """
    Deletes folders that start with "checkpoint-" in `path`

    Parameters
    path (str): path where checkpoint folders are deleted
    """
    if not os.path.exists(path):
        return False
    else:
        content = os.listdir(path)
        for f in content:
            full_path = os.path.join(path, f)
            if os.path.isdir(full_path) and f.startswith("checkpoint-"):
                print(f"Lösche: {full_path}")
                shutil.rmtree(full_path)

def random_string(length):
    """
    Returns a random string of length `length`

    Parameters
    length (int): length of string

    Returns
    (str)
    """
    letters = string.ascii_letters
    return ''.join(secrets.choice(letters) for _ in range(length))
