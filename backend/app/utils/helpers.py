from datetime import datetime
import os, shutil

def get_current_datetime():
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

def delete_checkpoints_folder(path):
    if not os.path.exists(path):
        return False
    else:
        content = os.listdir(path)
        for f in content:
            full_path = os.path.join(path, f)
            if os.path.isdir(full_path) and f.startswith("checkpoint-"):
                print(f"Lösche: {full_path}")
                shutil.rmtree(full_path)
