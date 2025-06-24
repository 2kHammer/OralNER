import json
import os

class JsonManager:
    def __init__(self,path_json):
        self._path_json = path_json

    def load_json(self):
        if os.path.exists(self._path_json):
            with open(self._path_json) as json_file:
                data_list = json.load(json_file)
                return data_list
        else:
            return None

    def update_json(self, data_list):
        with open(self._path_json, "w") as f:
            json.dump(data_list, f, indent=4)