import json
import os

class JsonManager:
    """
    Loads and updates a list of serializable objects in an json file (at `_path_json`)
    """
    def __init__(self,path_json):
        self._path_json = path_json

    def load_json(self):
        """
        Loads the list of serializable objects from a json file

        Returns
        (List): of dicts
        """
        if os.path.exists(self._path_json):
            with open(self._path_json) as json_file:
                data_list = json.load(json_file)
                return data_list
        else:
            return None

    def update_json(self, data_list):
        """
        Updates serializable objects in the json file

        Parameters
        data_list (List): of dicts
        """
        with open(self._path_json, "w") as f:
            json.dump(data_list, f, indent=4)