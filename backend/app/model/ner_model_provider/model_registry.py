from .ner_model import NERModel
from app.utils.json_manager import JsonManager


class ModelRegistry:
    _instance = None
    
    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model, path_metadata):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        self._current_model = model
        self._json_manager = JsonManager(path_metadata)
        data_as_list = self._json_manager.load_json()
        if data_as_list is not None:
            self._models = [NERModel.from_dict(d) for d in data_as_list]
        else:
            self._models = []

    @property
    def current_model(self):
        return self._current_model

    def add_model(self,model:NERModel):
        model.set_id(1)
        print(model.id)
        self._models.append(model)
        print(self._models)
        self._udpate_metadata()

    def _udpate_metadata(self):
        self._json_manager.update_json([model.to_dict() for model in self._models])

    