from .ner_model import NERModel, TrainingResults
from app.utils.json_manager import JsonManager
from app.utils.helpers import get_current_datetime
from app.utils.config import MODEL_METADATA_PATH, MODIFIED_MODELS_PATH
from ..framework_provider.framework import FrameworkNames



class ModelRegistry:
    _instance = None
    
    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, path_metadata,current_model_id=1):
        self._json_manager = JsonManager(path_metadata)
        data_as_list = self._json_manager.load_json()
        if data_as_list is not None:
            self._models = [NERModel.from_dict(d) for d in data_as_list]
            index_current_model = self._get_index_model_id(current_model_id)
            if index_current_model is not None:
                self._current_model= self._models[index_current_model]
            else:
                # set default model
                self._current_model = self._models[0]
        else:
            self._models = []


    @property
    def current_model(self):
        return self._current_model
    
    def create_modified_model(self, new_model_name, base_model):
        return NERModel(3, new_model_name, base_model.framework_name, base_model.name,MODIFIED_MODELS_PATH+"/"+new_model_name)

    def set_current_model(self,id):
        index_model = self._get_index_model_id(id)
        if index_model is not None:
            self._current_model = self._models[index_model]
            return True
        else:
            return False

    def add_model(self,model:NERModel):
        model.set_id(self._get_next_id())
        model.name = self._check_make_name_unique(model.name,get_current_datetime())
        self._models.append(model)
        self._udpate_metadata()
        return model.id

    def add_training(self,id,dataset_name:str, dataset_id:int, metrics:TrainingResults, trainings_args: dict):
        index_model = self._get_index_model_id(id)
        # erhält None von inde_model
        model = self._models[index_model]
        model.set_state(2)
        model.append_training(get_current_datetime(),dataset_name,dataset_id,metrics,trainings_args)
        self._udpate_metadata()


    def list_models(self):
        print(len(self._models))
        return self._models

    def list_model(self, id):
        index = self._get_index_model_id(id)
        if index is not None:
            return self._models[index]
        else:
            return None

    def _udpate_metadata(self):
        self._json_manager.update_json([model.to_dict() for model in self._models])

    def _get_index_model_id(self, id):
        ids = [model.id for model in self._models]
        try:
            return ids.index(id)
        except ValueError:
            return None

    def _get_next_id(self):
        ids = [model.id for model in self._models]
        if len(ids) == 0:
            return 0
        max_id = max(ids)
        for i in range(0,max_id):
            if i not in ids:
                return i
        return max_id +1

    def _check_make_name_unique(self, new_name,date):
        names = [model.name for model in self._models]
        if new_name in names:
            return new_name + str(date)
        else:
            return new_name

model_registry = ModelRegistry(MODEL_METADATA_PATH,1)