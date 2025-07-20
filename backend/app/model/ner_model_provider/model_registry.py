from .ner_model import NERModel, TrainingResults
from app.utils.json_manager import JsonManager
from app.utils.helpers import get_current_datetime, random_string
from app.utils.config import MODEL_METADATA_PATH, MODIFIED_MODELS_PATH
from ..framework_provider.framework import FrameworkNames



# -------------------------------------
# class "DataRegistry"
# -------------------------------------
class ModelRegistry:
    """ Managing the models metadata"""
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

    # -------------------------------------
    # public functions
    # -------------------------------------
    @property
    def current_model(self):
        if len(self._models) > 0:
            return self._current_model
        else:
            return None
    
    def create_modified_model(self, new_model_name, base_model):
        """
        Creates outgoing from `base_model` a modified model

        Parameters
        new_model_name (str)
        base_model (NERModel)

        Returns
        (NERModel): the modified model
        """
        # change path to relative
        abs_path = MODIFIED_MODELS_PATH+"/"+new_model_name
        index_store = abs_path.find("app/")
        relative_modified_model_path = abs_path[index_store:]
        return NERModel(3, new_model_name, base_model.framework_name, base_model.name,relative_modified_model_path)

    def set_current_model(self,id):
        """
        Sets the model with `id` as active

        Parameters
        id (int)

        Returns
        (bool): true if successful, else false
        """
        index_model = self._get_index_model_id(id)
        if index_model is not None:
            self._current_model = self._models[index_model]
            return True
        else:
            return False

    def add_model(self,model:NERModel):
        """
        Adds `model` to the ModelRegistry

        Parameters
        model (NERModel)

        Returns
        (int): the model id
        """
        model.set_id(self._get_next_id())
        model.name = self._check_make_name_unique(model.name)
        self._models.append(model)
        if(len(self._models) == 1):
            self._current_model = model
        self._udpate_metadata()
        return model.id

    def add_training(self,id,dataset_name:str, dataset_id:int, metrics:TrainingResults, trainings_args: dict):
        """
        Adds a training to a model

        Parameters
        id (int)
        dataset_name (str):
        dataset_id (int):
        metrics (TrainingResults):
        trainings_args (dict):

        Returns
        (bool): if the adding was successful
        """
        index_model = self._get_index_model_id(id)
        if index_model is not None:
            model = self._models[index_model]
            model.set_state(2)
            model.append_training(get_current_datetime(),dataset_name,dataset_id,metrics,trainings_args)
            self._udpate_metadata()
            return True
        else:
            return False


    def list_models(self):
        """
        Returns all models of the model registry
        """
        return self._models

    def list_model(self, id):
        """
        Returns the model with `id`
        """
        index = self._get_index_model_id(id)
        if index is not None:
            return self._models[index]
        else:
            return None

    # -------------------------------------
    # private functions
    # -------------------------------------
    def _udpate_metadata(self):
        """
        Updates the model registry json file
        """
        self._json_manager.update_json([model.to_dict() for model in self._models])

    def _get_index_model_id(self, id):
        """
        Return the index for the model with `id` in self._models

        Parameters
        id (int)

        Returns
        (int)
        """
        ids = [model.id for model in self._models]
        try:
            return ids.index(id)
        except ValueError:
            return None

    def _get_next_id(self):
        """
        Return the next free ID in ._models

        Returns
        (int)
        """
        ids = [model.id for model in self._models]
        if len(ids) == 0:
            return 0
        max_id = max(ids)
        for i in range(0,max_id):
            if i not in ids:
                return i
        return max_id +1

    def _check_make_name_unique(self, new_name):
        """
        Checks if a name is unique and returns a unique name (name, data, random string) if not

        Parameters
        new_name (str): name to check

        Returns:
        (str): the name you should use
        """
        names = [model.name for model in self._models]
        if new_name in names:
            return new_name + "_"+ get_current_datetime()+"_" + random_string(10)
        else:
            return new_name

    @classmethod
    def _reset_instance(cls):
        """ Only for testing purposes """
        cls._instance = None

model_registry = ModelRegistry(MODEL_METADATA_PATH,1)