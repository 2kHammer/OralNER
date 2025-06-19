from .ner_model import NERModel

class ModelRegistry:
    _instance = None
    
    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        self._current_model = model
        
    @property
    def current_model(self):
        return self._current_model

    