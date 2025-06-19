#class TrainingResults:
    

class NERModel:
    def __init__(self, id, name, framework_name, base_model_name, storage_path):
        self.id = id
        self.name = name
        self.framework_name = framework_name
        self.base_model_name = base_model_name
        self.storage_path = storage_path
    
    