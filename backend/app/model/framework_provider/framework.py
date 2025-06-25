from abc import ABC, abstractmethod
from enum import Enum

class FrameworkNames(Enum):
    HUGGINGFACE = 1
    FLAIR = 2
    SPACY = 3

class Framework(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def apply_ner(self, text):
        pass

    @abstractmethod
    def prepare_training_data(self,rows, train_size=0.7, validation_size=0.1, test_size=0.2):
        pass
    
    @abstractmethod
    def finetune_ner_model(self,base_model_path,data_dict, label_id,name,new_model_path):
        pass