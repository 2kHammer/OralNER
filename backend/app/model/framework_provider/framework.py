from abc import ABC, abstractmethod

class Framework(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def apply_ner(self, text):
        pass

    @abstractmethod
    def prepare_training_data(self,rows):
        pass