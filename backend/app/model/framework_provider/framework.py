from abc import ABC, abstractmethod
from enum import Enum

class FrameworkNames(Enum):
    HUGGINGFACE = 1
    FLAIR = 2
    SPACY = 3

entity_types = []

class Framework(ABC):
    @abstractmethod
    def load_model(self,model):
        """
        Load the `model` in the class. Necessary for the other functions.

        Parameters:
        model (NERModel): The model to load.
        """
        pass
    
    @abstractmethod
    def apply_ner(self, text):
        """
        Applies NER with the loaded model on `text`

        Parameters:
        text ([str]): List of the statements on which NER is sequential applied. The statements can't be bigger than the max token size (512 tokens for BERT).

        Returns:
        []: The NER-Result for each statement in the framework-specific style.
        """
        pass

    @abstractmethod
    def prepare_training_data(self,rows, tokenizer_path, train_size=0.8, validation_size=0, test_size=0.2, split_sentences=False):
        """
        Convert the rows to the framework specific finetuning/training format

        Parameters:
        rows ([ADGRow]): Input data
        tokenizer_path (str): Path of the tokenizer from the model which should be finetuned
        train_size (float): Share of the rows can bd used for training
        validation_size (float): Share of the rows can be used for validation
        test_size (float): Share of the rows can be used for testing
        split_sentences (bool): Should the statements be split for finetuning

        Returns:
        ({}, {}): First dict contains the split data, second contains the entity types with their numbers
        """
        pass
    
    @abstractmethod
    def finetune_ner_model(self,base_model_path,data_dict, label_id,name,new_model_path):
        pass

    @abstractmethod
    def convert_ner_results(self,ner_results, ner_input):
        pass