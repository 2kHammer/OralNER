from abc import ABC, abstractmethod
from enum import Enum
import random
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

class FrameworkNames(Enum):
    HUGGINGFACE = 1
    FLAIR = 2
    SPACY = 3

entity_types = []

class Framework(ABC):
    @property
    @abstractmethod
    def default_finetuning_params(self):
        """
        Returns a dictionary with the default finetuning params.
        """
        pass

    @abstractmethod
    def load_model(self,model):
        """
        Load the `model` in the class. Necessary for the other functions.

        Parameters:
        model (NERModel): The model to load.
        """
        pass

    # think about if the texts should be automatically be splitted into sentences
    @abstractmethod
    def apply_ner(self, texts):
        """
        Applies NER with the loaded model on `text`

        Parameters:
        text ([str]): List of the statements on which NER is sequential applied. The statements can't be bigger than the max token size (512 tokens for BERT).

        Returns:
        The NER-Result for each statement in the framework-specific style.
        """
        pass

    @abstractmethod
    def prepare_training_data(self,rows, tokenizer_path=None, train_size=0.8, validation_size=0.1, test_size=0.1, split_sentences=False):
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
        (any, dict): first object contains the split data in the framework specific format, second contains the entity types with their numbers
        """
        pass
    
    @abstractmethod
    def finetune_ner_model(self,base_model_path,data_dict, label_id,name,new_model_path, params=None):
        """
        Finetune the NER model under `base_model_path`

        Parameters:
        base_model_path (str): path were the model to finetune is saved
        data_dict (dict): contains the split data in the framework specific format
        label_id (dict): contains the entity types with their numbers
        name (str): Name of the finetuned model
        new_model_path (str): path were the model to finetune is saved
        params (dict): framework specific parameters for the finetuning

        Returns:
        (TrainingResults, dict)
        """
        pass

    @abstractmethod
    def convert_ner_results(self,ner_results, ner_input):
        """
        Converts the framework-specific `ner-results` to BIO-format. If the input data is in the format List[ADGRow], you receive metrics.

        Parameters:
        ner_results (List): List of framework-specific NER results
        ner_input (List): List of strings or of ADGRow

        Returns:
        (List, List, Dict): First List contains the tokens, second the predicted lables and the dict the metrics
        """
        pass

    def _calc_metrics(self, annoted_labels, predicted_labels):
        return {
            "f1": round(float(f1_score(annoted_labels, predicted_labels)),2),
            "recall": round(float(recall_score(annoted_labels, predicted_labels)),2),
            "precision": round(float(precision_score(annoted_labels, predicted_labels)),2),
            "accuracy": round(float(accuracy_score(annoted_labels, predicted_labels)),2)
        }

    # only für flair and spacy
    def _convert_ner_results_to_format(self, ner_results):
        results, tokens = ner_results
        labels = []
        for index, token_sentence in enumerate(tokens):
            label_sentence = ["O"] * len(token_sentence)
            for entity in results[index]:
                start_token = entity["start_token"]
                end_token = entity["end_token"]
                typ = entity["type"]
                label_sentence[start_token] = "B-" + typ
                for i in range(start_token + 1, end_token + 1):
                    label_sentence[i] = "I-" + typ
            labels.append(label_sentence)
        return tokens, labels

    def _train_test_split(self,data, train_size=0.8, valid_size=0.1, test_size=0.1, seed=True):
        if seed:
            random.seed(42)
        shuffled = data.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_size)
        valid_end = int(n * (valid_size + train_size))
        return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]