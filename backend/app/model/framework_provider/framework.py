from abc import ABC, abstractmethod
from enum import Enum
import random
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score



class FrameworkNames(Enum):
    HUGGINGFACE = 1
    FLAIR = 2
    SPACY = 3

entity_types = []


# -------------------------------------
# abstract class Framework
# -------------------------------------
class Framework(ABC):
    """Abstract class for the frameworks"""

    # -------------------------------------
    # public functions
    # -------------------------------------
    @property
    @abstractmethod
    def default_finetuning_params(self):
        """
        Returns a dictionary with the default finetuning params
        """
        pass

    @abstractmethod
    def load_model(self,model):
        """
        Load the `model` in the class. Necessary for the other functions.

        Parameters:
        model (NERModel): The model to load
        """
        pass

    @abstractmethod
    def process_ner_pipeline(self,model, ner_content, use_sentences = False):
        """
        Loads the `model`, applys NER and converts the results

        Parameters:
        model (NERModel): The model to load
        ner_content (List[str] | List[ADGRow]): the content on which ner should be applied
        use_sentences (boolean): only relevant for adg_rows, should the statement be splitted into sentences

        Returns:
        (List, List, Dict): tokens, labels, Metrics
        """
        pass

    # think about if the texts should be automatically be splitted into sentences
    @abstractmethod
    def apply_ner(self, texts):
        """
        Applies NER with the loaded model on `texts`

        Parameters:
        text (List[str]): List of the statements on which NER is sequential applied.

        Returns:
        The NER-Result for each statement in the framework-specific style.
        """
        pass

    # only flair would use 3 datasets -> use there only two
    @abstractmethod
    def prepare_training_data(self,rows, tokenizer_path=None, train_size=0.8, validation_size=0.2, split_sentences=False, seed=None):
        """
        Convert the rows to the framework specific finetuning/training format

        Parameters:
        rows (List[ADGRow]): Input data
        tokenizer_path (str): Path of the tokenizer from the model which should be finetuned
        train_size (float): Share of the rows can be used for training
        validation_size (float): Share of the rows can be used for validation
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
        base_model_path (str): path were the model to finetune is stored
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
    def convert_ner_results(self,ner_results, ner_input, sentences = None):
        """
        Converts the framework-specific `ner-results` to BIO-format. If the input data is in the format List[ADGRow], you receive metrics.

        Parameters:
        ner_results (List): List of framework-specific NER results
        ner_input (List[str] | List[ADGRow]): List of strings or of ADGRow
        sentences (List[ADGSentence]| None): List of sentences, if the ADGRows should be splitted in sentences

        Returns:
        (List, List, Dict): First List contains the tokens, second the predicted lables and the dict the metrics
        """
        pass

    # -------------------------------------
    # private non-abstract functions
    # -------------------------------------
    def _calc_metrics(self, annoted_labels, predicted_labels):
        """Calcs the default metrics from the predicted and annoted labels

        Parameters:
        annoted_labels (list[list])
        predicted_labels (list[list])

        Returns
        (dict): with "f1", "recall", "precision" and "accuracy"

        """
        return {
            "f1": round(float(f1_score(annoted_labels, predicted_labels)),2),
            "recall": round(float(recall_score(annoted_labels, predicted_labels)),2),
            "precision": round(float(precision_score(annoted_labels, predicted_labels)),2),
            "accuracy": round(float(accuracy_score(annoted_labels, predicted_labels)),2)
        }

    # only für flair and spacy
    def _convert_ner_results_to_format(self, ner_results):
        """
        Convert the `ner-results` to tokens and and labels

        Parameters:
        ner_results (list, list): the output from flair and spacy apply_ner(), the second list contains the tokens

        Returns:
        (list, list): the first list contains the tokens, the second the annoted labels
        """
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

    # danger of data leakage: some sentences of statement in train, some in test
    # -> the statements should be splitted and the parted into sentences
    def _train_test_split(self,data, train_size=0.8, valid_size=0.2, test_size=0, seed=None):
        """
        Splits the `data` randomly according to the specified sizes

        Parameters
        data (list): the data to split, splits the elements in the list
        train_size (float):  share the training dataset should get
        valid_size (float): share the valid dataset should get
        test_size (float): share the test dataset should get
        seed (int): for reproducibility

        Returns:
        (list, list,list): train, valid and test
        """
        if seed:
            random.seed(seed)
        shuffled = data.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_size)
        valid_end = int(n * (valid_size + train_size))
        return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]

