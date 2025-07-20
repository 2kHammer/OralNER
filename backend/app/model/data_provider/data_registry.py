import csv

import spacy
import json

from .adg_row import ADGRow, extract_adg_row, ADGSentence
from app.utils.helpers import get_current_datetime
from app.utils.json_manager import JsonManager
from dataclasses import dataclass, asdict
from app.utils.config import TRAININGSDATA_METADATA_PATH, TRAININGSDATA_CONVERTED_PATH, DEFAULT_TOKENIZER_PATH

@dataclass
class TrainingData:
    """
    Represents the metadata of a dataset
    """
    id: int
    name: str
    path: str
    upload_date : str

# --------------------------------------
# public functions
# --------------------------------------


# -------------------------------------
# class "DataRegistry"
# -------------------------------------
class DataRegistry:
    """
    Manages the the trainings data sets with related metadata. Can store and load them.
    """
    _instance = None
    # is available for all instances
    nlp = spacy.load(DEFAULT_TOKENIZER_PATH)

    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path_metadata, path_store=None):
        self._json_manager = JsonManager(path_metadata)
        if path_store is None:
            index_app = TRAININGSDATA_CONVERTED_PATH.find("app/")
            trainingsdata_rel_path = TRAININGSDATA_CONVERTED_PATH[index_app:]
            self._data_path = trainingsdata_rel_path
        else:
            self._data_path = path_store
        data_as_list = self._json_manager.load_json()
        if data_as_list is not None:
            self._datasets = [TrainingData(**d) for d in data_as_list]
        else:
            self._datasets = []

    # -------------------------------------
    # public functions
    # -------------------------------------
    def load_training_data(self, trainingsdata_id):
        """
        Returns the ADGRow from the trainings dataset with `trainingsdata_id`

        Parameters:
        trainingsdata_id (int): the id of the training dataset

        Returns:
        (List[ADGRow]): if the `trainingsdata_id` is existent
        """
        index = self._get_index_trainingsdata_id(trainingsdata_id)
        if index is None:
            return None
        path_trainingsdata = self._datasets[index].path
        rows = self._load_rows_as_json(path_trainingsdata)
        return rows

    def list_training_data(self):
        """
        Returns the metadata of all datasets

        Returns:
        (List[TrainingData])
        """
        return self._datasets

    def prepare_data_with_labels(self, data_to_process):
        """
        Converts the file `data_to_process` into a list of ADGRows

        Parameters:
        data_to_process (file): csv-file in the ADG-Format

        Returns:
        (List[ADGRow])
        """
        return self._read_convert_adg_file(data_to_process)

    def prepare_data_without_labels(self, data_to_process):
        """
        Converts the `data_to_process` text into a list of sentences

        Parameters:
        data_to_process (str)

        Return:
        (List[str])
        """
        return self._simple_split_sentences(data_to_process)[0]

    def add_training_data(self, dataset_name,filename, file, path=None):
        """
        Saves the dataset in `file` and updates the belonging metadata

        Parameters
        dataset_name (str): the name of the dataset
        filename (str): the name of the file
        file (file): the dataset file
        path (str): the path where the file should be stored, if none specified: default path

        Returns
        (TrainingData): the metadata of the added dataset
        """
        # get the path
        filename_new = filename.replace(".csv", ".json")
        data_path = self._data_path
        if path is not None:
            data_path = path
        complete_file_path = data_path + "/" + filename_new

        if self.save_training_data(file, data_path, filename_new):
            # overwrite dataset with the same path
            paths = [ds.path for ds in self._datasets]
            new_trainingsdata = None
            if complete_file_path in paths:
                new_trainingsdata = self._datasets[paths.index(complete_file_path)]
                new_trainingsdata.name = dataset_name
                new_trainingsdata.upload_date = get_current_datetime()
            else:
                id = self._get_next_id()
                new_trainingsdata = TrainingData(id=id, name=dataset_name, path=complete_file_path, upload_date=get_current_datetime())
                self._datasets.append(new_trainingsdata)
            self._update_metadata()
            return new_trainingsdata
        else:
            return None

    def save_training_data(self, file, path_to_save, filename_json):
        """
        Converts and saves the `file` to ADGRow and Json

        Parameters
        file (file)
        path_to_save (str): path where the json file should be stored
        filename_json (str): name of the json file

        Returns
        (bool)
        """
        adg_rows = self._read_convert_adg_file(file)
        if adg_rows is not None:
            write_path = path_to_save + "/" + filename_json
            self._save_rows_as_json(adg_rows, write_path)
            return True
        else:
            return False

    def get_training_data_name(self, id):
        """
        Returns the name of the training dataset with `id`

        Parameters
        id (int)

        Returns
        (str)
        """
        index = self._get_index_trainingsdata_id(id)
        if index is not None:
            return self._datasets[index].name
        else:
            return ""
    
    def split_training_data_sentences(self,rows):
        """
        Converts lists of ADGRows into ADGSentences

        Parameters
        rows (List[ADGRow])

        Returns
        (List[ADGSentence])
        """
        sentences_data = []
        for row in rows:
            row_index = row.idx
            sentences_statement, sentences_indexes_statement = self._simple_split_sentences(row.text)
            full_tokens_sen = row.tokens
            full_labels_sen = row.labels
            full_indexes_sen = row.indexes
            for ind, sentence in enumerate(sentences_statement):
                tokens_sen, index_sen = self._simple_tokenizer(sentence)
                #tokens
                sentence_tokens =full_tokens_sen[:len(tokens_sen)]
                full_tokens_sen = full_tokens_sen[len(tokens_sen):]
                #labels
                sentence_labels = full_labels_sen[:len(tokens_sen)]
                full_labels_sen = full_labels_sen[len(tokens_sen):]
                #indexes
                sentence_token_indexes = full_indexes_sen[:len(tokens_sen)]
                full_indexes_sen = full_indexes_sen[len(tokens_sen):]
                startind = sentence_token_indexes[0]
                adapted_token_indexes = [ind - startind for ind in sentence_token_indexes]
                sentences_data.append(ADGSentence(sentence, sentence_tokens, sentence_labels, sentences_indexes_statement[ind], row_index, adapted_token_indexes))

        #remove irrelevant sentences, problems with flair
        sentences_data = [sen for sen in sentences_data if (sen.text != '"' and sen.text != ' ')]
        return sentences_data

    def check_convert_adg_file(self, file, steps=5):
        """
        Checks on sampled lines of file is in the ADG-Format

        Parameters
        file (file)
        steps (int): every `steps` lines are checked

        Returns
        (bool)
        """
        one_line_checked = False
        try:
            reader = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONE)
            step = 0
            for row in reader:
                step += 1
                if step == steps:
                    if len(row) > 0:
                        extract_adg_row(row, self.__class__.nlp, 1)
                        one_line_checked = True
                    step = 0
        except:
            return False
        return one_line_checked

    def _read_convert_adg_file(self, file):
        """
        Convert `file` to a list of ADGRows

        Parameters
        file (file)

        Returns
        (List[ADGRow])
        """
        reader = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONE)
        rows = []
        idx = 1
        try:
            for row in reader:
                if len(row) > 0:
                    rows.append(extract_adg_row(row, self.__class__.nlp, idx))
                    idx += 1
            return rows
        except Exception as e:
            print("Error in saving and converting adg file")
            return None


    def _save_rows_as_json(self, rows,path):
        """
        Converts the `rows` to dicts and saves them as JSON

        Parameters
        rows (List[ADGRow])
        path (str)
        """
        rows_dicts = []
        for row in rows:
            rows_dicts.append(asdict(row))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows_dicts, f, indent=4, ensure_ascii=False)

    def _load_rows_as_json(self, path):
        """
        Loads the metadata from the json file in `path`

        Parameters
        path (str)

        Returns
        (List[ADGRow])
        """
        with open(path, "r", encoding="utf-8") as f:
            rows_dicts = json.load(f)
        return [ADGRow(**row_dict) for row_dict in rows_dicts]

    def _get_next_id(self):
        """
        Return the next free ID in ._datasets

        Returns
        (int)
        """
        ids = [dataset.id for dataset in self._datasets]
        if len(ids) == 0:
            return 0
        max_id = max(ids)
        for i in range(0,max_id):
            if i not in ids:
                return i
        return max_id +1
    
    def _get_index_trainingsdata_id(self, id):
        """
        Return the index for the dataset with `id` in self._datasets

        Parameters
        id (int)

        Returns
        (int)
        """
        ids = [td.id for td in self._datasets]
        try:
            return ids.index(id)
        except ValueError:
            return None

    def _update_metadata(self):
        """Saves the Model Metadata to JSON"""
        self._json_manager.update_json([asdict(dataset) for dataset in self._datasets])

    def _simple_tokenizer(self,text):
        """
        Convert a Text to tokens.

        Parameters
        text(str)

        Returns
        (List[str],List[int]): the first list contains the tokens, the second list contains the startindex of the token in `text`
        """
        doc = DataRegistry.nlp.tokenizer(text)
        indexes = [token.idx for token in doc]
        tokens = [token.text for token in doc]
        return tokens, indexes

    def _simple_split_sentences(self,text):
        """
        Convert a Text into his sentences.

        Parameters
        text(str)

        Returns
        (List[str],List[int]): the first list contains the sentences, the second list contains the start- and endindexes of the tokens in `text`
        """
        doc = self.nlp(text)
        sentence_indexes = []
        sentences = []
        for sent in doc.sents:

            sentences.append(sent.text)
            sentence_indexes.append((sent.start, sent.end))
        return sentences, sentence_indexes

    @classmethod
    def _reset_instance(cls):
        """ Resets the singleton, only for testing purposes """
        cls._instance = None

# creating of the Data Registry
data_registry = DataRegistry(TRAININGSDATA_METADATA_PATH)
