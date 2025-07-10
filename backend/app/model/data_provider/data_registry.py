import csv
from operator import truediv

import spacy
import re
import json

from .adg_row import ADGRow, extract_ADG_row
from app.utils.helpers import get_current_datetime
from app.utils.json_manager import JsonManager
from dataclasses import dataclass, asdict
from app.utils.config import TRAININGSDATA_METADATA_PATH, TRAININGSDATA_CONVERTED_PATH, DEFAULT_TOKENIZER_PATH


@dataclass
class TrainingData:
    id: int
    name: str
    path: str
    upload_date : str

def simple_split_sentences(text):
    doc = DataRegistry.nlp(text)
    sentence_indexes = []
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
        sentence_indexes.append((sent.start, sent.end))
    return sentences, sentence_indexes

def simple_tokenizer(text):
    doc = DataRegistry.nlp.tokenizer(text)
    indexes = [token.idx for token in doc]
    tokens = [token.text for token in doc]
    return tokens, indexes

class DataRegistry:
    _instance = None
    # is available for all instances
    nlp = spacy.load(DEFAULT_TOKENIZER_PATH)

    # Singleton
    def __new__(cls, *args,**kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path_metadata):
        self._json_manager = JsonManager(path_metadata)
        data_as_list = self._json_manager.load_json()
        if data_as_list is not None:
            self._datasets = [TrainingData(**d) for d in data_as_list]
        else:
            self._datasets = []

    def load_training_data(self, trainingsdata_id):
        index = self._get_index_trainingsdata_id(trainingsdata_id)
        path_trainingsdata = self._datasets[index].path
        rows = self._load_rows_as_json(path_trainingsdata)
        return rows

    def list_training_data(self):
        return self._datasets

    # only supported format is the adg-format or normal text
    # bert Modelle have a maximal input of 512 tokens -> rest is cut
    # every interview statement is processed sequential
        # split into sentences
    # split in two functions
    # only convert a file over the framework
    def prepare_data_with_labels(self, data_to_process):
        return self._read_convert_adg_file(data_to_process)

    def prepare_data_without_labels(self, data_to_process):
        doc = self.__class__.nlp(data_to_process)
        return [sent.text for sent in doc.sents]


    def add_training_data(self, dataset_name,filename, file):
        if self.save_training_data(file, TRAININGSDATA_CONVERTED_PATH, filename):
            id = self._get_next_id()
            filename_new = filename.replace(".csv",".json")
            index_store = TRAININGSDATA_CONVERTED_PATH.find("app/")
            rel_path = TRAININGSDATA_CONVERTED_PATH[index_store:]
            training_data = TrainingData(id=id, name=dataset_name, path=rel_path+"/"+filename_new, upload_date=get_current_datetime())
            self._datasets.append(training_data)
            self._update_metadata()
            return training_data
        else:
            return None

    # check the adg-format
    def save_training_data(self, file, path_to_save, filename):
        try:
            adg_rows = self._read_convert_adg_file(file)
            write_path = path_to_save+"/"+filename.replace(".csv",".json")
            self._save_rows_as_json(adg_rows,write_path)
            return True
        except Exception as e:
            print("Error in saving and converting adg file")
            return False

    def get_training_data_name(self, id):
        index = self._get_index_trainingsdata_id(id)
        return self._datasets[index].name
    
    def split_training_data_sentences(self,rows):
        sentences_tokens = []
        sentences_labels = []
        for row in rows:
            sentences_statement, sentences_indexes_statement = simple_split_sentences(row.text)
            full_tokens_sen = row.tokens
            full_labels_sen = row.labels
            for sentence in sentences_statement:
                tokens_sen, index_sen = simple_tokenizer(sentence)
                sentences_tokens.append(full_tokens_sen[:len(tokens_sen)])
                full_tokens_sen = full_tokens_sen[len(tokens_sen):]
                sentences_labels.append(full_labels_sen[:len(tokens_sen)])
                full_labels_sen = full_labels_sen[len(tokens_sen):]
        return sentences_tokens, sentences_labels

    def _read_convert_adg_file(self, file):
        reader = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONE)
        rows = []
        idx = 1
        for row in reader:
            if len(row) > 0:
                rows.append(extract_ADG_row(row, self.__class__.nlp, idx))
                idx += 1
        return rows

    def _save_rows_as_json(self, rows,path):
        rows_dicts = []
        for row in rows:
            rows_dicts.append(asdict(row))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows_dicts, f, indent=4, ensure_ascii=False)

    def _load_rows_as_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            rows_dicts = json.load(f)
        return [ADGRow(**row_dict) for row_dict in rows_dicts]

    def _get_next_id(self):
        ids = [dataset.id for dataset in self._datasets]
        if len(ids) == 0:
            return 0
        max_id = max(ids)
        for i in range(0,max_id):
            if i not in ids:
                return i
        return max_id +1
    
    def _get_index_trainingsdata_id(self, id):
        ids = [td.id for td in self._datasets]
        try:
            return ids.index(id)
        except ValueError:
            return None

    def _update_metadata(self):
        self._json_manager.update_json([asdict(dataset) for dataset in self._datasets])


data_registry = DataRegistry(TRAININGSDATA_METADATA_PATH)
