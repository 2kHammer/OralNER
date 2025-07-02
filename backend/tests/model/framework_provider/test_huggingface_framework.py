from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.utils.config import BASE_MODELS_PATH

hf = HuggingFaceFramework()

path_tokenizer = BASE_MODELS_PATH + "/mschiesser_ner-bert-german"
possible_entities = ["PER","ROLE","ORG","LOC","WORK_OF_ART","NORP","EVENT","DATE"]

def check_if_entity_types_are_possible(dataset_id):
    rows = data_registry.load_training_data(dataset_id)
    dataset, label_id = hf.prepare_training_data(rows,path_tokenizer)
    types = list(label_id.keys())
    for type in types:
        if type != "O":
            assert type[2:] in possible_entities

def test_entity_types():
    for i in range(0,4):
        check_if_entity_types_are_possible(i)

def check_split_size(dataset_id=0):
    rows = data_registry.load_training_data(dataset_id)
    dataset, label_id = hf.prepare_training_data(rows, path_tokenizer, 0.4,0.2,0.4)
    len_ds = len(rows)-1
    train_size = dataset["train"].num_rows
    test_size = dataset["test"].num_rows
    vali_size = dataset["validation"].num_rows
    assert (train_size + test_size + vali_size) == len_ds
    assert train_size/ len_ds >= 0.39 and train_size/ len_ds <= 0.41
    assert vali_size/ len_ds >= 0.19 and vali_size/ len_ds <= 0.21

def test_split_sizes():
    for i in range(0,4):
        check_split_size(i)
