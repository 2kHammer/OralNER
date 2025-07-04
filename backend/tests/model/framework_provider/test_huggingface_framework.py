from app.model.data_provider.data_registry import data_registry
from app.model.ner_model_provider.model_registry import model_registry
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.utils.config import BASE_MODELS_PATH

hf = HuggingFaceFramework()

path_tokenizer = BASE_MODELS_PATH + "/mschiesser_ner-bert-german"
possible_entities = ["PER","ROLE","ORG","LOC","WORK_OF_ART","NORP","EVENT","DATE"]

'''
    Unit Tests
'''

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

def check_split_size(dataset_id=0, split_sentences=False):
    rows = data_registry.load_training_data(dataset_id)
    dataset, label_id = hf.prepare_training_data(rows, path_tokenizer, 0.4,0.2,0.4, split_sentences)
    len_ds = len(rows)-1
    if split_sentences:
        sentences_tokens, sentences_labels = hf._split_training_data_sentences(rows)
        len_ds = len(sentences_tokens)-1
    train_size = dataset["train"].num_rows
    test_size = dataset["test"].num_rows
    vali_size = dataset["validation"].num_rows
    assert (train_size + test_size + vali_size) == len_ds
    assert train_size/ len_ds >= 0.39 and train_size/ len_ds <= 0.41
    assert vali_size/ len_ds >= 0.19 and vali_size/ len_ds <= 0.21

def test_split_sizes():
    for i in range(0,4):
        check_split_size(i, True)



def test_ner_results_adg(dataset_id=0):
    hf.load_model(model_registry.current_model)
    rows = data_registry.load_training_data(dataset_id)
    rows = rows[:100]
    texts = [row.text for row in rows]
    ner_results = hf.apply_ner(texts)
    tokens, predicted_labels, metrics  = hf.convert_ner_results(ner_results,rows)

    for index, pred_labels in enumerate(predicted_labels):
        assert len(pred_labels) == len(rows[index].tokens)

'''
    Integration Tests
'''
