from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from app.model.data_provider.data_registry import data_registry
from app.model.ner_model_provider.model_registry import model_registry
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.model.ner_model_provider.ner_model import TrainingResults
from app.utils.config import BASE_MODELS_PATH, STORE_TEMP_PATH
from tests.model.framework_provider.test_framework import run_pipeline_test

path_tokenizer = BASE_MODELS_PATH + "/mschiesser_ner-bert-german"
possible_entities = ["PER","ROLE","ORG","LOC","WORK_OF_ART","NORP","EVENT","DATE"]

"""
    Notes:
        most test are only running in the specific test environment, especially all integration tests
        they are using the datasets and models which are created in the model and data registry
        this allows the interaction of different functions to be tested
"""
# -------------------------------------
# unit tests
# -------------------------------------
def test_load_model():
    hf = HuggingFaceFramework()
    with pytest.raises(TypeError):
        hf.load_model('test')
    #test false model
    with pytest.raises(TypeError):
        hf.load_model(model_registry.list_model(3))
    with patch ("app.model.framework_provider.huggingface_framework.AutoModelForTokenClassification.from_pretrained", return_value=1), \
        patch("app.model.framework_provider.huggingface_framework.AutoTokenizer.from_pretrained",return_value=1):
        assert hf.model is None
        assert hf.tokenizer is None
        hf.load_model(model_registry.list_model(0))
        assert hf.model is not None
        assert hf.tokenizer is not None

def test_tokenize_align_labels():
    hf = HuggingFaceFramework()
    base_hf_model = model_registry.list_model(0)
    test_statement = Dataset.from_list([{"tokens":["Alexander","Hammer","feiert","Weihnachten"],"labels":["B-PER","I-PER","O","B-EVENT"]}])
    res =hf._tokenize_and_align_labels(test_statement, base_hf_model.storage_path)
    # the labels should have the same length as the splitted tokens
    assert len(res.labels[0]) == len(res.input_ids[0])
    # first and last label should be -100
    assert (res.labels[0][0] == -100) and (res.labels[0][-1] == -100)

def test_convert_ner_results_adg_example():
    hf = HuggingFaceFramework()
    # this examples covers all edge cases of the function
    test_labels = ['O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NORP', 'I-NORP', 'O', 'O', 'O', 'B-LOC', 'O']
    test_tokens = ['Wir', 'kamen', 'aus', 'Norddeutschland', ',', 'wir', 'waren', 'mit', 'der', 'Eisenbahn', 'gefahren', ',', 'und', 'da', 'wollte', 'man', '...', 'Deutsche', ',', 'deutsche', 'Polizeikräfte', ',', 'keine', ',', 'keine', ',', 'keine', 'Franzosen', ',', 'auf', 'dem', 'Bahnhof', '...']
    test_indexes = [0, 4, 10, 14, 29, 31, 35, 41, 45, 49, 59, 67, 69, 73, 76, 83, 87, 91, 99, 101, 110, 123, 125, 130, 132, 137, 139, 145, 154, 156, 160, 164, 172]
    # manually moved the start of the first entity +1 -> from 14 to 15 (to cover all edge cases)
    test_results = [[{'end': 18, 'entity': 'B-LOC', 'index': 4, 'score': 0.9981377, 'start': 14, 'word': 'Nord'}, {'end': 29, 'entity': 'B-LOC', 'index': 5, 'score': 0.9985862, 'start': 18, 'word': '##deutschland'}, {'end': 58, 'entity': 'B-ORG', 'index': 11, 'score': 0.9949338, 'start': 49, 'word': 'Eisenbahn'}]]
    #result = hf.apply_ner([test_row.text])
    tokens_res, labels_res, metrics = hf._convert_ner_results_adg(test_results, [SimpleNamespace(labels=test_labels, tokens=test_tokens, indexes=test_indexes)], None)
    assert test_tokens == tokens_res[0]
    ent1 = test_results[0][0]
    startindex_bloc = 3
    assert labels_res[0][startindex_bloc] == "B-LOC" and labels_res[0][startindex_bloc+1] != "B-LOC"
    assert labels_res[0][9] == "B-ORG"

# -------------------------------------
# integration tests
# -------------------------------------
def test_ner_pipeline(training_data_id=3, model_id=0, size_test=50):
    hf = HuggingFaceFramework()
    run_pipeline_test(framework=hf,training_data_id=training_data_id, model_id=model_id, size_test=size_test)


def test_check_if_entity_types_are_possible(dataset_id=4):
    hf = HuggingFaceFramework()
    rows = data_registry.load_training_data(dataset_id)
    dataset, label_id = hf.prepare_training_data(rows,path_tokenizer)
    types = list(label_id.keys())
    for type in types:
        if type != "O":
            assert type[2:] in possible_entities

def test_prepare_training_data(dataset_id=3, test_size = 150):
    hf = HuggingFaceFramework()
    rows = data_registry.load_training_data(dataset_id)[100:100+test_size]
    dataset, label_id = hf.prepare_training_data(rows,path_tokenizer,seed=42)
    tokens_row = [row.tokens for row in rows]
    # check if sizes match
    assert len(rows) == (len(dataset["train"]) + len(dataset["validation"]))
    # check if every statement is in the dataset dict
    for tokens_statement in tokens_row:
        assert ((tokens_statement in dataset["train"]["tokens"]) or (tokens_statement in dataset["validation"]["tokens"]))

    """
    tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
    # View test if the splitted tokens match to the annoted entities
    labels_row = [row.labels for row in rows]
    id_label = {v:k for k,v in label_id.items()}
    for index, labels in enumerate(dataset["train"]["labels"]):
        print(dataset["train"]["tokens"][index])
        label_back = [id_label.get(label, "-100") for label in labels]
        assert len(label_back) == len(dataset["train"]["input_ids"][index])
        subtokens = tokenizer.convert_ids_to_tokens(dataset["train"]["input_ids"][index])
        for index_label, label in enumerate(label_back):
            if label != "O" and label != "-100":
                print(subtokens[index_label], label)

        print("-"*50)
    """

    """
    check the mapping of the labels
    doesn't work 
        because convert_ids_to_tokens() doesnt always reconstruct the original token 
        make better maye
        
    id_label = {v: k for k, v in label_id.items()}
    for index, tokens in enumerate(dataset["train"]["tokens"]):
        tokens_checked = False
        # find matching row
        for row in rows:
            not_the_same = False
            for token in tokens:
                if not token in row.tokens:
                    not_the_same = True

            if not not_the_same:
                # check entities
                label_back = [id_label.get(label, "-100") for label in dataset["train"]["labels"][index]]
                subtokens = tokenizer.convert_ids_to_tokens(dataset["train"]["input_ids"][index])
                ent_texts = [ent["entity_text"] for ent in row.entities]
                print(ent_texts)
                for index_label, label in enumerate(label_back):
                    if subtokens[index_label] == "Große":
                        print("test")
                    if label != "O" and label != "-100":
                        print(subtokens[index_label])
                        assert any((subtokens[index_label] in ent_text) for ent_text in ent_texts)
                tokens_checked = True
            if tokens_checked:
                break
        """


def test_finetune_ner_model(model_id=0, dataset_id=3, test_size=30):
    base_model = model_registry.list_model(model_id)
    rows = data_registry.load_training_data(dataset_id)[50:50+test_size]
    hf = HuggingFaceFramework()
    data, data_dict =hf.prepare_training_data(rows,base_model.storage_path)
    params = hf.default_finetuning_params
    params["num_train_epochs"] = 1
    modified_name = "FlairFastTest"
    test_path = STORE_TEMP_PATH
    metrics, args = hf.finetune_ner_model(base_model.storage_path,data, data_dict,modified_name,test_path,params)
    # this tests _compute_metrics and _convert_metrics
    assert isinstance(metrics, TrainingResults)

def test_convert_ner_results_not_adg_example(model_id=0,dataset_id =3, test_size=100):
    hf = HuggingFaceFramework()
    rows =data_registry.load_training_data(dataset_id)[120:120+test_size]
    texts = [row.text for row in rows]
    hf.load_model(model_registry.list_model(model_id))
    ner_result = hf.apply_ner(texts)
    tokens_res, labels_res, _ = hf.convert_ner_results(ner_result,texts)
    for index, row in enumerate(rows):
        tokens_tokenizer = hf.tokenizer(row.text).tokens()
        len_tokens_without_subtokens = 0
        for token in tokens_tokenizer:
            if token[:2] != "##":
                len_tokens_without_subtokens += 1
        # check if the length is correct and all subtokens are mapped
        assert len(tokens_res[index]) == (len_tokens_without_subtokens-2)

    # check if the labels are correct mapped
    # if only a subtoken is annoted and not the first token -> is ignored
    for index_statement,labels in enumerate(labels_res):
        entities = ner_result[index_statement]
        token_labels = tokens_res[index_statement]
        for index,label in enumerate(labels):
            if label != "O":
                token_with_entity = token_labels[index]
                # remove the next subtokens from entities
                while(len(entities)>0 and (entities[0]["word"][:2] == "##")):
                    entities.pop(0)
                entity_text = (entities.pop(0))["word"]
                assert token_with_entity.startswith(entity_text)


