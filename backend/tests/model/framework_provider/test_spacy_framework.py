import copy
import os
import shutil
from unittest.mock import patch

import spacy
import pytest

from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider import spacy_framework
from app.model.framework_provider.framework import FrameworkNames
from app.model.framework_provider.spacy_framework import SpacyFramework
from app.model.ner_model_provider.model_registry import model_registry
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.config import BASE_MODELS_PATH, SPACY_TRAININGSDATA_PATH, MODIFIED_MODELS_PATH, STORE_PATH, \
    DEFAULT_TOKENIZER_PATH
from tests.model.framework_provider.test_framework import run_pipeline_test

"""
    Notes:
        most test are only running in the specific test environment, especially all integration tests
        they are using the datasets and models which are created in the model and data registry
        this allows the interaction of different functions to be tested
"""

# -------------------------------------
# init & helpers
# -------------------------------------
def create_get_dummy_model(model_id):
    dummy = model_registry.list_model(model_id)
    if dummy == None:
        new_model_id = model_registry.add_model(NERModel(1, "SpacyDummy",FrameworkNames.SPACY,"de_core_new_sm",BASE_MODELS_PATH+"/de_core_news_sm"))
        return new_model_id
    else:
        return model_id

def create_get_ner_bert_german(model_id):
    ner_bert_german = model_registry.list_model(model_id)
    if ner_bert_german == None:
        new_model_id = model_registry.add_model(NERModel(1, "Spacy-mschiesser/ner-bert-german", FrameworkNames.SPACY, "mschiesser/ner-bert-german",
                                                         BASE_MODELS_PATH + "/spacy_ner-bert-german/model-best"))
        return new_model_id
    else:
        return model_id

# -------------------------------------
# unit tests
# -------------------------------------

def test_load_model():
    sf = SpacyFramework()
    with pytest.raises(TypeError):
        sf.load_model('test')
    #test false model
    with pytest.raises(TypeError):
        sf.load_model(model_registry.list_model(0))
    with patch ("app.model.framework_provider.spacy_framework.spacy.load", return_value=1):
        assert sf.ner_model is None
        sf.load_model(model_registry.list_model(7))
        assert sf.ner_model is not None

def test_logic_finetune_ner_model():
    with patch("app.model.framework_provider.spacy_framework.SpacyFramework._finetune_transformer_spacy", return_value=(1,1)) as mock_transformer_finetune, \
        patch("app.model.framework_provider.spacy_framework.SpacyFramework._finetune_default_spacy", return_value=(2,2)) as mock_default_finetune:
            sf = SpacyFramework()
            base_model = model_registry.list_model(7)
            assert sf.finetune_ner_model(base_model.storage_path, {}, {}, "test",MODIFIED_MODELS_PATH, {}) == (2,2)
            base_model_transformer = model_registry.list_model(8)
            assert sf.finetune_ner_model(base_model_transformer.storage_path, {}, {}, "test",MODIFIED_MODELS_PATH, {}) == (1,1)

def test_convert_ner_results_adg(dataset_id=3):
    rows =data_registry.load_training_data(dataset_id)
    tokens = [row.tokens for row in rows]
    result = ([],copy.deepcopy(tokens))
    sf = SpacyFramework()
    tokens[0].pop(0)
    with pytest.raises(ValueError):
        sf._convert_ner_results_adg(result, rows, None)

def test_get_correct_model_path():
    sf = SpacyFramework()
    test_path = "/test"
    with patch("os.path.isfile") as mock_isfile:
        mock_isfile.side_effect = lambda p: p == (test_path+ "/meta.json")
        assert sf._get_correct_model_path(test_path) == test_path
        mock_isfile.side_effect = lambda p: p == (test_path + "/model-best/meta.json")
        assert test_path+ "/model-best" == sf._get_correct_model_path(test_path)
        with pytest.raises(ValueError):
            new_test_path  = test_path + "/t/"
            sf._get_correct_model_path(new_test_path)

# -------------------------------------
# integration tests
# -------------------------------------
def test_ner_pipeline(model_id=8, training_data_id=2, size_test=50):
    sf = SpacyFramework()
    run_pipeline_test(framework=sf,training_data_id=training_data_id, model_id=model_id, size_test=size_test)


def test_load_apply_ner_bert_german_model(model_id=8,dataset_id=2, test_size=100):
    model_id = create_get_ner_bert_german(model_id)
    sf = spacy_framework.SpacyFramework()
    sf.load_model(model_registry.list_model(model_id))
    rows =data_registry.load_training_data(dataset_id)
    texts = [row.text for row in rows]
    results, model_tokens = sf.apply_ner(texts)
    for index, res in enumerate(results):
        for ent in res:
            ent_text = ent["text"]
            start = ent["start_pos"]
            text_row = rows[index].text[start:start+len(ent_text)]
            # check if position of the extracted entities is correct
            assert ent_text == text_row
            start_token = ent["start_token"]
            end_token = ent["end_token"]
            ent_tokens =rows[index].tokens[start_token:end_token]
            for ent_token in ent_tokens:
                assert ent_token in ent["text"]
    # check if the model tokens and the default data tokens are the same
    for i,m_token in enumerate(model_tokens):
        assert m_token == rows[i].tokens

def test_check_if_the_model_tokens_are_the_same_as_the_default_tokens_for_all_datasets():
    for i in range(0,5):
        test_load_apply_ner_bert_german_model(model_id=8, dataset_id=i)

def test_prepare_training_data(training_data_id=2, split_sen=True):
    rows = data_registry.load_training_data(training_data_id)
    sf = SpacyFramework()
    sf.prepare_training_data(rows, tokenizer_path=None, train_size=0.8, validation_size=0.2, split_sentences=split_sen)
    train_examples = sf._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/train.spacy")
    valid_examples = sf._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/valid.spacy")
    train_texts = [example.reference.text for example in train_examples]
    valid_texts = [example.reference.text for example in valid_examples]

    sentences = data_registry.split_training_data_sentences(rows)
    sentences_text = [sen.text for sen in sentences]

    #check if the length is the same
    if(split_sen):
        assert len(sentences_text) == (len(train_texts) + len(valid_texts))
    else:
        assert len(rows) == (len(train_texts) + len(valid_texts))
    doc_bin_files_test(rows)

def doc_bin_files_test(rows):
    sf = SpacyFramework()
    train_examples = sf._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/train.spacy")
    valid_examples = sf._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/valid.spacy")
    counts = {}
    for train_example in train_examples:
        print(train_example.text)
        for ent in train_example.reference.ents:
            print(f"Entity: {ent.text}, Label: {ent.label_}, Start Char: {ent.start_char}, End Char: {ent.end_char}")
            counts[ent.label_] = counts.get(ent.label_, 0) + 1
        print("-" * 40)

    counts_valid = {}
    for valid_example in valid_examples:
        for ent in valid_example.reference.ents:
            counts_valid[ent.label_] = counts_valid.get(ent.label_, 0) + 1

    print("Label counts:", counts)
    print("Label counts_valid:", counts_valid)

    counts_real = {}
    for row in rows:
        for label in row.labels:
            if label != "O" and label[0] != "I":
                counts_real[label[2:]] = counts_real.get(label[2:],0) + 1
        """
        for ent in row.entities:
            counts_real[ent["typ"]] = counts_real.get(ent["typ"], 0) + len(ent["indexes"])
        """
    print("real label counts: ",counts_real)

    for key in counts_real.keys():
        assert  abs((counts.get(key,0) + counts_valid.get(key,0))-counts_real[key]) < 3

def test_finetune(base_model_id=7, dataset_size=100):
    # test a very fast finetune
    base_model = model_registry.list_model(base_model_id)
    modified_model =model_registry.create_modified_model("TestSpacy",base_model)
    sf = SpacyFramework()
    rows = data_registry.load_training_data(0)[100:100+dataset_size]
    sf.prepare_training_data(rows, tokenizer_path=None)
    metrics, params = sf.finetune_ner_model(base_model.storage_path,None,None,modified_model.name,STORE_PATH+"/Temp",{"max_epochs":0,"max_steps":2,"eval_frequency":1})
    assert isinstance(metrics, TrainingResults)

def test_finetune_default_spacy(base_model_id=7, dataset_size=100):
    # test a very fast finetune
    base_model = model_registry.list_model(base_model_id)
    modified_model =model_registry.create_modified_model("TestSpacyDummy",base_model)
    sf = SpacyFramework()
    rows = data_registry.load_training_data(0)[100:100+dataset_size]
    sf.prepare_training_data(rows, tokenizer_path=None)
    metrics, args = sf.finetune_ner_model(base_model.storage_path,None,None,modified_model.name,modified_model.storage_path,None)
    assert isinstance(metrics, TrainingResults)


def test_evaluate_finetune_spacy(model_id=7):
    ff = SpacyFramework()
    base_model = model_registry.list_model(model_id)
    metrics = ff._evaluate_transformer_model(base_model.storage_path)
    assert isinstance(metrics, TrainingResults)
    # check if some entities were recognized
    assert metrics.f1 > 0
    assert metrics.precision > 0
    assert metrics.recall > 0

def test_write_read_spacy_files(dataset_id=4):
    sf = SpacyFramework()
    nlp = spacy.load(DEFAULT_TOKENIZER_PATH)
    rows = data_registry.load_training_data(dataset_id)
    sf._bio_to_spacy(rows, STORE_PATH+"/Temp/test.spacy")
    examples = sf._get_training_examples_docbin(STORE_PATH+"/Temp/test.spacy")
    #check if the right document was written
    assert len(examples) == len(rows)

