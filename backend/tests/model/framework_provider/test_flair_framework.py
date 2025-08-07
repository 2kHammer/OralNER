from itertools import chain
from unittest.mock import patch, Mock, mock_open

from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.flair_framework import FlairFramework, save_to_conll
from app.model.framework_provider.framework import FrameworkNames, Framework
from app.model.ner_model_provider.model_registry import model_registry
from app.model.ner_model_provider.ner_model import TrainingResults
from app.utils.config import MODIFIED_MODELS_PATH, STORE_PATH, STORE_TEMP_PATH
from tests.model.framework_provider.test_framework import run_pipeline_test
from tests.model.framework_provider.test_spacy_framework import base_model_id

"""
    Notes:
        most test are only running in the specific test environment, especially all integration tests
        they are using the datasets and models which are created in the model and data registry
        this allows the interaction of different functions to be tested
"""
base_model_id =2
finetuned_model_id = 26
# -------------------------------------
# unit tests
# -------------------------------------
def test_apply_ner():
    text = [["Auf","Tokens","angewandt"]]
    text2 = ["Auf Sätze angewandt"]
    ff = FlairFramework()
    ff.model =Mock()
    with patch("app.model.framework_provider.flair_framework.Sentence") as mock_sentence:
        ff.apply_ner(text)
        mock_sentence.assert_any_call(text[0],use_tokenizer=False)

def test_logic_prepare_training_data(dataset_id = 2, testsize =100):
    #check sentence split
    rows = data_registry.load_training_data(dataset_id)[100:100+testsize]
    sen = data_registry.split_training_data_sentences(rows)
    ff = FlairFramework()
    with patch.object(ff, "_create_conll_files", return_value=None) as mock_create, \
        patch("app.model.framework_provider.flair_framework.ColumnCorpus") as mock_corpus:
        mock_corpus_instance = Mock()
        mock_corpus_instance.make_label_dictionary.return_value = None
        mock_corpus.return_value = mock_corpus_instance

        test = ff.prepare_training_data(rows,split_sentences=True)
        args, kwargs = mock_create.call_args
        train = args[0]
        valid = args[1]
        assert len(train)+len(valid) == len(sen)
        # check shuffle
        assert sen[3].tokens != train[3]["tokens"]

def test_create_conll_files():
    ff = FlairFramework()
    test =[{"tokens":['Hallo','ich','bin','Alex'],"labels":["O","O","O","B-PER"]}]
    valid =[{"tokens":['Tschüss','ich','bin','Alex'],"labels":["O","O","O","B-PER"]}]
    with patch("app.model.framework_provider.flair_framework.save_to_conll",return_value = None) as mock_save:
        ff._create_conll_files(test, valid)
        calls = mock_save.call_args_list
        assert(len(calls)==2)
        assert calls[0][0][0][0] == test[0]["tokens"]
        assert calls[1][0][0][0] == valid[0]["tokens"]

def test_get_pt_file():
    ff = FlairFramework()
    with patch("os.listdir",return_value=["model.pt","test.txt"]):
        assert ff._get_pt_file("/path") == "/path/model.pt"
    with patch("os.listdir",return_value=["test1.txt","test.txt"]):
        assert ff._get_pt_file("/path") == None
    with patch("os.listdir",return_value=["model.pt","model2.pt"]):
        assert ff._get_pt_file("/path") == "/path/model.pt"

def test_save_to_conll():
    tokens = [['Hallo','Alex'],['Alex']]
    labels =[["O","B-PER"],["B-PER"]]
    m = mock_open()
    with patch("builtins.open",m):
        save_to_conll(tokens,labels,"/path")

    handle = m()
    handle.write.assert_any_call("Hallo O\n")
    handle.write.assert_any_call("Alex B-PER\n")
    handle.write.assert_any_call("\n")
    handle.write.assert_any_call("Alex B-PER\n")



# -------------------------------------
# integration tests
# -------------------------------------
#3 is the id of the base model
def test_load_modified_model(model_id1=base_model_id, model_id2=finetuned_model_id):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id1))
    old_model = ff.model
    ff.load_model(model_registry.list_model(model_id2))
    assert old_model != ff.model

# maybe abstract this test #was model_id=3
def test_ner_pipeline(model_id=finetuned_model_id,training_data_id=2, size_test=50):
    ff = FlairFramework()
    run_pipeline_test(framework=ff,training_data_id=training_data_id, model_id=model_id, size_test=size_test)

def test_load_apply_model(model_id=base_model_id, training_data_id=0, size_test=50):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(training_data_id)
    sentences = [data_registry.prepare_data_without_labels(row.text) for row in rows[:size_test]]
    flat_sentences = list(chain.from_iterable(sentences))
    ner_results, tokens =ff.apply_ner(flat_sentences)
    # check if label text is corresponding to token
    for index, ner_result in enumerate(ner_results):
        if len(ner_result)> 0:
            #check only for entities with 1 token
            for ent in ner_result:
                if ent["start_token"] == ent["end_token"]:
                    assert ent["text"] == tokens[index][ent["start_token"]]

def test_prepare_training_data(model_id=base_model_id,training_data_id=1,size_test=200):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(training_data_id)[100:100+size_test]
    corpus, label_dict =ff.prepare_training_data(rows)

    #check amount labels between the rows and the corpus (without sentence split)
    amount_corpus_labels = 0
    for sen in corpus.train:
        if len(sen.annotation_layers) == 1:
            amount_corpus_labels += len(sen.annotation_layers["ner"])
    for sen in corpus.dev:
        if len(sen.annotation_layers) == 1:
            amount_corpus_labels += len(sen.annotation_layers["ner"])
    amount_rows_labels = 0
    for row in rows:
        for label in row.labels:
            if label != "O" and label[0] != "I":
                amount_rows_labels += 1
    assert amount_corpus_labels == amount_rows_labels

def test_finetune_model(model_id=base_model_id,training_data_id=1, dataset_size=100):
    ff = FlairFramework()
    test_params ={
        "learning_rate": 0.0025,
        "mini_batch_size": 64,
        "max_epochs": 1,
    }
    base_model = model_registry.list_model(model_id)
    modified_name = "FlairFastTest"
    test_path = STORE_TEMP_PATH
    rows = data_registry.load_training_data(training_data_id)
    corpus, label_dict = ff.prepare_training_data(rows, split_sentences=True,seed=42)
    metrics, args =ff.finetune_ner_model(base_model.storage_path, corpus, label_dict,modified_name,test_path, params=test_params)
    assert isinstance(metrics,TrainingResults)

def test_convert_ner_results(model_id=base_model_id, training_data_id=1, size_test=50):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(training_data_id)[200:200+size_test]

    # no adg
    sentences = data_registry.split_training_data_sentences(rows)
    flat_sentences = [sen.text for sen in sentences]
    ner_results, tokens =ff.apply_ner(flat_sentences)
    tokens, labels, metrics = ff.convert_ner_results((ner_results,tokens),flat_sentences)
    #check if all labels from ner_results are in labels
    for index,label_sen in enumerate(labels):
        indexes_labels = []
        for ner_result in ner_results[index]:
            for i in range(ner_result["start_token"], ner_result["end_token"]+1):
                indexes_labels.append(i)
        for index_label,label in enumerate(label_sen):
            if label != "O":
                assert index_label in indexes_labels

    # test adg
    results_adg = ff.apply_ner([sen.tokens for sen in sentences])
    tokens_adg, labels_adg, metrics = ff.convert_ner_results(results_adg,rows, sentences)
    expected_metrics = {'f1', 'recall', 'precision', 'accuracy'}
    assert expected_metrics.issubset(metrics.keys())




