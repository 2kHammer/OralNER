from itertools import chain

from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.flair_framework import FlairFramework, train_test_split
from app.model.framework_provider.framework import FrameworkNames
from app.model.ner_model_provider.model_registry import model_registry
from app.utils.config import MODIFIED_MODELS_PATH


#3 is the id of the base model
def test_load_apply_model(model_id=3, training_data_id=0, size_test=50):
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

def test_convert_ner_results_not_adg(model_id=3, training_data_id=1, size_test=100):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(training_data_id)
    sentences = [data_registry.prepare_data_without_labels(row.text) for row in rows[50:50+size_test]]
    flat_sentences = list(chain.from_iterable(sentences))
    ner_results, tokens =ff.apply_ner(flat_sentences)
    tokens, labels, metrics = ff.convert_ner_results((ner_results,tokens),flat_sentences)
    for index,label_sen in enumerate(labels):
        indexes_labels = []
        for ner_result in ner_results[index]:
            for i in range(ner_result["start_token"], ner_result["end_token"]+1):
                indexes_labels.append(i)
        for index_label,label in enumerate(label_sen):
            if label != "O":
                assert index_label in indexes_labels

def test_apply_ner_with_labels(model_id=3, training_data_id=2, size_test=100):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(training_data_id)
    sentence_tokens, sentence_labels =data_registry.split_training_data_sentences(rows)
    ner_results, tokens =ff.apply_ner(sentence_tokens)
    assert sentence_tokens == tokens
    tokens, predicted_labels, metrics = ff.convert_ner_results((ner_results,tokens),sentence_tokens,sentence_labels)
    expected_metrics = {'f1','recall','precision','accuracy'}
    assert expected_metrics.issubset(metrics.keys())

def test_prepare_training_data(model_id=3,training_data_id=2,size_test=100):
    ff = FlairFramework()
    ff.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(training_data_id)[100:100+size_test]
    corpus, label_dict =ff.prepare_training_data(rows)
    #check amount labels
    amount_corpus_labels = 0
    for sen in corpus.train:
        if len(sen.annotation_layers) == 1:
            amount_corpus_labels += len(sen.annotation_layers["ner"])
    for sen in corpus.test:
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

def test_finetune_model(model_id=3,training_data_id=1):
    ff = FlairFramework()
    base_model = model_registry.list_model(model_id)
    ff.load_model(base_model)
    rows = data_registry.load_training_data(training_data_id)
    corpus, label_dict = ff.prepare_training_data(rows)
    ff.finetune_ner_model(base_model.storage_path, corpus, label_dict,"FlairFirstTry",MODIFIED_MODELS_PATH)


def test_train_test_split(train_size=0.8, valid_size=0.1,test_size=0.1):
    test_data = [1]*int(100*train_size) + [2]*int(100*(valid_size+test_size))
    train, valid, test = train_test_split(test_data,train_size=train_size,valid_size=valid_size,test_size=test_size)
    assert len(train) + len(valid) + len(test) == len(test_data)
    assert len(train) == (100 * train_size)
    # test shuffle
    assert 2 in train




