from collections import Counter

from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.flair_framework import FlairFramework
from app.model.ner_model_provider.model_registry import model_registry

def run_pipeline_test(framework, training_data_id, model_id, size_test):
    rows = data_registry.load_training_data(training_data_id)[25:25 + size_test]
    model = model_registry.list_model(model_id)
    tokens, pred_labels, metrics = framework.process_ner_pipeline(model, rows)
    expected_metrics = {'f1', 'recall', 'precision', 'accuracy'}
    assert expected_metrics.issubset(metrics.keys())
    # check if the lengths matches
    for i, row in enumerate(rows):
        assert len(row.tokens) == len(tokens[i]) == len(pred_labels[i])
        assert row.tokens == tokens[i]
    # check if the model recognizes some entities
    assert metrics["f1"] > 0.1

    # with sentence split
    tokens_sen, pred_labels_sen, metrics_sen = framework.process_ner_pipeline(model, rows, True)
    expected_metrics = {'f1', 'recall', 'precision', 'accuracy'}
    assert expected_metrics.issubset(metrics_sen.keys())
    # check if the model recognizes some entities
    assert metrics["f1"] > 0.1

    # without adg
    texts = [row.text for row in rows]
    sent_not_adg = []
    for text in texts:
        sent_not_adg += data_registry.prepare_data_without_labels(text)
    tokens_not_adg, labels_not_adg, metrics_not_adg = framework.process_ner_pipeline(model, sent_not_adg, True)
    assert metrics_not_adg == None

def test_train_test_split(train_size=0.8, valid_size=0.1,test_size=0.1):
    ff = FlairFramework()
    test_data = [1]*int(100*train_size) + [2]*int(100*(valid_size+test_size))
    train, valid, test = ff._train_test_split(test_data,train_size=train_size,valid_size=valid_size,test_size=test_size)
    assert len(train) + len(valid) + len(test) == len(test_data)
    assert len(train) == (100 * train_size)
    # test shuffle
    assert 2 in train

def test_convert_ner_results_to_format():
    test = ["Ich Alexander Hammer komme aus Regensburg und bin am 19.05.2000 geboren, also püntklich zu meinem Geburtstag"]
    #result and tokens was generated with the default flair model
    results = [[{'end_token': 2, 'start_pos': 4, 'start_token': 1, 'text': 'Alexander Hammer', 'type': 'PER'}, {'end_token': 5, 'start_pos': 31, 'start_token': 5, 'text': 'Regensburg', 'type': 'LOC'}]]
    tokens =[['Ich', 'Alexander', 'Hammer', 'komme', 'aus', 'Regensburg', 'und', 'bin', 'am', '19.05.2000', 'geboren', ',', 'also', 'püntklich', 'zu', 'meinem', 'Geburtstag']]
    ff = FlairFramework()
    tokens_res, labels = ff._convert_ner_results_to_format((results,tokens))
    assert labels[0][results[0][0]["start_token"]] == "B-"+results[0][0]["type"]
    assert labels[0][results[0][0]["end_token"]] == "I-"+results[0][0]["type"]
    assert labels[0][results[0][1]["end_token"]] == "B-"+results[0][1]["type"]

    labels_amount = Counter(labels[0])
    assert labels_amount["O"] == (len(labels[0])-3)
