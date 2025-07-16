from collections import Counter

from app.model.framework_provider.flair_framework import FlairFramework
from app.model.ner_model_provider.model_registry import model_registry


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
