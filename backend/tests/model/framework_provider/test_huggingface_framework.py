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
    dataset, label_id = hf.prepare_training_data(rows, path_tokenizer, 0.8,0.2,split_sentences=split_sentences, seed=42)
    len_ds = len(rows)
    if split_sentences:
        sentence_data = data_registry.split_training_data_sentences(rows)
        len_ds = len(sentence_data)
    train_size = dataset["train"].num_rows
    vali_size = dataset["validation"].num_rows
    assert (train_size  + vali_size) == len_ds
    assert train_size/ len_ds >= 0.79 and train_size/ len_ds <= 0.91
    assert vali_size/ len_ds >= 0.19 and vali_size/ len_ds <= 0.21

def test_split_sizes():
    for i in range(1,2):
        check_split_size(i, True)



def test_ner_results_adg(dataset_id=0, model_id=1, test_size=100):
    hf.load_model(model_registry.list_model(model_id))
    rows = data_registry.load_training_data(dataset_id)
    rows = rows[100: 100+test_size]
    texts = [row.text for row in rows]
    ner_results = hf.apply_ner(texts)
    tokens, predicted_labels, metrics  = hf.convert_ner_results(ner_results,rows)

    for index, pred_labels in enumerate(predicted_labels):
        assert len(pred_labels) == len(rows[index].tokens)

    expected_metrics = {'f1', 'recall', 'precision', 'accuracy'}
    assert expected_metrics.issubset(metrics.keys())

def test_ner_pipeline_adg(dataset_id=3, model_id=1, test_size=100):
    hf = HuggingFaceFramework()
    model = model_registry.list_model(model_id)
    rows = data_registry.load_training_data(dataset_id)[100: 100+test_size]
    tokens_row, labels_row, metrics_row =hf.process_ner_pipeline(model, rows)
    tokens_sen, labels_sen, metrics_sen =hf.process_ner_pipeline(model, rows, True)
    all_tokens_sen= [token for tokens in tokens_sen for token in tokens]
    all_tokens_row = [token for tokens in tokens_row for token in tokens]
    # check if the tokens are in the correct order
    assert all_tokens_sen == all_tokens_row
    expected_metrics = {'f1', 'recall', 'precision', 'accuracy'}
    assert expected_metrics.issubset(metrics_row.keys())
    assert expected_metrics.issubset(metrics_sen.keys())

    #test without adg
    sentences = data_registry.split_training_data_sentences(rows)
    sen_text = [sen.text for sen in sentences]
    tokens_no_adg, labels_no_adg, _= hf.process_ner_pipeline(model, sen_text)

'''
    Integration Tests
'''
