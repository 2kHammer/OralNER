from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider import spacy_framework
from app.model.framework_provider.framework import FrameworkNames
from app.model.framework_provider.spacy_framework import SpacyFramework
from app.model.ner_model_provider.model_registry import model_registry
from app.model.ner_model_provider.ner_model import NERModel
from app.utils.config import BASE_MODELS_PATH


def test_prepare_training_data(training_data_id=3):
    rows = data_registry.load_training_data(training_data_id)
    sf = SpacyFramework()
    sf.prepare_training_data(rows, tokenizer_path=None, train_size=0.7, validation_size=0.3, test_size=0, split_sentences=False)

def create_get_dummy_model(model_id):
    dummy = model_registry.list_model(model_id)
    if dummy == None:
        new_model_id = model_registry.add_model(NERModel(1, "SpacyDummy",FrameworkNames.SPACY,"de_core_new_sm",BASE_MODELS_PATH+"/NLP/de_core_news_sm"))
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

def test_check_if_the_model_tokens_are_the_same_for_all_datasets():
    for i in range(0,4):
        test_load_apply_ner_bert_german_model(model_id=8, dataset_id=i)

def test_convert_ner_results_with_adg(model_id=8, dataset_id=2):
    model_id = create_get_ner_bert_german(model_id)
    sf = spacy_framework.SpacyFramework()
    sf.load_model(model_registry.list_model(model_id))
    rows =data_registry.load_training_data(dataset_id)
    texts = [row.text for row in rows]
    ner_results = sf.apply_ner(texts)
    tokens, pred_labels, metrics=sf.convert_ner_results(ner_results, rows)
    expected_metrics = {'f1', 'recall', 'precision', 'accuracy'}
    assert expected_metrics.issubset(metrics.keys())

def test_finetune(base_model_id=8):
    base_model = model_registry.list_model(base_model_id)
    modified_model =model_registry.create_modified_model("TestSpacy",base_model)
    sf = SpacyFramework()
    sf.finetune_ner_model(base_model.storage_path,None,None,modified_model.name,modified_model.storage_path,None)

def test_finetune_default_spacy(base_model_id=7):
    base_model = model_registry.list_model(base_model_id)
    modified_model =model_registry.create_modified_model("TestSpacyDummy",base_model)
    sf = SpacyFramework()
    metrics, args = sf.finetune_ner_model(base_model.storage_path,None,None,modified_model.name,modified_model.storage_path,None)
    assert metrics.f1 > 0
    assert metrics.precision > 0
    assert metrics.recall > 0
    assert metrics.duration > 0
    
def test_evaluate_finetune_spacy(model_id=8):
    #model = model_registry.list_model(model_id)
    ff = SpacyFramework()
    #metrics = ff._evaluate_transformer_model(ff._get_correct_model_path(model.storage_path))
    metrics = ff._evaluate_transformer_model("/home/alex/Nextcloud3/Alex/Uni/Master/SoSe25/CBIE/OralNER/backend/app/store/NER-Models/modified/spacy-ner-bert-german-second/model-best")
    print(metrics)
    assert metrics.f1 > 0
    assert metrics.precision > 0
    assert metrics.recall > 0
