from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider import spacy_framework
from app.model.framework_provider.framework import FrameworkNames
from app.model.framework_provider.spacy_framework import SpacyFramework
from app.model.ner_model_provider.model_registry import model_registry
from app.model.ner_model_provider.ner_model import NERModel
from app.utils.config import BASE_MODELS_PATH


def test_prepare_training_data(training_data_id=1):
    rows = data_registry.load_training_data(training_data_id)
    sf = SpacyFramework()
    sf.prepare_training_data(rows)

def create_get_dummy_model(model_id):
    dummy = model_registry.list_model(model_id)
    if dummy == None:
        new_model_id = model_registry.add_model(NERModel(1, "SpacyDummy",FrameworkNames.SPACY,"de_core_new_sm",BASE_MODELS_PATH+"/NLP/de_core_news_sm"))
        return new_model_id
    else:
        return model_id

def test_load_apply_dummy_model(model_id=7,dataset_id=2, test_size=100):
    model_id = create_get_dummy_model(model_id)
    sf = spacy_framework.SpacyFramework()
    sf.load_model(model_registry.list_model(model_id))
    rows =data_registry.load_training_data(dataset_id)[200:200+test_size]
    texts = [row.text for row in rows]
    sf.apply_ner(texts)

