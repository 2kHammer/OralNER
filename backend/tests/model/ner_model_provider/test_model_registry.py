import os

from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.ner_model_provider.model_registry import ModelRegistry
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.config import STORE_PATH, STORE_TEMP_PATH

test_path = STORE_TEMP_PATH +  "/metadata.json"
ModelRegistry._reset_instance()
def delete_test_metadata():
    if os.path.exists(test_path):
        print("Deleting test metadata")
        os.remove(test_path)

def test_init_metadata():
    test_mr = ModelRegistry(test_path)
    test_model = NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH)
    id = test_mr.add_model(test_model)
    assert test_mr.current_model.id == id


def test_set_current_model_unique_names_and_ids():
    test_mr = ModelRegistry(test_path)
    test_model = NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH)
    test_mr.add_model(test_model)
    assert test_mr.set_current_model(4) == False
    test_mr.add_model(NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH))
    test_mr.add_model(NERModel(3, "test", FrameworkNames["HUGGINGFACE"],"base_model2",STORE_PATH))
    test_mr.add_model(NERModel(2, "test", FrameworkNames["SPACY"],"base_model3",STORE_PATH+"/test"))
    test_mr.add_training(2, "test_dataset",0,TrainingResults(0.99,0.99,0.99,0.99,0.99),{"test":0})
    assert len(test_mr._models) == 4
    assert len(set([model.id for model in test_mr._models])) ==4
    assert len(set([model.name for model in test_mr._models])) ==4
    delete_test_metadata()

