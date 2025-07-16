import os

from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.ner_model_provider.model_registry import ModelRegistry
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults, ModelState
from app.utils.config import STORE_PATH, STORE_TEMP_PATH

# -------------------------------------
# init & helpers
# -------------------------------------
test_path = STORE_TEMP_PATH +  "/models_metadata.json"
ModelRegistry._reset_instance()
def delete_test_metadata():
    if os.path.exists(test_path):
        print("Deleting test metadata")
        os.remove(test_path)

# -------------------------------------
# unit tests
# -------------------------------------
def test_init_metadata():
    delete_test_metadata()
    test_mr = ModelRegistry(test_path)
    test_model = NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH)
    id = test_mr.add_model(test_model)
    assert test_mr.current_model.id == id


def test_set_current_model_unique_names_and_ids():
    delete_test_metadata()
    test_mr = ModelRegistry(test_path)
    #check if no current model
    assert test_mr.current_model is None
    test_model = NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH)
    test_mr.add_model(test_model)
    # test if current model was set automatically
    assert test_mr.current_model.id == 0

    #test set current model
    assert test_mr.set_current_model(4) == False

    test_mr.add_model(NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH))
    test_mr.add_model(NERModel(3, "test", FrameworkNames["HUGGINGFACE"],"base_model2",STORE_PATH))
    test_mr.add_model(NERModel(2, "test_2", FrameworkNames["SPACY"],"base_model3",STORE_PATH+"/test"))
    test_mr.add_training(2, "test_dataset",0,TrainingResults(0.99,0.99,0.99,0.99,0.99),{"test":0})
    # test add training
    assert test_mr.list_model(2).state == ModelState.MODIFIED

    assert len(test_mr._models) == 4
    #test set current model
    assert test_mr.set_current_model(3) == True
    assert test_mr.current_model.name == "test_2"
    #test unique id and models
    assert len(set([model.id for model in test_mr._models])) ==4
    assert len(set([model.name for model in test_mr._models])) ==4

    # test list model
    assert test_mr.list_model(100) == None

def test_create_modified_model():
    delete_test_metadata()
    test_mr = ModelRegistry(test_path)
    test_model = NERModel(1, "test", FrameworkNames["FLAIR"],"base_model",STORE_PATH)
    test_mr.add_model(test_model)
    mod_model_return =test_mr.create_modified_model("mod_model", test_model)
    assert mod_model_return.state == ModelState.IN_TRAINING
    assert mod_model_return.base_model_name == test_model.name
    relPath= mod_model_return.storage_path.startswith("app/")
    assert relPath == True

def test_get_nextid():
    delete_test_metadata()
    test_mr = ModelRegistry(test_path)
    test_model = NERModel(1, "test", FrameworkNames["FLAIR"], "base_model", STORE_PATH)
    test_mr.add_model(test_model)
    assert test_mr._get_next_id() == 1
    test_mr.add_model(test_model)
    test_mr.add_model(test_model)
    test_mr._models.pop(1)
    assert test_mr._get_next_id() ==1




