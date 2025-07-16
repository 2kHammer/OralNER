from app.model.framework_provider.framework import FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.helpers import get_current_datetime


def test_model_to_dict():
    name = "test_model"
    base_model = "test_base_model"
    test_path = "/"
    model = NERModel(1, "test_model",FrameworkNames.SPACY,base_model, test_path)
    current_date = get_current_datetime()
    dataset = "test_dataset"
    dataset_id = 0
    results = TrainingResults(0.99,0.99,0.99,0.01, 0.01)
    test_args = {"arg1":0,"arg2":0}
    model.append_training(current_date, dataset, dataset_id, results,test_args)
    model_dic = model.to_dict()
    assert model_dic["id"] == None
    assert model_dic["storage_path"] == test_path
    assert isinstance(model_dic["trainings"], list)

    model2 = NERModel.from_dict(model_dic)
    assert model.base_model_name == model2.base_model_name
    assert model.trainings[-1]["date"] == model2.trainings[-1]["date"]