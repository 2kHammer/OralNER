import io
import os
import shutil
import time
from unittest.mock import patch

import pytest
from sympy.functions.elementary.exponential import ExpBase

from app.model.data_provider.data_registry import data_registry
from app.controller import ner_manager
from app.model.ner_model_provider.model_registry import model_registry
from app.model.ner_model_provider.ner_model import ModelState


test_model_id = 0
# --------------------------------------
# unit tests
# --------------------------------------

def test_start_ner_false_file():
    assert ner_manager.start_ner(["Ein beliebiger Eingabetext"],True, True) == "-1"
    fake_file = io.StringIO("Virtuelle; csv; \n Virtuelle2; csv2")
    assert ner_manager.start_ner(fake_file, True, True) == "-1"

def test_get_ner_results_false_job_id():
    with pytest.raises(KeyError):
        ner_manager.get_ner_results("-1")
# --------------------------------------
# integration tests
# --------------------------------------
def test_apply_ner(dataset_id = 2):
        path = data_registry._datasets[dataset_id].path
        path = path.replace(".json",".csv").replace("/Converted","")
        with open(path,"r") as f:
            file = f.read().splitlines()
            job_id = ner_manager.start_ner(file, True, True)
            assert job_id in ner_manager.ner_jobs

        results = None
        while not results:
            results = ner_manager.get_ner_results(job_id)
            time.sleep(5)
            print(results)

        print(results)
        assert len(results) == 3
        assert isinstance(results[0], list)

def test_finetune_ner(dataset_id = 3):
    # test the finetuning for spacy "de_core_news_sm" -> fastest finetune
    base_model_id = test_model_id
    mod_model_id = None
    mod_model = None
    try:
        mod_model_id =ner_manager.finetune_ner(base_model_id, dataset_id, "TestFinetune", False)
        mod_model = model_registry.list_model(mod_model_id)
        assert mod_model.state == ModelState.IN_TRAINING
        while mod_model.state == ModelState.IN_TRAINING:
            time.sleep(5)
            print("in training")
        assert mod_model.state == ModelState.MODIFIED
        # check if entities were recognized
        assert mod_model.trainings[0]["metrics"].f1 > 0.1
        assert mod_model.trainings[0]["metrics"].duration > 30
    finally:
        # delete the trained model
        ind =model_registry._get_index_model_id(mod_model_id)
        path = mod_model.storage_path
        if ind is not None:
            model_registry._models.pop(ind)
            model_registry._udpate_metadata()
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)

def test_finetune_error():
    base_model_id = test_model_id
    mod_model_id = None
    try:
        with patch("app.controller.ner_manager.data_registry.load_training_data", return_value=None):
            mod_model_id = ner_manager.finetune_ner(base_model_id, 2, "TestFinetune", False)
            mod_model = model_registry.list_model(mod_model_id)
            time.sleep(10)
            assert mod_model.state == ModelState.ERROR
    finally:
        # delete the trained model
        ind =model_registry._get_index_model_id(mod_model_id)
        if ind is not None:
            model_registry._models.pop(ind)
            model_registry._udpate_metadata()
