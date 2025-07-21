import io
from unittest.mock import patch

from flask import Flask
import pytest

from app.model.framework_provider.framework import FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel
from app.service import app_router
from app.service.app_router import api


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api)
    app.testing = True
    return app.test_client()


# --------------------------------------
# unit tests
# --------------------------------------
def test_get_models(client):
    test_model = NERModel(1, "test",FrameworkNames.SPACY, "base","/path")
    test_data = [test_model.to_dict()]
    with patch("app.service.app_router.model_manager.get_models", return_value=test_data):
        res =client.get("/models")
        assert res.status == "200 OK"
    with patch("app.service.app_router.model_manager.get_models", side_effect=Exception):
        res = client.get("/models")
        assert res.status == "500 INTERNAL SERVER ERROR"

def test_get_model(client):
    test_model = NERModel(1, "test",FrameworkNames.SPACY, "base","/path")
    with patch("app.service.app_router.model_manager.get_model", return_value=None):
        res = client.get("/models/1")
        assert res.status == "404 NOT FOUND"
    with patch("app.service.app_router.model_manager.get_model", return_value=test_model.to_dict()):
        res = client.get("/models/2")
        assert res.status == "200 OK"

def test_set_active_model(client):
    with patch("app.service.app_router.model_manager.set_model_active", return_value=True):
        res = client.put("/models/active/4")
        assert res.status == "204 NO CONTENT"
    with patch("app.service.app_router.model_manager.set_model_active", return_value=False):
        res = client.put("/models/active/4")
        assert res.status == "404 NOT FOUND"

def test_apply_ner(client):
    # no params
    res =client.post("/ner")
    assert res.status == "400 BAD REQUEST"

    #no text
    res =client.post("/ner", json={"notext":"notext"})
    assert res.status == "400 BAD REQUEST"
    #text
    with patch("app.service.app_router.ner_manager.start_ner", return_value="2"):
        res = client.post("/ner", json={"text": "text"})
        assert res.status == "200 OK"

    # no adg file
    data = {
        'file': (io.BytesIO(b"Test file content"), 'test.txt')
    }
    response = client.post('/ner', data=data, content_type='multipart/form-data')
    assert response.status == "422 UNPROCESSABLE ENTITY"

def test_get_ner_job_result(client):
    with patch("app.service.app_router.ner_manager.get_ner_results", return_value=None):
        res = client.get("/ner/1")
        assert res.status == "202 ACCEPTED"
    with patch("app.service.app_router.ner_manager.get_ner_results", return_value="error"):
        res = client.get("/ner/1")
        assert res.status == "500 INTERNAL SERVER ERROR"
    with patch("app.service.app_router.ner_manager.get_ner_results", return_value=(["token"],["O"],None)):
        res = client.get("/ner/1")
        assert res.status == "200 OK"
    with patch("app.service.app_router.ner_manager.get_ner_results", side_effect=KeyError):
        res = client.get("/ner/1")
        assert res.status == "404 NOT FOUND"

def test_finetune(client):
    # no body
    res = client.post("/ner/finetune")
    assert res.status == "400 BAD REQUEST"
    # missing split sentences
    res = client.post("/ner/finetune",json={"model_id":0,"dataset_id":1,"parameters":{"new_model_name":"test"}})
    assert res.status == "400 BAD REQUEST"
    # all parameters
    with patch("app.service.app_router.ner_manager.finetune_ner", return_value="1"):
        res = client.post("/ner/finetune",json={"model_id":0,"dataset_id":1,"parameters":{"new_model_name":"test", "split_sentences":True}})
        assert res.status == "202 ACCEPTED"

def test_add_trainingsdata(client):
    res = client.post("/trainingdata")
    assert res.status == "400 BAD REQUEST"

    # no adg file
    data = {
        'file': (io.BytesIO("Test file content".encode("utf-8")), 'test.txt')
    }
    res = client.post('/trainingdata', data=data, content_type='multipart/form-data')
    assert res.status == "422 UNPROCESSABLE ENTITY"

    data2 = {
        'file': (io.BytesIO("Test file content".encode("utf-8")), 'test.txt')
    }
    data2["dataset_name"] ="test_ds"
    with patch("app.service.app_router.data_manager.add_training_data", return_value=True):
        res = client.post('/trainingdata', data=data2, content_type='multipart/form-data')
        assert res.status == "201 CREATED"

