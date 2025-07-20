from app.controller import model_manager


# --------------------------------------
# unit tests
# --------------------------------------
def test_get_models():
    models = model_manager.get_models()
    assert isinstance(models, list)
    if len(models) > 0:
        assert isinstance(models[0], dict)
        assert "state" in models[0].keys()
        assert "base_model_name" in models[0].keys()
        assert "trainings" in models[0].keys()

def test_get_model():
    assert isinstance(model_manager.get_model(0),dict)
    assert model_manager.get_model(100) is None

def test_active_model():
    default_model = model_manager.get_model_active()
    assert isinstance(default_model,dict)
    id_default = default_model["id"]
    model_manager.set_model_active(id_default+1)
    assert model_manager.get_model_active()["id"] == id_default+1

