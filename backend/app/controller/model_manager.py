from app.model.ner_model_provider.model_registry import model_registry

def get_models():
    models = model_registry.list_models()
    return [m.to_dict() for m in models]

def get_model(id):
    model = model_registry.list_model(id)
    if model is None:
        return None
    else:
        return model.to_dict()

def get_model_active():
    return model_registry.current_model.to_dict()

def set_model_active(id):
    if model_registry.set_current_model(id):
        return True
    else:
        return False
