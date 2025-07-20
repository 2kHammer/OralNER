from app.model.ner_model_provider.model_registry import model_registry

# --------------------------------------
# public functions
# --------------------------------------
def get_models():
    """
    Returns the model metadata

    Returns
    (List[dict]): with the model metadata in the `NERModel` format
    """
    models = model_registry.list_models()
    return [m.to_dict() for m in models]

def get_model(id):
    """
    Returns the model with `id`

    Parameters
    id (int): id of the model

    Returns
    (dict | None): the NERModel as dict or None
    """
    model = model_registry.list_model(id)
    if model is None:
        return None
    else:
        return model.to_dict()

def get_model_active():
    """
    Returns the active model
    """
    return model_registry.current_model.to_dict()

def set_model_active(id):
    """
    Sets the model with `id` as active

    Parameters
    id (int): id of the model

    Returns
    (bool): True if the model is active
    """
    if model_registry.set_current_model(id):
        return True
    else:
        return False
