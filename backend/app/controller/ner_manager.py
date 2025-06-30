from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.model.ner_model_provider.model_registry import model_registry

def ner(content, with_file):
    framework_name = model_registry.current_model.framework_name
    framework = None
    if framework_name.name == 'HUGGINGFACE':
        framework = HuggingFaceFramework()

    framework.load_model(model_registry.current_model)
    sentences = None
    adg_rows = None
    if with_file:
        adg_rows = data_registry.prepare_data_with_labels(content)
        sentences = [to.text for to in adg_rows]
    else:
        sentences = data_registry.prepare_data_without_labels(content)
    entities =framework.apply_ner(sentences)
    tokens = None
    labels = None
    metrics = None
    if with_file:
        tokens,labels,metrics = framework.convert_ner_results(entities, adg_rows)
    else:
        tokens, labels, metrics = framework.convert_ner_results(entities, sentences)
    return tokens, labels, metrics


