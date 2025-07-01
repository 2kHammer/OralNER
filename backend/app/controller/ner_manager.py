from threading import Thread

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

def finetune_ner(model_id, dataset_id, new_model_name):
    base_model = model_registry.list_model(model_id)

    modified_model = model_registry.create_modified_model(new_model_name, base_model)
    modified_model_id = model_registry.add_model(modified_model)

    def worker():
        framework = None
        if base_model.framework_name.name == 'HUGGINGFACE':
            framework = HuggingFaceFramework()
        training_dataset_rows = data_registry.load_training_data(dataset_id)
        data, label_id = framework.prepare_training_data(training_dataset_rows,base_model.storage_path)
        results, args = framework.finetune_ner_model(base_model.storage_path,data,label_id,new_model_name,modified_model.storage_path)
        model_registry.add_training(modified_model_id, data_registry.get_training_data_name(dataset_id), dataset_id, results, args)

    thread = Thread(target=worker)
    thread.start()

    return modified_model_id


