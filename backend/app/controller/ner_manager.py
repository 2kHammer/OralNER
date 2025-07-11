import threading
import uuid
from threading import Thread

from app.model.data_provider.data_registry import data_registry, simple_split_sentences
from app.model.framework_provider.flair_framework import FlairFramework
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.model.framework_provider.spacy_framework import SpacyFramework
from app.model.ner_model_provider.model_registry import model_registry

ner_jobs = {}
lock = threading.Lock()

def start_ner(content, with_file):
    framework_name = model_registry.current_model.framework_name
    framework = None
    if framework_name.name == 'HUGGINGFACE':
        framework = HuggingFaceFramework()
    elif framework_name.name == 'FLAIR':
        framework = FlairFramework()
    elif framework_name.name == 'SPACY':
        framework = SpacyFramework()
    framework.load_model(model_registry.current_model)

    job_id = str(uuid.uuid4())
    with lock:
        ner_jobs[job_id] = None

    thread = Thread(target=_ner_worker, args=(job_id,framework, framework_name, content, with_file))
    thread.start()
    return job_id

def get_ner_results(job_id):
    with lock:
        if job_id in ner_jobs:
            ner_result = ner_jobs[job_id]
            if ner_result is not None:
                ner_jobs.pop(job_id)
            return ner_result
        else:
            raise KeyError(f"Job {job_id} not found")



def finetune_ner(model_id, dataset_id, new_model_name):
    base_model = model_registry.list_model(model_id)

    modified_model = model_registry.create_modified_model(new_model_name, base_model)
    modified_model_id = model_registry.add_model(modified_model)

    def worker():
        framework = None
        if base_model.framework_name.name == 'HUGGINGFACE':
            framework = HuggingFaceFramework()
        elif base_model.framework_name.name == 'FLAIR':
            framework = FlairFramework()
        elif base_model.framework_name.name == 'SPACY':
            framework = SpacyFramework()
        training_dataset_rows = data_registry.load_training_data(dataset_id)
        data, label_id = framework.prepare_training_data(training_dataset_rows,base_model.storage_path)
        results, args = framework.finetune_ner_model(base_model.storage_path,data,label_id,new_model_name,modified_model.storage_path)
        model_registry.add_training(modified_model_id, data_registry.get_training_data_name(dataset_id), dataset_id, results, args)

    thread = Thread(target=worker)
    thread.start()

    return modified_model_id

def _ner_worker(job_id,framework, framework_name, content, with_file=False, use_sentences=False):
    sentences = None
    adg_rows = None
    if with_file:
        adg_rows = data_registry.prepare_data_with_labels(content)
        sentence_labels = None
        # flair needs tokens
        if framework_name.name == 'FLAIR':
            if use_sentences:
                sentences, sentence_labels = data_registry.split_data_with_labels(adg_rows)
            else:
                sentences = [row.tokens for row in adg_rows]
        else:
            if use_sentences:
                for row in adg_rows:
                    sentences_row, sentence_index_row = simple_split_sentences(row.text)
                    sentences += sentences_row
            else:
                sentences = [row.text for row in adg_rows]
    else:
        sentences = data_registry.prepare_data_without_labels(content)
    entities = framework.apply_ner(sentences)
    tokens = None
    labels = None
    metrics = None
    if with_file:
        # for flair - will be useful if data is splitted into sentences
        tokens, labels, metrics = framework.convert_ner_results(entities, adg_rows)
    else:
        tokens, labels, metrics = framework.convert_ner_results(entities, sentences)
    with lock:
        ner_jobs[job_id] = [tokens, labels, metrics]
