import threading
import uuid
from threading import Thread

from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.flair_framework import FlairFramework
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.model.framework_provider.spacy_framework import SpacyFramework
from app.model.ner_model_provider.model_registry import model_registry

ner_jobs = {}
lock = threading.Lock()

def start_ner(content, with_file, use_sentences=False):
    """
    Starts the NER-job with ´ner_worker()´ and save the job-id in ´ner_jobs´

    Parameters
    content (List[str] | str): the content on which NER should be applied
    with_file (bool): should NER be applied on an adg-file, is set from app_router.apply_ner
    use_sentences: if it should be applied on an adg-file, should the statements be split into sentences

    Returns
    (str): the job id
    """
    framework_name = model_registry.current_model.framework_name
    framework = None
    if framework_name.name == 'HUGGINGFACE':
        framework = HuggingFaceFramework()
    elif framework_name.name == 'FLAIR':
        framework = FlairFramework()
    elif framework_name.name == 'SPACY':
        framework = SpacyFramework()

    #check if file has the correct format
    if with_file:
        correct_file_format =data_registry.check_convert_adg_file(content)
        if not correct_file_format:
            return "-1"

    job_id = str(uuid.uuid4())
    with lock:
        ner_jobs[job_id] = None

    thread = Thread(target=_ner_worker, args=(job_id,framework, model_registry.current_model, content, with_file, use_sentences))
    thread.start()
    return job_id

def get_ner_results(job_id):
    """
    Check the status of a NER job.

    Parameters:
    job_id (str): the job id of the job to check

    Returns:
    (List[List,List,dict] | str | None): Return the tokens,labels and metrics if the job is finished, the error message if an error occured, or None if it is not finished
    """
    with lock:
        if job_id in ner_jobs:
            ner_result = ner_jobs[job_id]
            if ner_result is not None:
                ner_jobs.pop(job_id)
            return ner_result
        else:
            raise KeyError(f"Job {job_id} not found")



def finetune_ner(model_id, dataset_id, new_model_name, split_sentences):
    """
    Finetunes the model with `model_id` with the dataset with `dataset_id`

    Parameters:
    model_id (int): the id of the model to finetune
    dataset_id (int): the id of the dataset to finetune
    new_model_name (str): the name of the new model
    split_sentences (bool): whether to split sentences into training and test sets
    """
    base_model = model_registry.list_model(model_id)

    modified_model = model_registry.create_modified_model(new_model_name, base_model)
    modified_model_id = model_registry.add_model(modified_model)


    thread = Thread(target=_finetune_worker, args=(base_model,dataset_id,modified_model, split_sentences))
    thread.start()

    return modified_model_id

def _ner_worker(job_id,framework, model, content, with_file, use_sentences):
    """
    Applies NER in a separate thread

    Parameter:
    job_id (str): the results should be saved underneath
    framework (Framework): a derived Framework object
    model (NERModel): on which NER should be applied
    content (List[str] | str): the content on which NER should be applied
    with_file (bool):
    use_sentences (bool):
    """
    try:
        ner_input = None
        sentences = None
        if with_file:
            ner_input = data_registry.prepare_data_with_labels(content)
            if use_sentences:
                sentences = data_registry.split_training_data_sentences(ner_input)
        else:
            ner_input = data_registry.prepare_data_without_labels(content)
        tokens, labels, metrics = framework.process_ner_pipeline(model, ner_input, sentences)
        with lock:
            ner_jobs[job_id] = [tokens, labels, metrics]
    except Exception as e:
        with lock:
            ner_jobs[job_id] = str(e)

def _finetune_worker(base_model, dataset_id, modified_model, split_sentences):
    """
    Finetunes the `base_model` in an separate thread

    Parameters:
    base_model (NERModel): the model to finetune
    dataset_id (int): the id of the dataset to finetune
    modified_model (NERModel): the modified model, which will be finetuned
    split_sentences (bool)
    """
    try:
        new_model_name = modified_model.name
        modified_model_id = modified_model.id
        framework = None
        if base_model.framework_name.name == 'HUGGINGFACE':
            framework = HuggingFaceFramework()
        elif base_model.framework_name.name == 'FLAIR':
            framework = FlairFramework()
        elif base_model.framework_name.name == 'SPACY':
            framework = SpacyFramework()
        training_dataset_rows = data_registry.load_training_data(dataset_id)
        data, label_id = framework.prepare_training_data(training_dataset_rows,base_model.storage_path, split_sentences=split_sentences)
        results, args = framework.finetune_ner_model(base_model.storage_path,data,label_id,new_model_name,modified_model.storage_path)
        args['split_sentences'] = split_sentences
        model_registry.add_training(modified_model_id, data_registry.get_training_data_name(dataset_id), dataset_id, results, args)
    except Exception as e:
        model_registry.abort_finetuning(modified_model.id)
