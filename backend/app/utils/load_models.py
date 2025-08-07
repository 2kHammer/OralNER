import os

import spacy
from flair.models import SequenceTagger
from spacy.cli import download
from transformers import AutoTokenizer, AutoModelForTokenClassification

from app.model.framework_provider.framework import FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel
from app.utils.config import STORE_PATH, MODELS_PATH, BASE_MODELS_PATH, MODIFIED_MODELS_PATH, TRAININGSDATA_PATH, \
    TRAININGSDATA_CONVERTED_PATH, CONLL_PATH, SPACY_TRAININGSDATA_PATH, STORE_TEMP_PATH


def init_store_models():
    """
    Creates the store and loads the default models

    Returns
    (List[NERModel]): list of the downloaded NER-Models
    """
    mod_model_was_created = False
    # create path
    if not os.path.exists(STORE_PATH):
        os.makedirs(STORE_PATH)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    if not os.path.exists(BASE_MODELS_PATH):
        os.makedirs(BASE_MODELS_PATH)
    if not os.path.exists(MODIFIED_MODELS_PATH):
        os.makedirs(MODIFIED_MODELS_PATH)
        mod_model_was_created = True
    if not os.path.exists(TRAININGSDATA_PATH):
        os.makedirs(TRAININGSDATA_PATH)
    if not os.path.exists(TRAININGSDATA_CONVERTED_PATH):
        os.makedirs(TRAININGSDATA_CONVERTED_PATH)
    if not os.path.exists(CONLL_PATH):
        os.makedirs(CONLL_PATH)
    if not os.path.exists(SPACY_TRAININGSDATA_PATH):
        os.makedirs(SPACY_TRAININGSDATA_PATH)

    models_for_metadata = []
    
    model_spacy_transformer_base = check_load_default_spacy_transformer(BASE_MODELS_PATH, mod_model_was_created)
    if model_spacy_transformer_base is not None:
        models_for_metadata.append(model_spacy_transformer_base)
    
    model_spacy_default =check_load_default_spacy(BASE_MODELS_PATH)
    if model_spacy_default is not None:
        models_for_metadata.append(model_spacy_default)
        
    model_hf_default = check_load_default_huggingface(BASE_MODELS_PATH)
    if model_hf_default is not None:
        models_for_metadata.append(model_hf_default)

    model_fl_default = check_load_default_flair(BASE_MODELS_PATH)
    if model_fl_default is not None:
        models_for_metadata.append(model_fl_default)

    return models_for_metadata



def check_load_default_spacy(base_model_path):
    """
    Checks if the simple spacy model is in `base_model_path`

    Parameters
    base_model_path (str): the path where the base models are stored

    Returns
    (NERModel): if an model was loaded and saved at `base_model_path`
    """
    spacy_default = "de_core_news_sm"
    spacy_default_path = base_model_path + "/" + spacy_default
    if not os.path.exists(spacy_default_path):
        download(spacy_default)
        nlp = spacy.load(spacy_default)
        nlp.to_disk(spacy_default_path)
        return NERModel(1,spacy_default,FrameworkNames.SPACY,spacy_default,spacy_default_path)
    else:
        return None
    
def check_load_default_huggingface(base_model_path):
    """
    Checks if the default huggingface model is in `base_model_path`

    Parameters
    base_model_path (str): the path where the base models are stored

    Returns
    (NERModel): if an model was loaded and saved at `base_model_path`
    """
    huggingface_default = "mschiesser/ner-bert-german"
    huggingface_default_path = base_model_path+"/mschiesser_ner-bert-german"

    if not os.path.exists(huggingface_default_path):
        tok = AutoTokenizer.from_pretrained(huggingface_default)
        model = AutoModelForTokenClassification.from_pretrained(huggingface_default)

        tok.save_pretrained(huggingface_default_path)
        model.save_pretrained(huggingface_default_path)
        return NERModel(1,huggingface_default,FrameworkNames.HUGGINGFACE,huggingface_default,huggingface_default_path)
    else:
        return None

def check_load_default_flair(base_model_path):
    """
    Checks if the default flair model is in `base_model_path`

    Parameters
    base_model_path (str): the path where the base models are stored

    Returns
    (NERModel): if an model was loaded and saved at `base_model_path`
    """
    flair_default = "flair/ner-german"
    flair_default_path = base_model_path + "/flair_ner-german"

    if not os.path.exists(flair_default_path):
        os.makedirs(flair_default_path)
        tagger = SequenceTagger.load(flair_default)
        tagger.save(flair_default_path+ "/flair_ner_german.pt")
        return NERModel(1,flair_default,FrameworkNames.FLAIR,flair_default,flair_default_path)
    else:
        return None
    
def check_load_default_spacy_transformer(base_model_path, mod_models_created):
    """
    Check if default spacy transformer config file is in a under dic of the `base_model_path`

    Parameters
    base_model_path (str): the path where the base models are stored
    
    Returns 
    (NERModel): returns the NER-Model,if the config file for spacy transformer is in an under dic of`base_model_path`
    """
    base_transformer_path = None
    if mod_models_created:
        for d in os.listdir(base_model_path):
            full_path = os.path.join(base_model_path, d)
            if os.path.isdir(full_path):
                if os.path.exists(full_path+"/config.cfg"):
                    base_transformer_path = full_path

    if base_transformer_path is not None:
        return NERModel(5, "spacy_bert_base", FrameworkNames.SPACY, "mschiesser/ner-bert-german", base_transformer_path)
    else:
        return None




