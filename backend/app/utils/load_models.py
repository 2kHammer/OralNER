'''
So können die jeweiligen Modelle in den zugehörigen Ort gespeichert werden
model_name = "mschiesser/ner-bert-german"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    path =BASE_MODELS_PATH + "/mschiesser_ner-bert-german"
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
'''
import os

import spacy
from spacy.cli import download
from transformers import AutoTokenizer, AutoModelForTokenClassification

from app.model.framework_provider.framework import FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel
from app.utils.config import STORE_PATH, MODELS_PATH, BASE_MODELS_PATH, MODIFIED_MODELS_PATH, TRAININGSDATA_PATH, \
    TRAININGSDATA_CONVERTED_PATH, CONLL_PATH, SPACY_TRAININGSDATA_PATH, STORE_TEMP_PATH


def init_store_models():
    """
    Creates the store and loads the default models
    """
    # create path
    if not os.path.exists(STORE_PATH):
        os.makedirs(STORE_PATH)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    if not os.path.exists(BASE_MODELS_PATH):
        os.makedirs(BASE_MODELS_PATH)
    if not os.path.exists(MODIFIED_MODELS_PATH):
        os.makedirs(MODIFIED_MODELS_PATH)
    if not os.path.exists(TRAININGSDATA_PATH):
        os.makedirs(TRAININGSDATA_PATH)
    if not os.path.exists(TRAININGSDATA_CONVERTED_PATH):
        os.makedirs(TRAININGSDATA_CONVERTED_PATH)
    if not os.path.exists(CONLL_PATH):
        os.makedirs(CONLL_PATH)
    if not os.path.exists(SPACY_TRAININGSDATA_PATH):
        os.makedirs(SPACY_TRAININGSDATA_PATH)

    models_for_metadata = []
    model_spacy_default =check_load_default_spacy(STORE_TEMP_PATH)
    if model_spacy_default is not None:
        models_for_metadata.append(model_spacy_default)
        
    model_hf_default = check_load_default_huggingface(STORE_TEMP_PATH)
    if model_hf_default is not None:
        models_for_metadata.append(model_hf_default)

    print(len(models_for_metadata))




def check_load_default_spacy(base_model_path):
    spacy_default = "de_core_news_md"
    spacy_default_path = base_model_path + "/" + spacy_default
    if not os.path.exists(spacy_default_path):
        download(spacy_default)
        nlp = spacy.load(spacy_default)
        nlp.to_disk(spacy_default_path)
        return NERModel(1,spacy_default,FrameworkNames.SPACY,spacy_default,spacy_default_path)
    else:
        return None
    
def check_load_default_huggingface(base_model_path):
    huggingface_default = "mschiesser/ner-bert-german"
    huggingface_default_path = base_model_path+"/mschiesser_ner-bert-german"

    if not os.path.exists(huggingface_default_path):
        tok = AutoTokenizer.from_pretrained(huggingface_default)
        model = AutoModelForTokenClassification.from_pretrained(huggingface_default)

        tok.save_pretrained(huggingface_default_path)
        model.save_pretrained(huggingface_default_path)
        return NERModel(2,huggingface_default,FrameworkNames.HUGGINGFACE,huggingface_default,huggingface_default_path)
    else:
        return None




'''
tagger = SequenceTagger.load("flair/ner-german")

# Modell lokal speichern
# Oberverzeichnis muss zuerst erstellt werden
tagger.save(BASE_MODELS_PATH+"/flair_ner-german/flair_ner-german.pt")
'''

'''
Muss die Base Models auch zur Registry hinzufügen
def test_add_to_registry():
    model_registry.add_model(NERModel(1, "flair/ner-german", FrameworkNames.FLAIR, "flair/ner-german",
                                      BASE_MODELS_PATH + "/flair_ner-german/flair_ner-german.pt"))
'''

'''
Muss die Datensätze auch laden
'''