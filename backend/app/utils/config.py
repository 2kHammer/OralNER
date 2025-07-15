import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH  = os.path.abspath(os.path.join(CURRENT_DIR,"..","store"))
STORE_TEMP_PATH = os.path.abspath(os.path.join(STORE_PATH,"Temp"))
TRAININGSDATA_PATH = os.path.abspath(os.path.join(STORE_PATH,"Trainingsdata"))
TRAININGSDATA_CONVERTED_PATH = os.path.abspath(os.path.join(TRAININGSDATA_PATH,"Converted"))
MODELS_PATH = os.path.abspath(os.path.join(STORE_PATH,"NER-Models"))
BASE_MODELS_PATH = os.path.abspath(os.path.join(MODELS_PATH,"base"))
DEFAULT_TOKENIZER_PATH = os.path.abspath(os.path.join(BASE_MODELS_PATH,"NLP","de_core_news_sm"))
MODIFIED_MODELS_PATH = os.path.abspath(os.path.join(MODELS_PATH,"modified"))
MODEL_METADATA_PATH = os.path.abspath(os.path.join(MODELS_PATH,"models_metadata.json"))
TRAININGSDATA_METADATA_PATH = os.path.abspath(os.path.join(TRAININGSDATA_PATH,"trainingsdata_metadata.json"))
CONLL_PATH = os.path.abspath(os.path.join(TRAININGSDATA_PATH,"CoNLL"))
SPACY_TRAININGSDATA_PATH = os.path.abspath(os.path.join(TRAININGSDATA_PATH,"Spacy"))
'''
    model_name = "mschiesser/ner-bert-german"
    model_name_save = "mschiesser_ner-bert-german"
    base_models_path= models_path + "base/"
    models_modified_path = models_path +"modified/"
    first_modiefied_model = models_modified_path +"ner-first-ty/checkpoint-138/"
    path_model_metadata = models_path+"models_metadata.json"
'''