from flask import Flask
from model.ner_model_provider.model_registry import ModelRegistry
from model.ner_model_provider.ner_model import NERModel
from model.framework_provider.huggingface_framework import HuggingFaceFramework
from model.data_provider.data_registry import DataRegistry

def run_test():
    trainingsdata_path = "app/store/Trainingsdata/"
    traingsdata_name = "adg1220.csv"
    model_name = "mschiesser/ner-bert-german"
    model_name_save = "mschiesser_ner-bert-german"
    path_to_save = "app/store/NER-Models/base/"

    test_model = NERModel(1,model_name,"huggingface",model_name,path_to_save+model_name_save)
    model_registry = ModelRegistry(test_model)
    hf = HuggingFaceFramework()
    data_registry = DataRegistry()

    #hf.load_model(model_registry.current_model)
    data_registry.loadTrainingData(trainingsdata_path+traingsdata_name)
    #text = "Angela Merkel war Bundeskanzlerin in Deutschland."
    #hf.apply_ner(text)

    #app = Flask(__name__)

    #@app.route('/')
    #def home():
    #    return "Hallo von Flask in main.py!"


# Lade das deutsche Modell
#model_name = "mschiesser/ner-bert-german"
#model_name_save = "mschiesser_ner-bert-german"
#path_to_save = "backend/app/store/NER-Models/base/"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForTokenClassification.from_pretrained(model_name)
#tokenizer.save_pretrained(path_to_save + model_name_save)
#model.save_pretrained(path_to_save + model_name_save)


#Anwenden des deutschen Modells
#token = AutoTokenizer.from_pretrained(path_to_save + model_name_save)
#model = AutoModelForTokenClassification.from_pretrained(path_to_save+ model_name_save)

#nlp = pipeline("ner",model=model,tokenizer=token, aggregation_strategy="simple")
#print(nlp(text))

#print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Keine GPU")


# Beispieltext
#text = "Angela Merkel wurde 2005 zur Bundeskanzlerin gewählt und war bis 2021 im Amt. Sie wurde in Hamburg geboren."

#nlp.to_bytes().to_file("store/ner")


# muss das Modell laden für NLP: python -m spacy download en_core_web_sm

if __name__ == '__main__':
    run_test()
    #app.run(debug=True)