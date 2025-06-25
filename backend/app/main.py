from flask import Flask

from app.model.framework_provider.framework import FrameworkNames
from model.ner_model_provider.model_registry import ModelRegistry
from model.ner_model_provider.ner_model import NERModel
from model.framework_provider.huggingface_framework import HuggingFaceFramework
from model.data_provider.data_registry import DataRegistry

def run_test():
    store_path = "app/store/"
    trainingsdata_path = store_path+"Trainingsdata/"
    trainingsdata_converted_path = trainingsdata_path+ "Converted/"
    trainingsdata_name = "adg1220.csv"
    model_name = "mschiesser/ner-bert-german"
    model_name_save = "mschiesser_ner-bert-german"
    models_path = store_path+"NER-Models/"
    base_models_path= models_path + "base/"
    models_modified_path = models_path +"modified/"
    first_modiefied_model = models_modified_path +"ner-first-ty/checkpoint-138/"
    path_model_metadata = models_path+"models_metadata.json"

    #test_model = NERModel(1,model_name,FrameworkNames["HUGGINGFACE"],model_name,base_models_path+model_name)
    model_registry = ModelRegistry(path_model_metadata,2)
    '''
    #model_registry.add_model(test_model)
    model_registry.list_models()
    '''

    ''' apply NER Model
    hf = HuggingFaceFramework()
    data_registry = DataRegistry()
    hf.load_model(model_registry.current_model)
    test_data = "Wir sind, die erste Station war, glaub ich, Magolsheim, hieß es, genau. Das ist hinten bei Ulm, so Leipheim, Biberach, Magolsheim.  Kleines, verschlafenes Dörfchen. Da sagt sich Fuchs und Hase, ""Grüß Gott und guten Morgen"" natürlich [lacht]. Und ja, dort, äh  die ersten Tage so Kindergarten schnell dran gewöhnt. Und irgendwann (mal?) hat ja auch alles Spaß gemacht. Und wir hatten ein riesen Bauernhaus noch und ganz viele Kammern da drin und Räume dadrin. Und jeden Tag konntest du da was anderes entdecken, und ja, war schön, muss ich sagen, also so das erste Jahr."
    hf.apply_ner(test_data)
    '''

    # Trainingsdaten speichern
    #data_registry.saveTrainingData(trainingsdata_path+trainingsdata_name)

    # Modell feinanpassen
    new_model_name = "NER-Second-Try"
    data_registry = DataRegistry()
    hf = HuggingFaceFramework()
    rows = data_registry.loadTrainingData((trainingsdata_converted_path+trainingsdata_name).replace(".csv",".json"))
    #vielleicht das erstellen das models auch in Registry verstecken
    modified_model_id = modified_model = NERModel(3, new_model_name, FrameworkNames["HUGGINGFACE"], model_name, models_modified_path + new_model_name)
    model_registry.add_model(modified_model)
    data, labelid = hf.prepare_training_data(rows,base_models_path+model_name_save)
    results, args = hf.finetune_ner_model(base_models_path+model_name_save,data,labelid,new_model_name,models_modified_path)
    model_registry.add_training(modified_model_id, trainingsdata_name, 2,results, args)



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