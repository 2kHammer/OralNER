from flask import Flask
from flask_cors import CORS

from app.model.data_provider.data_registry import data_registry
from app.model.ner_model_provider.model_registry import model_registry
from app.utils.config import STORE_PATH, TRAININGSDATA_PATH
from service.app_router import api

from app.model.framework_provider.framework import FrameworkNames
from model.ner_model_provider.model_registry import ModelRegistry
from model.ner_model_provider.ner_model import NERModel
from app.model.framework_provider.huggingface_framework import HuggingFaceFramework
from app.model.data_provider.data_registry import data_registry

def run_test():
    

    #test_model = NERModel(1,model_name,FrameworkNames["HUGGINGFACE"],model_name,base_models_path+model_name)
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
    #new_model_name = "NER-Second-Try"
    #hf = HuggingFaceFramework()
    #rows = data_registry.load_training_data((trainingsdata_converted_path + trainingsdata_name).replace(".csv", ".json"))
    #vielleicht das erstellen das models auch in Registry verstecken
    #modified_model = NERModel(3, new_model_name, FrameworkNames["HUGGINGFACE"], model_name, models_modified_path + new_model_name)
    #modified_model_id = model_registry.add_model(modified_model)
    #data, labelid = hf.prepare_training_data(rows,base_models_path+model_name_save)
    #results, args = hf.finetune_ner_model(base_models_path+model_name_save,data,labelid,new_model_name,models_modified_path)
    #model_registry.add_training(modified_model_id, trainingsdata_name, 2,results, args)

    hf = HuggingFaceFramework()
    rows = data_registry.load_training_data(3)
    test_objs = rows[20:100]

    hf.load_model(model_registry.current_model)
    ner_res = hf.apply_ner([to.text for to in test_objs])
    test= hf.convert_ner_results(ner_res,[to.text for to in test_objs])
    print(test[1])


    '''
    Read in Dataset
    filename = "adg2983.csv"
    with open(TRAININGSDATA_PATH+"/"+filename,newline='') as csvfile:
        data_registry.add_training_data("adg2983",filename,csvfile)
    '''



    #app = Flask(__name__)

# Lade das deutsche Modell
#model_name = "mschiesser/ner-bert-german"
#model_name_save = "mschiesser_ner-bert-german"
#path_to_save = "backend/app/store/NER-Models/base/"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForTokenClassification.from_pretrained(model_name)
#tokenizer.save_pretrained(path_to_save + model_name_save)
#model.save_pretrained(path_to_save + model_name_save)

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(api)
    CORS(app)
    app.run(debug=True)
    #run_test()
    #app.run(debug=True)