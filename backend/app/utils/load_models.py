'''
So können die jeweiligen Modelle in den zugehörigen Ort gespeichert werden
model_name = "mschiesser/ner-bert-german"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    path =BASE_MODELS_PATH + "/mschiesser_ner-bert-german"
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
'''


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