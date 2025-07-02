'''
So können die jeweiligen Modelle in den zugehörigen Ort gespeichert werden
model_name = "mschiesser/ner-bert-german"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    path =BASE_MODELS_PATH + "/mschiesser_ner-bert-german"
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
'''