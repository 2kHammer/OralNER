from .framework import Framework
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ..ner_model_provider.ner_model import NERModel

class HuggingFaceFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None
        self.tokenizer = None

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        self.ner_model = model
        self.model = AutoModelForTokenClassification.from_pretrained(model.storage_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model.storage_path)
        print("Model is loaded")
        
    def apply_ner(self, text):
        nlp = pipeline("ner",model=self.model,tokenizer=self.tokenizer, aggregation_strategy="simple")
        print(nlp(text))
        