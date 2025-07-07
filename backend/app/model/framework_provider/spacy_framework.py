import spacy
from spacy.tokens import DocBin

from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel
from app.utils.config import SPACY_TRAININGSDATA_PATH

'''
    Notizen hierzu:
        kann Transformer Modell nur als Encoder einfügen oder eine eigene Komponente
        - eine Komponente ist nur ein Wrapper für die transformer Libary
        
'''
class SpacyFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.SPACY:
            raise TypeError("Expects an model for Spacy")
        self.ner_model = spacy.load(model.storage_path)

    def apply_ner(self, texts):
        for doc in self.ner_model.pipe(texts):
            print(doc.text)
            print([(ent.text,ent.start, ent.label_) for ent in doc.ents])

    def prepare_training_data(self, rows, tokenizer_path=None, train_size=0.8, validation_size=0.1, test_size=0.1,
                              split_sentences=False):
        train, valid, test  = self._train_test_split(rows, train_size, validation_size, test_size)
        self._bio_to_spacy(train,SPACY_TRAININGSDATA_PATH+"/train.spacy")
        self._bio_to_spacy(valid,SPACY_TRAININGSDATA_PATH+"/valid.spacy")

    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        pass

    def convert_ner_results(self, ner_results, ner_input, annoted_labels=None):
        pass

    @property
    def default_finetuning_params(self):
        pass

    def _bio_to_spacy(self, rows, output_path, lang="de"):
        nlp = spacy.blank(lang)
        doc_bin = DocBin()

        for row in rows:
            doc = self._create_doc(nlp,row.tokens, row.labels)
            doc_bin.add(doc)

        doc_bin.to_disk(output_path)

    def _create_doc(self, nlp, tokens, labels):
        doc = nlp.make_doc(" ".join(tokens))
        ents = []
        start = 0
        current_ent = None
        for token, label in zip(doc, labels):
            end = start + len(token.text)
            if label.startswith("B-"):
                if current_ent:
                    ents.append(current_ent)
                current_ent = (start, end, label[2:])
            elif label.startswith("I-") and current_ent:
                current_ent = (current_ent[0], end, current_ent[2])
            else:
                if current_ent:
                    ents.append(current_ent)
                    current_ent = None
            start = end + 1  # include space
        if current_ent:
            ents.append(current_ent)
        doc.ents = [doc.char_span(start, end, label=label) for start, end, label in ents if
                    doc.char_span(start, end, label=label)]
        return doc