import os

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import load_config
from spacy.cli.train import train
from typing_extensions import override

from app.model.data_provider.adg_row import ADGRow
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

    @property
    def default_finetuning_params(self):
        pass

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.SPACY:
            raise TypeError("Expects an model for Spacy")
        correct_model_path = self._get_correct_model_path(model.storage_path)
        self.ner_model = spacy.load(correct_model_path)

    def apply_ner(self, texts):
        if not isinstance(texts, list):
            if not isinstance(texts[0], str):
                raise TypeError("Expects a list of strings")

        results = []
        tokens = []
        for doc in self.ner_model.pipe(texts):
            tokens.append([token.text for token in doc])
            results.append([{"text":ent.text, "type":ent.label_,"start_token":ent.start,"end_token":ent.end-1,"start_pos":ent.start_char} for ent in doc.ents])
        return results, tokens

    def prepare_training_data(self, rows, tokenizer_path=None, train_size=0.8, validation_size=0.1, test_size=0.1,
                              split_sentences=False):
        train, valid, test  = self._train_test_split(rows, train_size, validation_size, test_size)
        self._bio_to_spacy(train,SPACY_TRAININGSDATA_PATH+"/train.spacy")
        self._bio_to_spacy(valid,SPACY_TRAININGSDATA_PATH+"/valid.spacy")

    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        config = load_config(self._get_correct_model_path(base_model_path)+"/config.cfg")
        config["paths"]["train"] = SPACY_TRAININGSDATA_PATH+"/train.spacy"
        config["paths"]["dev"] = SPACY_TRAININGSDATA_PATH+"/valid.spacy"
        config["training"]["max_steps"] = 600
        new_config_path = self._get_correct_model_path(base_model_path)+"/train_config.cfg"
        config.to_disk(new_config_path)
        train(new_config_path, output_path=new_model_path)

    def _finetune_default_spacy(self, base_model_path):
        nlp = spacy.load(self._get_correct_model_path(base_model_path))
        # only train embedding and ner
        frozen = [pipe for pipe in nlp.pipe_names if pipe not in ("tok2vec", "ner")]
        with nlp.disable_pipes(*frozen):
            optimizer = nlp.resume_training()
            for iteration in range(20):
                losses = {}



    def convert_ner_results(self, ner_results, ner_input):
        if isinstance(ner_input[0], ADGRow):
            return self._convert_ner_results_adg(ner_results, ner_input)
        else:
            tokens, predicted_labels = self._convert_ner_results_to_format(ner_results)
            return tokens, predicted_labels, None

    def _convert_ner_results_adg(self, ner_results,ner_input):
        results, tokens = ner_results
        for index, result in enumerate(results):
            # check if the adg-tokens and the model tokens are the same -> if not error
            if tokens[index] != ner_input[index].tokens:
                return ValueError("ADG-default tokens and model tokens are not identical")
        _, predicted_labels = self._convert_ner_results_to_format(ner_results)
        annoted_labels = [row.labels for row in ner_input]
        metrics = self._calc_metrics(annoted_labels, predicted_labels)
        return tokens, predicted_labels, metrics

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

    def _get_correct_model_path(self, path):
        if os.path.isfile(path+"/meta.json"):
            return path
        else:
            best_model_path = path + "/model-best"
            if os.path.isfile(best_model_path+"/meta.json"):
                return best_model_path
            else:
                raise ValueError(f"Cannot find model in {path}")

    def _get_training_examples_docbin(self, path,nlp):
        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        examples = [Example(doc, doc) for doc in docs]
        return examples