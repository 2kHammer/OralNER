import os
import time
from random import shuffle

import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import load_config, minibatch
from spacy.cli.train import train
from typing_extensions import override

from app.model.data_provider.adg_row import ADGRow
from app.model.data_provider.data_registry import simple_split_sentences, data_registry
from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
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
        return {
            'max_epochs': 0,
            'max_steps': 800,
            'eval_frequency': 20
        }

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.SPACY:
            raise TypeError("Expects an model for Spacy")
        correct_model_path = self._get_correct_model_path(model.storage_path)
        self.ner_model = spacy.load(correct_model_path)
    
    def process_ner_pipeline(self, model,ner_content, use_sentences = False):
        if not isinstance(ner_content, list):
            if not isinstance(ner_content[0], str) or not isinstance(ner_content[0], ADGRow):
                raise TypeError("Excepts a list of strings or ADGRows")
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.SPACY:
            raise ValueError("Expects an model for Spacy")

        self.load_model(model)
        adg_sentences = None
        results = None
        if isinstance(ner_content[0], ADGRow):
            if use_sentences:
                adg_sentences = data_registry.split_training_data_sentences(ner_content)
                sen = [sen.text for sen in adg_sentences]
                results = self.apply_ner(sen)
            else:
                results = self.apply_ner([row.text for row in ner_content])
        else:
            results = self.apply_ner(ner_content)
        tokens, predicted_labels, metrics =self.convert_ner_results(results,ner_content,adg_sentences)
        return tokens, predicted_labels, metrics




    def apply_ner(self, texts):
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
        return None, None
    
    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        if params is None:
            params = self.default_finetuning_params

        correct_base_model_path = self._get_correct_model_path(base_model_path)
        nlp = spacy.load(correct_base_model_path)
        metrics = None
        args = None
        if "transformer" in nlp.pipe_names:
            metrics, args =self._finetune_transformer_spacy(correct_base_model_path, new_model_path,params)
        else:
            metrics, args =self._finetune_default_spacy(correct_base_model_path, new_model_path)
        return metrics, args

    #https://medium.com/@zielemanj/training-and-fine-tuning-ner-transformer-models-using-spacy3-and-spacy-annotator-c3cd95fdfd23
    def _finetune_transformer_spacy(self, correct_base_model_path, new_model_path, params):
        correct_base_model_path = self._get_correct_model_path(correct_base_model_path)
        config = load_config(correct_base_model_path+"/config.cfg")
        config["paths"]["train"] = SPACY_TRAININGSDATA_PATH+"/train.spacy"
        config["paths"]["dev"] = SPACY_TRAININGSDATA_PATH+"/valid.spacy"
        config["training"]["max_epochs"] = params["max_epochs"]
        config["training"]["max_steps"] = params["max_steps"]
        config["training"]["eval_frequency"] = params["eval_frequency"]
        if "factory" in config["components"]["ner"]:
            config["components"]["ner"].pop("factory",None)
        config["components"]["ner"]["source"] = correct_base_model_path
        if "factory" in config["components"]["transformer"]:
            config["components"]["transformer"].pop("factory",None)
        config["components"]["transformer"]["source"] = correct_base_model_path

        new_config_path = correct_base_model_path+"/train_config.cfg"
        config.to_disk(new_config_path)
        start = time.time()
        train(new_config_path, output_path=new_model_path)
        end = time.time()
        metrics =self._evaluate_transformer_model(self._get_correct_model_path(new_model_path))
        metrics.duration = end-start
        return metrics, params

    def _finetune_default_spacy(self, correct_base_model_path, new_model_path, epochs=30, minibatch_size=16):
        nlp = spacy.load(self._get_correct_model_path(correct_base_model_path))
        train_examples = self._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/train.spacy", nlp)
        valid_examples = self._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/valid.spacy", nlp)
        # only train embedding and ner
        start = time.time()
        best_metrics = None
        frozen = [pipe for pipe in nlp.pipe_names if pipe not in ("tok2vec", "ner")]
        with nlp.disable_pipes(*frozen):
            optimizer = nlp.resume_training()
            for iteration in range(epochs):
                losses = {}
                shuffle(train_examples)
                batches = minibatch(train_examples, minibatch_size)
                for batch in batches:
                    nlp.update(batch, sgd=optimizer, losses=losses)
                scores = self._evaluate_finetune_spacy(nlp, valid_examples)
                metrics = TrainingResults(f1=scores["ents_f"],recall=scores["ents_r"],precision=scores["ents_p"],duration=None, accuracy=None)

                if (best_metrics is None) or (metrics.f1 > best_metrics.f1):
                    print("better model")
                    best_metrics = metrics
                    nlp.to_disk(new_model_path)

        end = time.time()
        best_metrics.duration = end - start
        return best_metrics, {"epochs":epochs, "minibatch_size":minibatch_size}
    
    def _evaluate_transformer_model(self, path):
        nlp = spacy.load(path)
        valid_examples =self._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/valid.spacy", nlp)
        scores = self._evaluate_finetune_spacy(nlp, valid_examples)
        return TrainingResults(f1=scores["ents_f"],recall=scores["ents_r"],precision=scores["ents_p"],duration=None, accuracy=None)
        
    def _evaluate_finetune_spacy(self, nlp, examples):
        scorer = Scorer()
        examples_with_ref = []
        for example in examples:
            pred = nlp(example.text)
            examples_with_ref.append(Example(pred, example.reference))
        return scorer.score(examples_with_ref)

    def convert_ner_results(self, ner_results, ner_input, sentences = None):
        if isinstance(ner_input[0], ADGRow):
            return self._convert_ner_results_adg(ner_results, ner_input, sentences)
        else:
            tokens, predicted_labels = self._convert_ner_results_to_format(ner_results)
            return tokens, predicted_labels, None

    def _convert_ner_results_adg(self, ner_results,ner_input, sentences):
        results, tokens = ner_results
        annoted_labels = None
        if sentences:
            # check if the tokens of this spacy model are the same as the default ones
            for index, tokens_sen in enumerate(tokens):
               if tokens_sen != sentences[index].tokens:
                   return ValueError("ADG-default tokens and model tokens are not identical")
            annoted_labels = [sen.labels for sen in sentences]
        else:
            # check if the tokens of this spacy model are the same as the default ones
            for index, tokens_sen in enumerate(tokens):
                # check if the adg-tokens and the model tokens are the same -> if not error
                if tokens_sen != ner_input[index].tokens:
                    return ValueError("ADG-default tokens and model tokens are not identical")
            annoted_labels= [row.labels for row in ner_input]

        _, predicted_labels = self._convert_ner_results_to_format(ner_results)
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
        examples = []
        for doc in docs:
            pred_doc = nlp.make_doc(doc.text)
            examples.append(Example(pred_doc, doc))
        refs = [example.reference.ents for example in examples]
        return examples