import gc
import os
import time
from random import shuffle

import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import load_config, minibatch
from spacy.cli.train import train

from app.model.data_provider.adg_row import ADGRow, ADGSentence
from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.framework_provider.framework_utils import type_check_process_ner_pipeline
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.config import SPACY_TRAININGSDATA_PATH, BASE_MODELS_PATH, DEFAULT_TOKENIZER_PATH

'''
    notes:
        - transformer model only es encoder, no own component 
            own component: would only be a wrapper for the transformers libary 
        - to use "mschiessner/ner-bert-german" a base model has to be trained
            - the ner component for the transformer-embeddings has to be trained
'''
# -------------------------------------
# class SpacyFramework
# -------------------------------------
class SpacyFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None

    # -------------------------------------
    # public functions
    # -------------------------------------

    @property
    def default_finetuning_params(self):
        return {
            'max_epochs': 25,
            'max_steps': 0,
            'eval_frequency': 20,
            'learn_rate_warmup_steps':20,
            'learn_rate_total_steps':200
        }

    def load_model(self, model):
        """
        Loads the model. For further documentation see `framework.py`
        """
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.SPACY:
            raise TypeError("Expects an model for Spacy")
        correct_model_path = self._get_correct_model_path(model.storage_path)
        self.ner_model = spacy.load(correct_model_path)
    
    def process_ner_pipeline(self, model,ner_content, use_sentences = False):
        """
        Processing the ner pipeline. For further documentation see `framework.py`
        """
        type_check_process_ner_pipeline(model, ner_content, FrameworkNames.SPACY)
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
        """
        Applies NER on `texts`. For further documentation see `framework.py`
        """
        results = []
        tokens = []
        for doc in self.ner_model.pipe(texts):
            tokens.append([token.text for token in doc])
            results.append([{"text":ent.text, "type":ent.label_,"start_token":ent.start,"end_token":ent.end-1,"start_pos":ent.start_char} for ent in doc.ents])
        return results, tokens

    def prepare_training_data(self, rows, tokenizer_path=None, train_size=0.8, validation_size=0.2,
                              split_sentences=False, seed=None):
        """
        Converts the rows to train.spacy and valid.spacy files in /Trainingsdata/Spacy

        Returns
        (None,None)
        """
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")
        train, valid, test  = self._train_test_split(rows, train_size, validation_size, seed=seed)
        if split_sentences:
            train = data_registry.split_training_data_sentences(train)
            valid = data_registry.split_training_data_sentences(valid)
        self._bio_to_spacy(train,SPACY_TRAININGSDATA_PATH+"/train.spacy")
        self._bio_to_spacy(valid,SPACY_TRAININGSDATA_PATH+"/valid.spacy")
        return None, None
    
    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        """
        Finetunes the NER Model. Distinguishes between spacy models and transformer models.
         For further documentation see `framework.py`.
        """
        if params is None:
            params = self.default_finetuning_params

        correct_base_model_path = self._get_correct_model_path(base_model_path)
        metrics = None
        args = None
        if correct_base_model_path is None:
            if self._check_contain_base_config(base_model_path):
                # finetune/train from base config
                metrics, args =self._finetune_transformer_spacy(base_model_path, new_model_path,params, True)
        else:
            nlp = spacy.load(correct_base_model_path)
            if "transformer" in nlp.pipe_names:
                # finetune already finetuned transformer model
                metrics, args =self._finetune_transformer_spacy(correct_base_model_path, new_model_path,params, False)
            else:
                # finetune default spacy model
                metrics, args =self._finetune_default_spacy(correct_base_model_path, new_model_path)

        # manually call garbage collection -test
        gc.collect()
        return metrics, args


    def convert_ner_results(self, ner_results, ner_input, sentences = None):
        """
        Convert the ner-results.
        For further documentation see `framework.py`.
        """
        if isinstance(ner_input[0], ADGRow):
            return self._convert_ner_results_adg(ner_results, ner_input, sentences)
        else:
            tokens, predicted_labels = self._convert_ner_results_to_format(ner_results)
            return tokens, predicted_labels, None

    # -------------------------------------
    # private functions
    # -------------------------------------
    def _finetune_transformer_spacy(self, correct_base_model_path, new_model_path, params, from_base=False):
        """
        finetunes/trains a transformer-based model from only config.cfg or finetunes such a finetuned model
        Source: 
            https://medium.com/@zielemanj/training-and-fine-tuning-ner-transformer-models-using-spacy3-and-spacy-annotator-c3cd95fdfd23
            https://github.com/explosion/spaCy/discussions/9233

        Parameters:
        correct_base_model_path (str): the path of the base model from _get_correct_model_path
        new_model_path (str): the path of the new model
        params (dict): the parameters to apply like in `default_finetuning_params`

        Returns:
        (dict, params): the first dict contains f1, precision, recall, duration, accuracy = None, the second dict returns the params
        """
        #load config
        config = load_config(correct_base_model_path+"/config.cfg")

        # set train and valid datasets
        config["paths"]["train"] = SPACY_TRAININGSDATA_PATH+"/train.spacy"
        config["paths"]["dev"] = SPACY_TRAININGSDATA_PATH+"/valid.spacy"

        # set the training steps
        config["training"]["max_epochs"] = params["max_epochs"]
        # if the steps == 0, it is controller over epochs
        if(params["max_steps"]==0):
            config["training"].pop("max_steps",None)
        else:
            config["training"]["max_steps"] = params["max_steps"]
        config["training"]["eval_frequency"] = params["eval_frequency"]
        if "learn_rate_warmup_steps" in params:
            config["training"]["optimizer"]["learn_rate"]["warmup_steps"] = params["learn_rate_warmup_steps"]
        if "learn_rate_total_steps" in params:
            config["training"]["optimizer"]["learn_rate"]["total_steps"] = params["learn_rate_total_steps"]

        # if finetuning is applied on an already finetuned model
        if not from_base:
            if "factory" in config["components"]["ner"]:
                config["components"]["ner"].pop("factory",None)
            config["components"]["ner"]["source"] = correct_base_model_path
            config["initialize"]["components"]["transformer"] = {"source": correct_base_model_path}
            if "factory" in config["components"]["transformer"]:
                config["components"]["transformer"].pop("factory",None)
            config["components"]["transformer"]["source"] = correct_base_model_path
            config["initialize"]["components"]["ner"] = {"source": correct_base_model_path}

        # saves the training config for the new model in the directory of the base model
        new_config_path = correct_base_model_path+"/train_config.cfg"
        config.to_disk(new_config_path)
        start = time.time()
        train(new_config_path, output_path=new_model_path)
        end = time.time()
        metrics =self._evaluate_transformer_model(self._get_correct_model_path(new_model_path))
        metrics.duration = end-start
        return metrics, params

    def _finetune_default_spacy(self, correct_base_model_path, new_model_path, epochs=30, minibatch_size=16):
        """
        Finetunes a spacy model

        Parameters
        correct_base_model_path:
        correct_base_model_path (str): the path of the base model from _get_correct_model_path
        epochs (int): iterations that should be trained
        minibatch_size (int): minibatch size

        Returns:
        (dict, params): the first dict contains f1, precision, recall, duration, accuracy = None, the second dict returns the params
        """
        nlp = spacy.load(self._get_correct_model_path(correct_base_model_path))
        train_examples = self._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/train.spacy")
        valid_examples = self._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/valid.spacy")
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
        """
        Evaluates the transformers model by applying it on the valid data.

        Parameters
        path (str): the path of the transformers model

        Returns
        (TrainingResults)
        """
        nlp = spacy.load(path)
        valid_examples =self._get_training_examples_docbin(SPACY_TRAININGSDATA_PATH+"/valid.spacy")
        scores = self._evaluate_finetune_spacy(nlp, valid_examples)
        return TrainingResults(f1=scores["ents_f"],recall=scores["ents_r"],precision=scores["ents_p"],duration=None, accuracy=None)
        
    def _evaluate_finetune_spacy(self, nlp, examples):
        """
        Evaluates a spacy model by applying it on the valid data.

        Parameters
        nlp: A spacy language object from spacy.load
        examples (Example):

        Returns:
        (dict): with the metrics in "ents_f", "ents_r", "ents_p"
        """
        scorer = Scorer()
        examples_with_ref = []
        for example in examples:
            pred = nlp(example.text)
            examples_with_ref.append(Example(pred, example.reference))
        return scorer.score(examples_with_ref)


    def _convert_ner_results_adg(self, ner_results,ner_input, sentences):
        """
        Convert the ner_results for adg inputs. Checks if the tokens from the applied model are the same as the default adg tokens.
        For further documentation see `framework.py`.
        """
        results, tokens = ner_results
        annoted_labels = None

        # test if labels are the same
        if sentences:
            # check if the tokens of this spacy model are the same as the default ones
            for index, tokens_sen in enumerate(tokens):
               if tokens_sen != sentences[index].tokens:
                   raise ValueError("ADG-default tokens and model tokens are not identical")
            annoted_labels = [sen.labels for sen in sentences]
        else:
            # check if the tokens of this spacy model are the same as the default ones
            for index, tokens_sen in enumerate(tokens):
                # check if the adg-tokens and the model tokens are the same -> if not error
                if tokens_sen != ner_input[index].tokens:
                    raise ValueError("ADG-default tokens and model tokens are not identical")
            annoted_labels= [row.labels for row in ner_input]

        _, predicted_labels = self._convert_ner_results_to_format(ner_results)
        metrics = self._calc_metrics(annoted_labels, predicted_labels)
        return tokens, predicted_labels, metrics

    def _bio_to_spacy(self, data, output_path):
        """
        Creates .spacy file the annotated data

        Parameters
        data (ADGRow | ADGSentence): the data that should be saved in an .spacy file
        output_path (str): the path where the .spacy file should be saved
        """
        on_sentence = False
        if isinstance(data[0], ADGSentence):
            on_sentence = True

        doc_bin = DocBin()
        nlp = spacy.load(DEFAULT_TOKENIZER_PATH)
        for row_sen in data:
            doc = self._create_doc(row_sen, nlp=nlp)
            doc_bin.add(doc)

        doc_bin.to_disk(output_path)

    def _create_doc(self, statement, nlp):
        """
        Creates a spacy DocBin Object from `tokens` and `labels`

        Parameters
        statement (ADGRow | ADGSentence): data that should be converted to DocBin
        nlp (spacy language object)


        Returns
        (DocBin)
        """
        on_sentence = False
        if isinstance(statement, ADGSentence):
            on_sentence = True

        # make a doc with the default tokenizer
        doc = nlp.make_doc(statement.text)
        ents = []
        start = 0
        current_ent = None
        for index, label in enumerate(statement.labels):
            token = statement.tokens[index]
            startindex_label = None
            if on_sentence:
                startindex_label = statement.token_indexes[index]
            else:
                startindex_label = statement.indexes[index]

            start = startindex_label
            end = startindex_label + len(token)
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
        if current_ent:
            ents.append(current_ent)
        doc.ents = [doc.char_span(start, end, label=label) for start, end, label in ents if
                    doc.char_span(start, end, label=label)]
        return doc

    def _get_correct_model_path(self, path):
        """
        Returns the path with meta.json from a directory, even if the model is in model-best

        Parameters
        path (str): path to check

        Returns
        (str | None)
        """
        if os.path.isfile(path+"/meta.json"):
            return path
        else:
            best_model_path = path + "/model-best"
            if os.path.isfile(best_model_path+"/meta.json"):
                return best_model_path
            else:
                return None

    def _check_contain_base_config(self, path):
        """
        Check if `path` is the path of a base model

        Parameters
        path (str): path to check

        Returns
        (bool)
        :return:
        """
        if os.path.isfile(path+"/config.cfg"):
            return True
        else:
            return False

    def _get_training_examples_docbin(self, path):
        """
        Returns the data from an spacy file as a List of Example objects

        Parameters
        path (str): the path where the .spacy file is stored

        Returns
        (List[Example])
        """
        nlp = spacy.load(DEFAULT_TOKENIZER_PATH)
        doc_bin = DocBin().from_disk(path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        examples = []
        for doc in docs:
            pred_doc = nlp.make_doc(doc.text)
            examples.append(Example(pred_doc, doc))
        #refs = [example.reference.ents for example in examples]
        return examples