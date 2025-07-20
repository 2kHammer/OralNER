import os
import time

from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from app.model.data_provider.adg_row import ADGRow
from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.framework_provider.framework_utils import type_check_process_ner_pipeline
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.config import CONLL_PATH

'''
    notes:
        - can't finetune the bilstm-crf layer of flair/ner-german if the entity-types are different
        - so we train a new bilstm layer while the embeddings of flair/ner-german are finetuned
'''
# -------------------------------------
# class FlairFramework
# -------------------------------------
class FlairFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None


    # -------------------------------------
    # public functions
    # -------------------------------------
    @property
    def default_finetuning_params(self):
        return {
            "learning_rate": 0.0025,
            "mini_batch_size": 64,
            "max_epochs": 20,
        }

    def load_model(self, model):
        """
        Loads the model. For further documentation see `framework.py`
        """
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.FLAIR:
            raise TypeError("Expects an model for Flair")
        self.ner_model = model
        file_path = self._get_pt_file(model.storage_path)
        self.model = SequenceTagger.load(file_path)

    def process_ner_pipeline(self, model, ner_content, use_sentences = False):
        """
        Processing the ner pipeline. For further documentation see `framework.py`
        """
        type_check_process_ner_pipeline(model, ner_content, FrameworkNames.FLAIR)

        self.load_model(model)
        results = None
        adg_sentences = None
        if isinstance(ner_content[0], ADGRow):
            if use_sentences:
                adg_sentences = data_registry.split_training_data_sentences(ner_content)
                tokens = [sent.tokens for sent in adg_sentences]
                results = self.apply_ner(tokens)
            else:
                results = self.apply_ner([row.tokens for row in ner_content])
        else:
           results = self.apply_ner(ner_content)

        tokens, predicted_labels, metrics = self.convert_ner_results(results,ner_content,adg_sentences)
        return tokens, predicted_labels, metrics



    #input has to be in sentences - check this
    # abstract type checks
    def apply_ner(self, texts):
        """
        Applies NER on `texts`. For further documentation see `framework.py`

        Parameters
        texts (List[str] | List[List[str]]): if it should be applied on adg-files, ´texts´ contains the list of tokens

        Returns
        (List[dict], List[str]): First List contains the dicts for each statement with: "text", "type", "start_token", "end_token", "start_pos", the second list contains the tokens
        """
        # check if it should be applied on tokens
        apply_on_labeled_data = False
        if isinstance(texts[0], list):
            apply_on_labeled_data = True

        tokens = []
        results = []
        sentences = []
        for text in texts:
            # if it is applied on a adg-file -> own tokens are used
            if apply_on_labeled_data:
                # the results are not as good for the base model, because it's trained on the flair tokens
                # but we want to use it on our finetuned models, so its okay
                sentences.append(Sentence(text, use_tokenizer=False))
            else:
                sentences.append(Sentence(text))
        self.model.predict(sentences, mini_batch_size=32, verbose=True)
        for sentence in sentences:
            results.append([{"text":label.data_point.text,"type":label.value,"start_token":label.data_point.tokens[0].idx-1,"end_token":label.data_point.tokens[-1].idx-1,"start_pos":label.data_point.start_position} for label in sentence.get_labels()])
            tokens.append([token.text for token in sentence.tokens])

        return results, tokens

    # can the data be loaded to keep the context of the sentences
    # the sentences contain a next and previous sentences object?
    def prepare_training_data(self, rows, tokenizer_path=None, train_size=0.8, validation_size=0.2,
                              split_sentences=False, seed=None):
        """
        Converts the rows to a train.txt and valid.txt file which are read in as a flair ColumnCorpus object
        For further documentation see `framework.py`.

        Returns
        (ColumnCorupus, dict)
        """
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")

        #make corpus of sentences,
        # Problem: sentences from statements are parted in the three datasets
        train, valid, _ = self._train_test_split(rows, train_size, validation_size, seed=seed)
        train_modified = None
        valid_modified = None
        if split_sentences:
            #statements shouldn't be split across datasets
            sentence_data_train = data_registry.split_training_data_sentences(train)
            sentence_data_valid = data_registry.split_training_data_sentences(valid)
            train_modified = [{"tokens":sen.tokens, "labels":sen.labels} for sen in sentence_data_train]
            valid_modified = [{"tokens":sen.tokens, "labels":sen.labels} for sen in sentence_data_valid]
        else:
            train_modified = [{"tokens":row.tokens, "labels":row.labels} for row in train]
            valid_modified = [{"tokens":row.tokens, "labels":row.labels} for row in valid]

        self._create_conll_files(train_modified, valid_modified)
        # use no test dataset -> include valid.txt twice, for no errors
        corpus = ColumnCorpus(CONLL_PATH,{0:'text',1:'ner'},train_file="train.txt",dev_file="valid.txt",test_file="valid.txt")
        return corpus, corpus.make_label_dictionary(label_type="ner")


    #Source: https://flairnlp.github.io/flair/v0.14.0/tutorial/tutorial-training/how-to-train-sequence-tagger.html

    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        """
        Finetunes the embeddings of the default model, trains a new bilstm-crf layer.
        For further documentation see `framework.py`.
        """
        if params is None:
            params = self.default_finetuning_params

        base_model = SequenceTagger.load(self._get_pt_file(base_model_path))

        #train new bilstm-crf classification layer, finetune the embeddings of the base model
        finetuned_model = SequenceTagger(hidden_size=256,
                                         embeddings=base_model.embeddings,
                                         tag_dictionary=label_id,
                                         tag_type="ner",
                                         use_crf=True)

        start = time.time()
        trainer = ModelTrainer(finetuned_model, data_dict)
        trainer.fine_tune(
            base_path=new_model_path,
            learning_rate=params["learning_rate"],
            mini_batch_size=params["mini_batch_size"],
            max_epochs=params["max_epochs"],
        )
        end = time.time()
        result =trainer.model.evaluate(data_dict.dev, gold_label_type="ner")
        class_report_micro = result.classification_report["micro avg"]
        train_res = TrainingResults(class_report_micro["f1-score"],class_report_micro["precision"],class_report_micro["recall"],end-start,result.scores["accuracy"])
        return train_res, params

    def convert_ner_results(self, ner_results, ner_input, sentences = None):
        """
        Convert the ner-results.
        For further documentation see `framework.py`.
        """
        if isinstance(ner_input[0], ADGRow):
            annoted_labels = [row.labels for row in ner_input]
            if sentences:
                annoted_labels = [sen.labels for sen in sentences]
            tokens, predicted_labels =  self._convert_ner_results_to_format(ner_results)
            metrics = self._calc_metrics(annoted_labels, predicted_labels)
            return tokens, predicted_labels, metrics
        else:
            tokens, predicted_labels = self._convert_ner_results_to_format(ner_results)
            return tokens, predicted_labels, None

    # -------------------------------------
    # private functions
    # -------------------------------------
    def _create_conll_files(self, train, valid, test=None):
        tokens_train = [t["tokens"] for t in train]
        labels_train = [t["labels"] for t in train]
        save_to_conll(tokens_train, labels_train, CONLL_PATH + "/train.txt")
        tokens_valid = [t["tokens"] for t in valid]
        labels_valid = [t["labels"] for t in valid]
        save_to_conll(tokens_valid, labels_valid, CONLL_PATH + "/valid.txt")
        if test:
            tokens_test = [t["tokens"] for t in test]
            labels_test = [t["labels"] for t in test]
            save_to_conll(tokens_test, labels_test, CONLL_PATH + "/test.txt")

    def _get_pt_file(self,path):
        for file in os.listdir(path):
            if file.endswith(".pt"):
                return os.path.join(path,file)
        return None


# -------------------------------------
# functions
# -------------------------------------
def save_to_conll(tokens, labels, path_to_save):
    with open(path_to_save, "w", encoding="utf-8") as f:
        for token_sentence, label_sentence in zip(tokens, labels):
            for token, label in zip(token_sentence, label_sentence):
                f.write(f"{token} {label}\n")
            f.write("\n")

