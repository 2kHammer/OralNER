import os
import time

from flair.data import Sentence, Corpus
from flair.datasets import ColumnCorpus
from sympy import false
import random
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.utils.checkpoint import checkpoint

from app.model.data_provider.adg_row import ADGRow
from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.framework import Framework, FrameworkNames
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.config import CONLL_PATH


class FlairFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None


    @property
    def default_finetuning_params(self):
        return {
            "learning_rate": 0.0025,
            "mini_batch_size": 64,
            "max_epochs": 20,
        }

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.FLAIR:
            raise TypeError("Expects an model for Flair")
        self.ner_model = model
        file_path = self._get_pt_file(model.storage_path)
        self.model = SequenceTagger.load(file_path)

    #input has to be in sentences - check this
    # abstract type checks
    def apply_ner(self, texts):
        """
        Applies NER on `texts` with the current flair model.

        Parameters:
        texts ([str]): have to be sentences

        Returns:
            List, List: first List contains dicts with the entities: "text","type","start_token","end_token","start_pos", second List contains Tokens
        """
        if not isinstance(texts, list):
            if not isinstance(texts[0], str) or not isinstance(texts[0], list):
                raise TypeError("Expects a list of strings or list of token-lists")

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
    def prepare_training_data(self, rows, tokenizer_path=None, train_size=0.8, validation_size=0.1, test_size=0.1,
                              split_sentences=False):
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")

        #make corpus of sentences,
        # Problem: sentences from statements are parted in the three datasets
        tokens = None
        labels = None
        if split_sentences:
            #statements shouldn't be split across datasets
            tokens, labels = data_registry.split_training_data_sentences(rows)
        else:
            tokens = [row.tokens for row in rows]
            labels  = [row.labels for row in rows]

        '''
        sentences = []
        for index, token_sen in enumerate(tokens):
            sen = Sentence(token_sen, use_tokenizer=False)
            for label_index,label in enumerate(labels[index]):
                if label != "O":
                    sen.tokens[label_index].add_label("ner",label)
            sentences.append(sen)
        train, valid, test = train_test_split(sentences, train_size,validation_size, test_size)
        '''
        tokens_with_labels = [{"tokens":token, "labels":labels[index]}for index,token in enumerate(tokens)]
        train, valid, test = self._train_test_split(tokens_with_labels, train_size,validation_size, test_size)
        self._create_conll_files(train, valid, test)
        corpus = ColumnCorpus(CONLL_PATH,{0:'text',1:'ner'},train_file="train.txt",dev_file="valid.txt",test_file="test.txt")
        return corpus, corpus.make_label_dictionary(label_type="ner")


    #Source: https://flairnlp.github.io/flair/v0.14.0/tutorial/tutorial-training/how-to-train-sequence-tagger.html
    # Optimation: document label features with transformer embeddings : https://arxiv.org/pdf/2011.06993
    # means -> the next / previous sentence object is for our case irrelevant
    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        if params is None:
            params = self.default_finetuning_params

        base_model = SequenceTagger.load(self._get_pt_file(base_model_path))

        #train new bilstm-crf classification layer, finetune the embeddings of the base model
        finetuned_model = SequenceTagger(hidden_size=256,
                                         embeddings=base_model.embeddings,
                                         tag_dictionary=label_id,
                                         tag_type="ner",
                                         use_crf=True)

        print(finetuned_model)
        start = time.time()
        trainer = ModelTrainer(finetuned_model, data_dict)
        trainer.fine_tune(
            base_path=new_model_path+"/"+name,
            learning_rate=params["learning_rate"],
            mini_batch_size=params["mini_batch_size"],
            max_epochs=params["max_epochs"],
        )
        end = time.time()
        result =trainer.model.evaluate(data_dict.test, gold_label_type="ner")
        class_report_micro = result.classification_report["micro avg"]
        train_res = TrainingResults(class_report_micro["f1-score"],class_report_micro["precision"],class_report_micro["recall"],end-start,result.scores["accuracy"])
        return train_res, params

    # abstract eventually
    def convert_ner_results(self, ner_results, ner_input, annoted_labels=None):
        if annoted_labels != None:
            tokens, predicted_labels =  self._convert_ner_results_to_format(ner_results, ner_input)
            metrics = self._calc_metrics(annoted_labels, predicted_labels)
            return tokens, predicted_labels, metrics
        else:
            tokens, predicted_labels = self._convert_ner_results_to_format(ner_results, ner_input)
            return tokens, predicted_labels, None

    def _convert_ner_results_to_format(self, ner_results, ner_input):
        results, tokens = ner_results
        labels = []
        for index,token_sentence in enumerate(tokens):
            label_sentence = ["O"]*len(token_sentence)
            for entity in results[index]:
                start_token = entity["start_token"]
                end_token = entity["end_token"]
                typ =entity["type"]
                label_sentence[start_token]="B-"+typ
                for i in range(start_token+1, end_token+1):
                    label_sentence[i]="I-"+typ
            labels.append(label_sentence)
        return tokens, labels

    def _create_conll_files(self, train, valid, test):
        tokens_train = [t["tokens"] for t in train]
        labels_train = [t["labels"] for t in train]
        save_to_conll(tokens_train, labels_train, CONLL_PATH+"/train.txt")
        tokens_valid = [t["tokens"] for t in valid]
        labels_valid = [t["labels"] for t in valid]
        save_to_conll(tokens_valid, labels_valid, CONLL_PATH+"/valid.txt")
        tokens_test = [t["tokens"] for t in test]
        labels_test = [t["labels"] for t in test]
        save_to_conll(tokens_test, labels_test, CONLL_PATH+"/test.txt")

    def _get_pt_file(self,path):
        for file in os.listdir(path):
            if file.endswith(".pt"):
                return os.path.join(path,file)
        return None

def save_to_conll(tokens, labels, path_to_save):
    with open(path_to_save, "w", encoding="utf-8") as f:
        for token_sentence, label_sentence in zip(tokens, labels):
            for token, label in zip(token_sentence, label_sentence):
                f.write(f"{token} {label}\n")
            f.write("\n")

