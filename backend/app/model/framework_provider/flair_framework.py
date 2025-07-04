from flair.data import Sentence
from sympy import false

from app.model.data_provider.adg_row import ADGRow
from app.model.data_provider.data_registry import data_registry
from app.model.framework_provider.framework import Framework, FrameworkNames
from flair.models import SequenceTagger

from app.model.ner_model_provider.ner_model import NERModel


class FlairFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None


    @property
    def default_finetuning_params(self):
        pass

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.FLAIR:
            raise TypeError("Expects an model for Flair")
        self.ner_model = model
        self.model = SequenceTagger.load(model.storage_path)

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

    def prepare_training_data(self, rows, tokenizer_path, train_size=0.8, validation_size=0.1, test_size=0.1,
                              split_sentences=False):
        tokens, labels = data_registry.split_training_data_sentences(rows)#
        sentences = []
        for index, token_sen in enumerate(tokens):
            sen = Sentence(token_sen, use_tokenizer=False)
            for label_index,label in enumerate(labels[index]):
                if label != "O":
                    sen.add_label("ner",label,label_index)
            sentences.append(sen)






    def finetune_ner_model(self, base_model_path, data_dict, label_id, name, new_model_path, params=None):
        pass

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
