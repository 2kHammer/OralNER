from lib2to3.btm_utils import tokens

from .framework import Framework
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset

from ..data_provider.data_registry import ADGRow
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

    def prepare_training_data(self, rows):
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")

        # sort the labels or insert all, if time
        all_labels = list(set(label for row in rows for label in row.labels))
        label_id = {label: i for i, label in enumerate(all_labels)}
        id_label = {i: label for i, label in enumerate(all_labels)}
        print(len(rows))
        data = Dataset.from_list([{"tokens":row.tokens,"labels":[label_id[label] for label in row.labels]} for row in rows[1:]])
        print(data)
        tokenized_data = data.map(self._tokenize_and_align_labels, batched=True)
        print(tokenized_data)
        '''
        testid = 29
        label_ids =[label_id[label] for label in rows[testid].labels]
        print(len(label_ids))

        inputs = self.tokenizer(rows[testid].tokens, is_split_into_words=True)
        words_ids =inputs.word_ids()
        print(len(rows[testid].tokens))
        print(len(words_ids))
        new_labels = self._align_labels_with_tokens(label_ids, words_ids)
        print(new_labels)
        '''

    #https: // huggingface.co / docs / transformers / tasks / token_classification
    def _tokenize_and_align_labels(self, statement):
        tokenized_inputs = self.tokenizer(statement["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(statement[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _align_labels_with_tokens(self,labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels
        