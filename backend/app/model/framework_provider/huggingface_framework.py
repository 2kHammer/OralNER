from lib2to3.btm_utils import tokens

from .framework import Framework
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np

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

    def prepare_training_data(self, rows, train_size=0.7, validation_size=0.1, test_size=0.2):
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")

        # trainingsdaten auf sätze umstellen

        # vielleicht später noch auf ein globales Dictionary umändern
        # sort the labels or insert all, if time
        all_labels = list(set(label for row in rows for label in row.labels))
        label_id = {label: i for i, label in enumerate(all_labels)}
        data = Dataset.from_list([{"tokens":row.tokens,"labels":[label_id[label] for label in row.labels]} for row in rows[1:]])
        # read full data
        tokenized_data = data.map(self._tokenize_and_align_labels, batched=True)

        # wird validation überhaupt benötigt?
        split_test = tokenized_data.train_test_split(test_size=test_size, seed=42)
        test_data = split_test["test"]
        train_data = split_test["train"]

        split_validation = train_data.train_test_split(test_size=validation_size/(1-test_size), seed=42)
        validation_data = split_validation["test"]
        train_data = split_validation["train"]

        dataset_dict = DatasetDict({"train":train_data, "validation":validation_data, "test":test_data})
        return dataset_dict, label_id

    def finetune_ner_model(self,base_model_path,data_dict, label_id, name,new_model_path):
        id_label = {value: key for key, value in label_id.items()}
        sorted_label_id = dict(sorted(label_id.items(), key=lambda item: item[1]))
        list_labels = list(sorted_label_id.keys())

        # wird über Path angeben
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForTokenClassification.from_pretrained(base_model_path,num_labels=len(list_labels),id2label=id_label, label2id=label_id,ignore_mismatched_sizes=True)
        # ausgliedern in JSON-Datei
        print(new_model_path+name)
        training_args = TrainingArguments(
            output_dir=new_model_path+name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: self.compute_metrics(p, list_labels),
        )

        trainer.train()
        metrics = trainer.evaluate()
        #Metriken und die Trainingsargumente speichern

        print("Training done")


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

    def compute_metrics(self,p, label_list):
        seqeval = evaluate.load("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }