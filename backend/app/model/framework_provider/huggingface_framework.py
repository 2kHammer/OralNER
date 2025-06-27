from lib2to3.btm_utils import tokens

from .framework import Framework
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

from app.model.data_provider.data_registry import ADGRow
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.helpers import delete_checkpoints_folder


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
        print(model.name + " is loaded")
        
    def apply_ner(self, texts):
        if not isinstance(texts, list):
            if not isinstance(texts[0], str):
                raise TypeError("Expects a list of strings")

        ner_results = []
        for text in texts:
            nlp = pipeline("ner",model=self.model,tokenizer=self.tokenizer, aggregation_strategy=None)
            ner_results.append(nlp(text))
        return ner_results

    def prepare_training_data(self, rows, tokenizer_path, train_size=0.7, validation_size=0.1, test_size=0.2):
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")

        # trainingsdaten auf sätze umstellen

        # vielleicht später noch auf ein globales Dictionary umändern
        # sort the labels or insert all, if time
        all_labels = list(set(label for row in rows for label in row.labels))
        label_id = {label: i for i, label in enumerate(all_labels)}
        data = Dataset.from_list([{"tokens":row.tokens,"labels":[label_id[label] for label in row.labels]} for row in rows[1:]])
        # read full data
        tokenized_data = data.map(lambda x: self._tokenize_and_align_labels(x,tokenizer_path), batched=True)

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

        # best modell is saved in directory as checkpoint-xx, then loaded in the trainer (load_best_model_at_the_end)
        # -> überschreibt manuell das Verzeichnis
        training_args = TrainingArguments(
            output_dir=new_model_path+name,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: self.compute_metrics(p, list_labels),
        )

        train_results =trainer.train()
        metrics = trainer.evaluate()
        args = trainer.args.to_dict()
        trainer.save_model(new_model_path+name)
        delete_checkpoints_folder(new_model_path+name)
        print("Training done")
        print(metrics)
        return self._convert_metrics(metrics,train_results.metrics["train_runtime"]), args

    # split function
    # add a less strict type comparison, without B- and I-
    def convert_ner_results(self,ner_results, ner_input):
        metrics = None
        tokens = []
        predicted_labels = []
        annoted_labels = []
        if isinstance(ner_input[0], ADGRow):
            for index,result in enumerate(ner_results):
                adg_row = ner_input[index]
                labels_row = ["O"] * len(adg_row.labels)
                for entity in result:
                    if entity["word"][:2] != '##':#
                        try:
                            index_label = -1
                            if entity["start"] in adg_row.indexes:
                                index_label = adg_row.indexes.index(entity["start"])
                            else:
                                # if only a part of the wort is recognized as entity -> apply to full word
                                for index, index_val in enumerate(adg_row.indexes):
                                    if index_val > entity["start"]:
                                        index_label = index -1
                                        break
                            labels_row[index_label] = entity["entity"]
                        except ValueError:
                            print("Error by assigning the predicted labels")
                            #print(adg_row.text)
                            #print(adg_row.entities)
                predicted_labels.append(labels_row)
                tokens.append(adg_row.tokens)
                annoted_labels.append(adg_row.labels)
            metrics = {
                "f1" :f1_score(annoted_labels,predicted_labels),
                "recall" : recall_score(annoted_labels,predicted_labels),
                "precision" : precision_score(annoted_labels,predicted_labels),
                "accuracy" : accuracy_score(annoted_labels,predicted_labels)
            }
        else:
            for index_sentence, sentence in enumerate(ner_input):

                # prepare tokens for return
                tokens_hf =self.tokenizer(sentence, return_offsets_mapping=True)
                sub_tokens = tokens_hf.tokens()[1:-1]
                sub_startindexes = [index[0] for index in tokens_hf["offset_mapping"]][1:-1]
                tokens_sentence = []
                startindexes = []
                last_append_index = 0
                for i in range(0, len(sub_tokens)):
                    if sub_tokens[i][:2] == "##":
                        tokens_sentence[last_append_index] += sub_tokens[i][2:]
                    else:
                        tokens_sentence.append(sub_tokens[i])
                        last_append_index = len(tokens_sentence)-1
                        startindexes.append(sub_startindexes[i])

                labels_sentence = ["O"]* len(tokens_sentence)
                # prepare recognized entities
                for entities in ner_results[index_sentence]:
                    if entities["word"][0:2] != "##":
                        #print(entities)
                        #print(startindexes)
                        index_entity = startindexes.index(entities["start"])
                        #print(tokens_sentence[index_entity])
                        labels_sentence[index_entity] = entities["entity"]

                tokens.append(tokens_sentence)
                predicted_labels.append(labels_sentence)
        return tokens, predicted_labels, metrics





    #https: // huggingface.co / docs / transformers / tasks / token_classification
    def _tokenize_and_align_labels(self, statement, tokenizer_path):
        model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenized_inputs = model_tokenizer(statement["tokens"], truncation=True, is_split_into_words=True)

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

    # diese Funktion für alle Framework Klassen abstrakt machen
    def _convert_metrics(self, metrics, duration):
        print(duration)
        return TrainingResults(f1=metrics["eval_f1"], recall=metrics["eval_recall"], precision=metrics["eval_precision"], duration=duration, accuracy=metrics["eval_accuracy"])
