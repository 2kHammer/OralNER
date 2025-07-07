from lib2to3.btm_utils import tokens

from .framework import Framework, FrameworkNames
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from evaluate import load
import numpy as np

from app.model.data_provider.data_registry import simple_split_sentences, simple_tokenizer, data_registry
from app.model.data_provider.adg_row import ADGRow
from app.model.ner_model_provider.ner_model import NERModel, TrainingResults
from app.utils.helpers import delete_checkpoints_folder


class HuggingFaceFramework(Framework):
    def __init__(self):
        self.ner_model = None
        self.model = None
        self.tokenizer = None

    @property
    def default_finetuning_params(self):
        return {
            "evaluation_strategy":"epoch",
            "save_strategy": "epoch",
            "save_total_limit" : 3,
            "learning_rate" :2e-5,
            "per_device_train_batch_size":4,
            "per_device_eval_batch_size" :4,
            "num_train_epochs": 3,
            "weight_decay" :0.01,
            "load_best_model_at_end" : True,
            "metric_for_best_model": "f1",
        }

    def load_model(self, model):
        if not isinstance(model, NERModel):
            raise TypeError("Expects an object of type NERModel")
        if model.framework_name != FrameworkNames.HUGGINGFACE:
            raise TypeError("Expects an model for HuggingFace")
        self.ner_model = model
        self.model = AutoModelForTokenClassification.from_pretrained(model.storage_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model.storage_path)

    def apply_ner(self, texts):
        if not isinstance(texts, list):
            if not isinstance(texts[0], str):
                raise TypeError("Expects a list of strings")

        ner_results = []
        for text in texts:
            #print(text)
            # maybe change the aggregation strategy
            # subtokens could be labeled as entities
            nlp = pipeline("ner",model=self.model,tokenizer=self.tokenizer, aggregation_strategy=None)
            ner_results.append(nlp(text))
        return ner_results

    def prepare_training_data(self, rows, tokenizer_path=None, train_size=0.8, validation_size=0.1, test_size=0.1, split_sentences=False):
        if not isinstance(rows, list) or not isinstance(rows[0], ADGRow):
            raise TypeError("Expects an object of type ADGRow")


        # create dictionary with entity-types in the trainingsdata
        all_labels = list(set(label for row in rows for label in row.labels))
        label_id = {label: i for i, label in enumerate(all_labels)}

        data = None
        # change statements to sentences and create the dataset
        if split_sentences:
            sentences_tokens, sentences_labels = data_registry.split_training_data_sentences(rows)
            data = Dataset.from_list(
                [{"tokens": sen_tokens, "labels": [label_id[label] for label in sentences_labels[i+1]]} for i, sen_tokens in enumerate(sentences_tokens[1:])])
        else:
            # create dataset from traingsdata
            data = Dataset.from_list([{"tokens": row.tokens, "labels": [label_id[label] for label in row.labels]} for row in rows[1:]])

        # tokenize training data with word ids
        tokenized_data = data.map(lambda x: self._tokenize_and_align_labels(x,tokenizer_path), batched=True)

        # is validation needed?
        split_test = tokenized_data.train_test_split(test_size=test_size, seed=42)
        test_data = split_test["test"]
        train_data = split_test["train"]

        split_validation = train_data.train_test_split(test_size=validation_size/(1-test_size), seed=42)
        validation_data = split_validation["test"]
        train_data = split_validation["train"]

        dataset_dict = DatasetDict({"train":train_data, "validation":validation_data, "test":test_data})
        return dataset_dict, label_id


    def finetune_ner_model(self,base_model_path,data_dict, label_id, name,new_model_path,params=None):
        if params is None:
            params = self.default_finetuning_params

        # dictionary which maps the ids to the labels
        id_label = {value: key for key, value in label_id.items()}
        # get a list of all labels
        sorted_label_id = dict(sorted(label_id.items(), key=lambda item: item[1]))
        list_labels = list(sorted_label_id.keys())

        # init base tokenzier and base models
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model = AutoModelForTokenClassification.from_pretrained(base_model_path,num_labels=len(list_labels),id2label=id_label, label2id=label_id,ignore_mismatched_sizes=True)

        # best modell is saved in directory as checkpoint-xx, then loaded in the trainer (load_best_model_at_the_end)
        # overwrite the dict manually
        training_args = TrainingArguments(
            output_dir=new_model_path,
            evaluation_strategy=params["evaluation_strategy"],
            save_strategy=params["save_strategy"],
            save_total_limit=params["save_total_limit"],
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            num_train_epochs=params["num_train_epochs"],
            weight_decay=params["weight_decay"],
            load_best_model_at_end=params["load_best_model_at_end"],
            metric_for_best_model=params["metric_for_best_model"],
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_dict["train"],
            eval_dataset=data_dict["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: self._compute_metrics(p, list_labels),
        )

        train_results =trainer.train()
        metrics = trainer.evaluate()
        args = trainer.args.to_dict()
        trainer.save_model(new_model_path)
        delete_checkpoints_folder(new_model_path)
        print("Training done")
        print(metrics)
        return self._convert_metrics(metrics,train_results.metrics["train_runtime"]), args


    # split function
    # add a less strict type comparison, without B- and I-
    def convert_ner_results(self,ner_results, ner_input, annoted_labels=None):
        if isinstance(ner_input[0], ADGRow):
            return self._convert_ner_results_adg(ner_results, ner_input)
        else:
            tokens, predicted_labels = self._convert_ner_results_not_adg(ner_results, ner_input)
            return tokens, predicted_labels, None

    def _convert_ner_results_adg(self, ner_results, ner_input):
        metrics = None
        tokens = []
        predicted_labels = []
        annoted_labels = []
        for index, result in enumerate(ner_results):
            adg_row = ner_input[index]
            labels_row = ["O"] * len(adg_row.labels)
            # map the entities to labels_row
            for entity in result:
                if entity["word"][:2] != '##':  #
                    try:
                        index_label = -1
                        # search the index of the corresponding token in adgrow
                        if entity["start"] in adg_row.indexes:
                            index_label = adg_row.indexes.index(entity["start"])
                        else:
                            # if only a part of the word is recognized as entity -> apply to full word
                            for index, index_val in enumerate(adg_row.indexes):
                                if index_val > entity["start"]:
                                    index_label = index - 1
                                    break
                        labels_row[index_label] = entity["entity"]
                    except ValueError:
                        print("Error by assigning the predicted labels")
            predicted_labels.append(labels_row)
            tokens.append(adg_row.tokens)
            annoted_labels.append(adg_row.labels)

        # calc metrics
        metrics = self._calc_metrics(annoted_labels,predicted_labels)
        return tokens, predicted_labels, metrics

    def _convert_ner_results_not_adg(self, ner_results, ner_input):
        tokens = []
        predicted_labels = []
        for index_sentence, sentence in enumerate(ner_input):

            # prepare tokens for return
            tokens_hf = self.tokenizer(sentence, return_offsets_mapping=True)
            sub_tokens = tokens_hf.tokens()[1:-1]
            sub_startindexes = [index[0] for index in tokens_hf["offset_mapping"]][1:-1]

            tokens_sentence = []
            startindexes = []
            last_append_index = 0
            for i in range(0, len(sub_tokens)):
                # append subtoken to last real token
                if sub_tokens[i][:2] == "##":
                    tokens_sentence[last_append_index] += sub_tokens[i][2:]
                # append real token
                else:
                    tokens_sentence.append(sub_tokens[i])
                    last_append_index = len(tokens_sentence) - 1
                    startindexes.append(sub_startindexes[i])

            labels_sentence = ["O"] * len(tokens_sentence)
            # prepare recognized entities
            for entities in ner_results[index_sentence]:
                if entities["word"][0:2] != "##":
                    # print(entities)
                    # print(startindexes)
                    index_entity = startindexes.index(entities["start"])
                    # print(tokens_sentence[index_entity])
                    labels_sentence[index_entity] = entities["entity"]

            tokens.append(tokens_sentence)
            predicted_labels.append(labels_sentence)
        return tokens, predicted_labels

    #from https: // huggingface.co / docs / transformers / tasks / token_classification
    def _tokenize_and_align_labels(self, statement, tokenizer_path):
        """
        Map the tokens to token-ids and give the subtokens -100 (they don't influence the model in the learning process, no impact on interference)

        Parameters:
        statement (dict): contains the word tokens
        tokenizer_path (str): path to the tokenizer

        Returns:
        Dateset: With "tokens", "labels" - contains the token-ids and the -100, "input_ids", "token_type_ids", "attention_mask"
        """
        model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenized_inputs = model_tokenizer(statement["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(statement[f"labels"]):
            # Map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            # Set the special tokens to -100.
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    '''
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
    '''

    def _compute_metrics(self, p, label_list):
        seqeval = load("seqeval")
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

    # make this for all framework classes abstract
    def _convert_metrics(self, metrics, duration):
        print(duration)
        return TrainingResults(f1=metrics["eval_f1"], recall=metrics["eval_recall"], precision=metrics["eval_precision"], duration=duration, accuracy=metrics["eval_accuracy"])
