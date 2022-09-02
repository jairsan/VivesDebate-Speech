import sys
from typing import List, Tuple
import numpy as np

import torch

from train_segment_classifier import generate_dataset, KEEP, DISCARD
from transformers import Pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,\
    TokenClassificationPipeline
from transformers import Trainer, TrainingArguments
import evaluate

class SegmentsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def generate_and_punct_split_dataset(document_name_list, pip: Pipeline) -> Tuple[List[str], List[int]]:
    samples, labels = generate_dataset(document_name_list=document_name_list)
    new_samples: List[str] = []
    new_labels: List[int] = []
    for sample, label in zip(samples, labels):
        if label == KEEP:
            new_samples.append(sample)
            new_labels.append(label)
        else:
            entities = pip(sample, aggregation_strategy="simple")
            curr_sent: str = ""
            for entity in entities:
                group = entity["entity_group"]
                curr_sent += entity["word"].strip() + " "
                if group == "." or group == "?" or group == ":":
                    new_samples.append(curr_sent.strip())
                    new_labels.append(DISCARD)
                    curr_sent = ""

            if len(curr_sent) > 0:
                new_samples.append(curr_sent.strip())
                new_labels.append(DISCARD)

    return new_samples, new_labels


def train_model(model_name: str, train_files: List[str], eval_files: List[str], output_dir_name: str):
    punct_name = "softcatala/fullstop-catalan-punctuation-prediction"
    punct_model = AutoModelForTokenClassification.from_pretrained(punct_name)
    punct_tokenizer = AutoTokenizer.from_pretrained(punct_name)
    pipeline = TokenClassificationPipeline(model=punct_model, tokenizer=punct_tokenizer)

    train_samples, train_labels = generate_and_punct_split_dataset(train_files, pip=pipeline)
    eval_samples, eval_labels = generate_and_punct_split_dataset(eval_files, pip=pipeline)

    # print("train", train_samples[-10:], train_labels[-10:])

    classifier_tknzr = AutoTokenizer.from_pretrained(model_name)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_encodings = classifier_tknzr(train_samples, truncation=True, padding=True)
    eval_encodings = classifier_tknzr(eval_samples, truncation=True, padding=True)

    train_dataset = SegmentsDataset(encodings=train_encodings, labels=train_labels)
    eval_dataset = SegmentsDataset(encodings=eval_encodings, labels=eval_labels)

    def compute_metrics(eval_preds):
        f1 = evaluate.load("f1")
        acc = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        # print("[II] predictions", predictions[-10:])
        # print("[II] labels", labels[-10:])
        return {"f1": f1.compute(predictions=predictions, references=labels), "acc": acc.compute(predictions=predictions, references=labels)}

    training_args = TrainingArguments(
        output_dir=output_dir_name + "_models",  # output directory
        num_train_epochs=2,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=250,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=output_dir_name + "_logs",  # directory for storing logs
        logging_steps=10,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=125
    )

    trainer = Trainer(
        model=classifier_model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    print("Finished preparing training")
    trainer.train()


if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4])
