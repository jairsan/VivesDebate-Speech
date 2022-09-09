from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

import torch

from train_segment_classifier import generate_dataset, KEEP, DISCARD
from transformers import Pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification,\
    TokenClassificationPipeline
from transformers import Trainer, TrainingArguments, HfArgumentParser
import evaluate
import argparse


@dataclass
class TrainingArgs:
    learning_rate: float = field(default=5e-5)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    per_device_eval_batch_size: int = field(default=16)
    num_train_epochs: int = field(default=3)


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


def generate_dataset_from_spans(document_name_list: List[str],
                                span_folder: str) -> Tuple[List[str], List[int]]:

    samples, labels = generate_dataset(document_name_list=document_name_list)

    samples_tokens_count = 0
    for sample in samples:
        samples_tokens_count += len(sample.strip().split())

    per_token_labels: List[int] = []

    for sample, label in zip(samples, labels):
        per_token_labels.extend([label] * len(sample.strip().split()))

    assert len(per_token_labels) == samples_tokens_count

    new_samples: List[str] = []
    new_labels: List[int] = []

    spans: List[List[str]] = []

    for filename in document_name_list:
        filename_raw=filename.split("/")[-1].split(".")[0]
        with open(span_folder + "/" + filename_raw + ".spans") as span_file:
            for line in span_file:
                spans.append(line.strip().split())

    span_tokens_count = 0
    for span in spans:
        span_tokens_count += len(span)

    assert span_tokens_count == samples_tokens_count

    for span in spans:
        new_samples.append(" ".join(span))
        this_labels = per_token_labels[:len(span)]

        tokens_I = this_labels.count(KEEP)
        tokens_O = len(this_labels) - tokens_I

        if tokens_O > tokens_I:
            label = DISCARD
        else:
            label = KEEP

        new_labels.append(label)

        per_token_labels = per_token_labels[len(span):]

    assert len(per_token_labels) == 0
    assert len(new_samples) == len(new_labels)

    return new_samples, new_labels


def train_model(model_name: str, train_files: List[str], eval_files: List[str], output_dir_name: str,
                generate_train_datasets_from_span_folder: str, generate_eval_datasets_from_span_folder: str,
                training_args: TrainingArgs):

    punct_name = "softcatala/fullstop-catalan-punctuation-prediction"
    punct_model = AutoModelForTokenClassification.from_pretrained(punct_name)
    punct_tokenizer = AutoTokenizer.from_pretrained(punct_name)
    pipeline = TokenClassificationPipeline(model=punct_model, tokenizer=punct_tokenizer)

    if generate_train_datasets_from_span_folder is not None:
        print("Generating train dataset from spans...")
        train_samples, train_labels = generate_dataset_from_spans(document_name_list=train_files,
                                                                  span_folder=generate_train_datasets_from_span_folder)
    else:
        train_samples, train_labels = generate_and_punct_split_dataset(train_files, pip=pipeline)

    if generate_eval_datasets_from_span_folder is not None:
        print("Generating dev dataset from spans...")
        eval_samples, eval_labels = generate_dataset_from_spans(document_name_list=eval_files,
                                                                span_folder=generate_eval_datasets_from_span_folder)
    else:
        eval_samples, eval_labels = generate_and_punct_split_dataset(eval_files, pip=pipeline)

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
        return {"f1": f1.compute(predictions=predictions, references=labels), "acc": acc.compute(predictions=predictions, references=labels)}

    training_args = TrainingArguments(
        output_dir=output_dir_name + "_models",  # output directory
        overwrite_output_dir=True,
        num_train_epochs=training_args.num_train_epochs,  # total number of training epochs
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,  # batch size per device during training
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=250,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_steps=10,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=classifier_model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    classifier_tknzr.save_pretrained(output_dir_name + "_tokenizer")
    print("Finished preparing training")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_files', type=str, required=True)
    parser.add_argument('--eval_files', type=str, required=True)
    parser.add_argument('--output_dir_name', type=str, required=True)
    parser.add_argument('--generate_train_datasets_from_spans_folder', type=str)
    parser.add_argument('--generate_eval_datasets_from_spans_folder', type=str)

    args, remaining_args = parser.parse_known_args()

    hf_parser = HfArgumentParser([TrainingArgs])

    output = hf_parser.parse_args_into_dataclasses(args=remaining_args)

    training_args_dc = output[0]

    train_model(model_name=args.model_name, train_files=args.train_files.split(), eval_files=args.eval_files.split(),
                output_dir_name=args.output_dir_name,
                generate_train_datasets_from_span_folder=args.generate_train_datasets_from_spans_folder,
                generate_eval_datasets_from_span_folder=args.generate_eval_datasets_from_spans_folder,
                training_args=training_args_dc)
