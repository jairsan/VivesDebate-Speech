import sys
from typing import List, Tuple

from train_segment_classifier import generate_dataset, KEEP, DISCARD
from transformers import Pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline


def generate_and_punct_split_dataset(document_name_list, pip: Pipeline) -> Tuple[List[str], List[int]]:
    samples, labels = generate_dataset(document_name_list=document_name_list)
    new_samples: List[str] = []
    new_labels: List[int] = []

    for sample, label in zip(samples, labels):
        if label == KEEP:
            new_samples.append(sample)
            new_labels.append(label)
        else:
            entities = pip(sample, aggregation_strategy="first")
            curr_sent: str = ""
            for entity in entities:
                group = entity["entity_group"]
                curr_sent += " " + entity["word"]
                if group == "." or group == "?" or group == ":":
                    new_samples.append(curr_sent)
                    new_labels.append(DISCARD)
                    curr_sent = ""

            if len(curr_sent) > 0:
                new_samples.append(curr_sent)
                new_labels.append(DISCARD)

    return new_samples, new_labels


def train_model(model_name, train_files, eval_files):
    punct_name = "softcatala/fullstop-catalan-punctuation-prediction"
    punct_model = AutoModelForTokenClassification.from_pretrained(punct_name)
    punct_tokenizer = AutoTokenizer.from_pretrained(punct_name)
    pipeline = TokenClassificationPipeline(model=punct_model, tokenizer=punct_model)

    train_samples, train_labels = generate_and_punct_split_dataset(train_files,pip=pipeline)
    eval_samples, eval_labels = generate_and_punct_split_dataset(eval_files,pip=pipeline)

    print(eval_samples[-5:], eval_labels[-5:])


if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2], sys.argv[3])
    #model_name = "softcatala/fullstop-catalan-punctuation-prediction"
    #model = AutoModelForTokenClassification.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
    #print(pipeline("Els investigadors suggereixen que tot i que es tracta de la cua d'un dinosaure jove la mostra revela un plomatge adult i no pas plomissol", aggregation_strategy="first"))
