from transformers import DataCollatorWithPadding, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import os
import numpy as np


def prepare_sample_sequence(bio_file):
    tags = []
    text = []
    sample = ''
    prev_lbl = ''
    for ln in bio_file:
        line = ln.split()
        if line[1] == 'B':
            if len(sample) > 0 and prev_lbl == 'O':
                text.append(sample)
                tags.append(0)
                sample = ''
            sample += ' '+line[0]
            prev_lbl = line[1]
        elif line[1] == 'I':
            sample += ' '+line[0]
            prev_lbl = line[1]
        elif line[1] == 'O':
            sample += ' '+line[0]
            prev_lbl = line[1]
        else:
            # E(nd) label treated as I
            tags.append(1)
            sample += ' '+line[0]
            text.append(sample)
            sample = ''
            prev_lbl = line[1]
    if len(sample) > 0 and prev_lbl == 'O':
        text.append(sample)
        tags.append(0)

    return tags, text


def prepare_sample_segmented_sequence(bio_file):
    text = []
    sample = ''
    for ln in bio_file:
        line = ln.split()
        if line[1] == 'B':
            if len(sample) > 0:
                text.append(sample)
            sample = ''
            sample += ' '+line[0]
        else:
            # X label
            sample += ' '+line[0]

    if len(sample) > 0:
        text.append(sample)

    return text


def prepare_samples_token(bio_file, only_seg):
    samples_tags = []
    samples_tokens = []
    tags = []
    tokens = []
    prev_tag = ''
    for ln in bio_file:
        line = ln.split()
        tokens.append(line[0])

        if line[1] == 'B':
            prev_tag = 'B'
            if only_seg:
                tags.append(1)
            else:
                tags.append(0)

        elif line[1] == 'I':
            prev_tag = 'I'
            if only_seg:
                tags.append(0)
            else:
                tags.append(1)

        elif line[1] == 'O':
            if only_seg and prev_tag == 'E':
                tags.append(1)
            elif only_seg:
                tags.append(0)
            else:
                tags.append(2)

            prev_tag = 'O'
        else:
            # E(nd) label treated as I
            if only_seg:
                tags.append(0)
            else:
                tags.append(1)
            prev_tag = 'E'

        if len(tags) == 25:
            samples_tags.append(tags)
            tags = []
            samples_tokens.append(tokens)
            tokens = []

    if len(tags) > 0:
        samples_tags.append(tags)
        samples_tokens.append(tokens)

    return samples_tags, samples_tokens


def load_segmented_dataset():
    data = {'dev': {}, 'test': {}}

    data['dev']['text'] = []
    data['test']['text'] = []

    for file in os.listdir("out/Segmentation/25/"):
        bio_file = open('out/Segmentation/25/' + file, 'r')

        if file == 'dev_hypothesis.txt':
            text = prepare_sample_segmented_sequence(bio_file)
            for c in range(len(text)):
                data['dev']['text'].append(text[c])

        elif file == 'test_hypothesis.txt':
            text = prepare_sample_segmented_sequence(bio_file)
            for c in range(len(text)):
                data['test']['text'].append(text[c])

    full_data = DatasetDict()
    # using your `Dict` object
    for k, v in data.items():
        full_data[k] = Dataset.from_dict(v)

    return full_data


def load_dataset(mode, flag):
    data = {'train': {}, 'dev': {}, 'test': {}}
    if mode == 'T-Sequence' or mode == 'P-Sequence':
        data['dev']['label'] = []
        data['dev']['text'] = []
        data['test']['label'] = []
        data['test']['text'] = []
        data['train']['label'] = []
        data['train']['text'] = []
    else:
        data['dev']['tags'] = []
        data['dev']['tokens'] = []
        data['test']['tags'] = []
        data['test']['tokens'] = []
        data['train']['tags'] = []
        data['train']['tokens'] = []

    for file in os.listdir("BIO_arg/"):
        bio_file = open('BIO_arg/' + file, 'r')
        if mode == 'T-Sequence' or mode == 'P-Sequence':
            tags, text = prepare_sample_sequence(bio_file)
            if file == 'Debate24.txt' or file == 'Debate25.txt' or file == 'Debate26.txt':
                for c in range(len(tags)):

                    data['dev']['label'].append(tags[c])
                    data['dev']['text'].append(text[c])

            elif file == 'Debate27.txt' or file == 'Debate28.txt' or file == 'Debate29.txt':
                for c in range(len(tags)):

                    data['test']['label'].append(tags[c])
                    data['test']['text'].append(text[c])

            else:
                for c in range(len(tags)):
                    data['train']['label'].append(tags[c])
                    data['train']['text'].append(text[c])
        else:
            tags, text = prepare_samples_token(bio_file, flag)
            if file == 'Debate24.txt' or file == 'Debate25.txt' or file == 'Debate26.txt':
                for c in range(len(tags)):
                    data['dev']['tags'].append(tags[c])
                    data['dev']['tokens'].append(text[c])

            elif file == 'Debate27.txt' or file == 'Debate28.txt' or file == 'Debate29.txt':
                for c in range(len(tags)):
                    data['test']['tags'].append(tags[c])
                    data['test']['tokens'].append(text[c])

            else:
                for c in range(len(tags)):
                    data['train']['tags'].append(tags[c])
                    data['train']['tokens'].append(text[c])

    full_data = DatasetDict()
    # using your `Dict` object
    for k, v in data.items():
        full_data[k] = Dataset.from_dict(v)

    return full_data


def tokenize_sequence(samples):
    return tknz(samples["text"], padding=True, truncation=True)


def tokenize_token(samples):
    tokenized_inputs = tknz(samples["tokens"], padding=True, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(samples["tags"]):
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


def load_model(n_lb):
    tokenizer_hf = AutoTokenizer.from_pretrained('projecte-aina/roberta-base-ca-v2')
    if mode == 'T-Sequence' or mode == 'P-Sequence':
        model = AutoModelForSequenceClassification.from_pretrained('projecte-aina/roberta-base-ca-v2', num_labels=2,
                                                                   ignore_mismatched_sizes=True)
    else:
        model = AutoModelForTokenClassification.from_pretrained('projecte-aina/roberta-base-ca-v2', num_labels=n_lb,
                                                            ignore_mismatched_sizes=True)

    return tokenizer_hf, model


def load_model_trained(path):
    if mode == 'T-Sequence' or mode == 'P-Sequence':
        tokenizer_hf = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        tokenizer_hf = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path)

    return tokenizer_hf, model


def train_model(model, tokenizer, data):

    training_args = TrainingArguments(
        output_dir="models",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        per_device_train_batch_size=50,
        per_device_eval_batch_size=50,
        num_train_epochs=50,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return trainer


def predictions_output_sequence(preds, partition):
    prediction_file = open('out/'+partition+'_predict.txt', 'w+')

    for j in range(len(preds)):

        text = tokenized_data[partition]['text'][j].split()

        if preds[j] == 1:
            label = 'B'
            for word in text:
                prediction_file.write(word + ' ' + label + '\n')
                label = 'I'

        elif preds[j] == 0:
            label = 'O'
            for word in text:
                prediction_file.write(word + ' ' + label + '\n')


def predictions_output_token(preds, partition, only_seg):
    prediction_file = open('out/'+partition+'_predict.txt', 'w+')

    for j in range(len(preds)):
        # Data batch
        ready2write = False
        write_buffer = ''
        label_buffer = ''
        for k in range(len(tokenized_data[partition]['input_ids'][j])):
            # Each token in batch

            token = tknz.convert_ids_to_tokens(tokenized_data[partition]['input_ids'][j][k])

            write_buffer += token

            if k < len(tokenized_data[partition]['input_ids'][j])-1:
                next_token = tknz.convert_ids_to_tokens(tokenized_data[partition]['input_ids'][j][k+1])

                if next_token[0] == 'Ġ' or next_token[0] == '<':
                    ready2write = True

            else:
                ready2write = True

            if token[0] == '<':
                ready2write = False
                write_buffer = ''
                label_buffer = ''
                pass

            if preds[j][k] == 0:
                label = 'B'
                if only_seg:
                    label = 'X'
            elif preds[j][k] == 1:
                label = 'I'
                if only_seg:
                    label = 'B'
            elif preds[j][k] == 2:
                label = 'O'
            else:
                label = 'I'

            label_buffer += label

            # print(token, label)

            if ready2write:
                # print('Write:', write_buffer[1:], label_buffer)
                prediction_file.write(write_buffer[1:]+' '+label+'\n')
                write_buffer = ''
                label_buffer = ''
                ready2write = False
            # print(tknz.convert_ids_to_tokens(tokenized_data['test']['input_ids'][j][k]), preds[j][k])


if __name__ == "__main__":
    # T-Token, P-Token, T-Sequence, P-Sequence
    mode = 'P-Token'
    # Activate only segmentation for T-Token and P-Token modes to detect argumentative spans instead of BIO tags.
    only_segmentation = True

    num_labels = 3
    if only_segmentation or mode == 'T-Sequence' or mode == 'P-Sequence':
        num_labels = 2

    if mode == 'P-Sequence':
        dataset = load_segmented_dataset()
    else:
        dataset = load_dataset(mode, only_segmentation)

    # Predict Model
    if mode == 'P-Sequence':
        tknz, mdl = load_model_trained('models/Classifier/checkpoint-429')
    elif mode == 'P-Token' and only_segmentation:
        tknz, mdl = load_model_trained('models/Segmentation/25/checkpoint-536')
    elif mode == 'P-Token' and only_segmentation is False:
        tknz, mdl = load_model_trained('models/BIO/5/checkpoint-252')
    # Train Model
    else:
        tknz, mdl = load_model(num_labels)

    # Data prepared for Sequence inputs
    if mode == 'T-Sequence' or mode == 'P-Sequence':
        tokenized_data = dataset.map(tokenize_sequence, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tknz)
    # Data prepared for Token inputs
    else:
        tokenized_data = dataset.map(tokenize_token, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tknz)

    # Predict Model
    if mode == 'P-Token' or mode == 'P-Sequence':
        trainer = Trainer(mdl)
    # Train Model
    else:
        trainer = train_model(mdl, tknz, tokenized_data)

    dev_predictions = trainer.predict(tokenized_data['dev'])
    dev_predict = np.argmax(dev_predictions.predictions, axis=-1)
    test_predictions = trainer.predict(tokenized_data['test'])
    test_predict = np.argmax(test_predictions.predictions, axis=-1)

    if mode == 'T-Sequence' or mode == 'P-Sequence':
        predictions_output_sequence(dev_predict, 'dev')
        predictions_output_sequence(test_predict, 'test')
    else:
        predictions_output_token(dev_predict, 'dev', only_segmentation)
        predictions_output_token(test_predict, 'test', only_segmentation)
