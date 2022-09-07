from transformers import DataCollatorWithPadding, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import os
import numpy as np


def prepare_sample_sequence(bio_file):
    tags = []
    text = []
    prev = '[CLS]'
    for ln in bio_file:
        line = ln.split()
        if line[1] == 'B':
            tags.append(0)
        elif line[1] == 'I':
            tags.append(1)
        elif line[1] == 'O':
            tags.append(2)
        else:
            # E(nd) label treated as I
            tags.append(1)
        text.append(prev+' '+line[0])
        prev = line[0]
    return tags, text


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

        if len(tags) == 5:
            samples_tags.append(tags)
            tags = []
            samples_tokens.append(tokens)
            tokens = []

    if len(tags) > 0:
        samples_tags.append(tags)
        samples_tokens.append(tokens)

    return samples_tags, samples_tokens


def load_dataset(mode, flag):
    data = {'train': {}, 'dev': {}, 'test': {}}
    if mode == 'Sequence':
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
        if mode == 'Sequence':
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
    model = AutoModelForTokenClassification.from_pretrained('projecte-aina/roberta-base-ca-v2', num_labels=n_lb, ignore_mismatched_sizes=True)

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
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        num_train_epochs=20,
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


def predictions_output_sequence(preds):
    prediction_file = open('out/predict.txt', 'w+')

    for j in range(len(preds)):

        if preds[j] == 0:
            label = 'B'
        elif preds[j] == 1:
            label = 'I'
        elif preds[j] == 2:
            label = 'O'
        else:
            label = 'E'

        text = tokenized_data['test']['text'][j]
        token = text.split()[-1]

        prediction_file.write(token + ' ' + label + '\n')


def predictions_output_token(preds, only_seg):
    prediction_file = open('out/predict.txt', 'w+')

    for j in range(len(preds)):
        # Data batch
        ready2write = False
        write_buffer = ''
        label_buffer = ''
        for k in range(len(tokenized_data['test']['input_ids'][j])):
            # Each token in batch

            token = tknz.convert_ids_to_tokens(tokenized_data['test']['input_ids'][j][k])

            write_buffer += token

            if k < len(tokenized_data['test']['input_ids'][j])-1:
                next_token = tknz.convert_ids_to_tokens(tokenized_data['test']['input_ids'][j][k+1])

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
    mode = 'Token'
    only_segmentation = True
    num_labels = 2
    dataset = load_dataset(mode, only_segmentation)

    tknz, mdl = load_model(num_labels)

    if mode == 'Sequence':
        tokenized_data = dataset.map(tokenize_sequence, batched=True)
        # tokenized_data = tokenized_data.remove_columns("text")
        data_collator = DataCollatorWithPadding(tokenizer=tknz)
    else:
        tokenized_data = dataset.map(tokenize_token, batched=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tknz)

    trainer = train_model(mdl, tknz, tokenized_data)
    predictions = trainer.predict(tokenized_data['test'])
    predict = np.argmax(predictions.predictions, axis=-1)

    if mode == 'Sequence':
        predictions_output_sequence(predict)
    else:
        predictions_output_token(predict, only_segmentation)
