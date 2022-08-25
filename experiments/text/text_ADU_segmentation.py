from transformers import TokenClassificationPipeline, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import os
import numpy as np


def prepare_sample_sequence(bio_file):
    samples = []
    tags = []
    text = []
    prev = '[CLS]'
    prev_label = 'N'
    for ln in bio_file:
        line = ln.split()
        tags.append(line[1])
        text.append(prev+' '+'['+prev_label+']'+' '+line[0])
        prev = line[0]
        prev_label = line[1]
    return tags, text


def prepare_samples_token(bio_file, file):
    samples_tags = []
    samples_tokens = []
    tags = []
    tokens = []
    for ln in bio_file:
        line = ln.split()
        tokens.append(line[0])

        if line[1] == 'B':
            tags.append(0)
        elif line[1] == 'I':
            tags.append(1)
        elif line[1] == 'O':
            tags.append(2)
        else:
            tags.append(3)
            if len(tags) >= 75:
                samples_tags.append(tags)
                tags = []
                samples_tokens.append(tokens)
                tokens = []
    samples_tags.append(tags)
    samples_tokens.append(tokens)

    return samples_tags, samples_tokens


def load_dataset(mode):
    data = {'train': {}, 'dev': {}, 'test': {}}
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
        else:
            tags, text = prepare_samples_token(bio_file, file)
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


def tokenize_and_align_labels(samples):
    tokenized_inputs = tknz(samples["tokens"], max_length=512, padding=True, truncation=False, is_split_into_words=True)
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


def load_model(mode):
    if mode == 'Sequence':
        tokenizer_hf = AutoTokenizer.from_pretrained('projecte-aina/roberta-base-ca-v2')
        model = AutoModelForSequenceClassification.from_pretrained('projecte-aina/roberta-base-ca-v2', num_labels=4)
    else:
        tokenizer_hf = AutoTokenizer.from_pretrained('projecte-aina/roberta-base-ca-v2-cased-pos')
        model = AutoModelForTokenClassification.from_pretrained('projecte-aina/roberta-base-ca-v2-cased-pos', num_labels=4, ignore_mismatched_sizes=True)
    return tokenizer_hf, model


def train_model(model, tokenizer, data):

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
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


def predictions_output(preds):
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

            if k < len(preds[j])-1:
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
            elif preds[j][k] == 1:
                label = 'I'
            elif preds[j][k] == 2:
                label = 'O'
            else:
                label = 'E'

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
    dataset = load_dataset(mode)

    tknz, mdl = load_model(mode)

    # tokenized_data = dataset.map(tokenize_text, batched=True)
    tokenized_data = dataset.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tknz, max_length=512)

    trainer = train_model(mdl, tknz, tokenized_data)
    predictions = trainer.predict(tokenized_data['test'])
    predict = np.argmax(predictions.predictions, axis=-1)

    predictions_output(predict)


