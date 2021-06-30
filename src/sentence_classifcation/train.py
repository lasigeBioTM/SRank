from __future__ import annotations

import json

import torch
import transformers  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    set_seed,
)


def tokenize(tokenizer, question, sentence, label):
    tokens = tokenizer(
        question,
        sentence,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return {
        'input_ids': tokens['input_ids'].squeeze(),
        'token_type_ids': tokens['token_type_ids'].squeeze(),
        'attention_mask': tokens['attention_mask'].squeeze(),
        'labels': torch.FloatTensor([label]),
    }


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: transformers.BertTokenizer,
        positive_pairs: tuple[str, str],
        negative_pairs: tuple[str, str]
    ) -> None:
        self.tokenizer = tokenizer

        self.data = [
            (question, sentence, 1)
            for question, sentence in positive_pairs
        ] + [
            (question, sentence, 0)
            for question, sentence in negative_pairs
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return tokenize(self.tokenizer, *self.data[idx])


def read_json(filename):
    with open(filename) as f:
        return json.load(f)


tokenizer = AutoTokenizer.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.1',
    do_lower_case=False,
)
model = AutoModelForSequenceClassification.from_pretrained(
    'dmis-lab/biobert-base-cased-v1.1',
    num_labels=1,
)

DIR = 'data/'

positive_train = read_json(DIR + 'question-sentence-positive-pairs.train.json')
positive_test = read_json(DIR + 'question-sentence-positive-pairs.test.json')
total_positive = len(positive_train) + len(positive_test)

negative_instances = read_json(DIR + 'question-sentence-negative-pairs.json')

negative_train_size = int(len(positive_train) / total_positive * len(negative_instances))

negative_train = negative_instances[:negative_train_size]
negative_test = negative_instances[negative_train_size:]

print(f'{len(positive_train)} positive training samples')
print(f'{len(negative_train)} negative training samples')

print(f'{len(positive_test)} positive testing samples')
print(f'{len(negative_test)} negative testing samples')

import random

random.seed(42)

random.shuffle(negative_train)
random.shuffle(positive_test)
random.shuffle(negative_test)

def sample(l, ratio):
    random.shuffle(positive_train)

    return l[:int(ratio * len(l))]

positive_train_sample = sample(positive_train, 0.4)
negative_train_sample = sample(negative_train, 0.4)

positive_test_sample = sample(positive_test, 0.1)
negative_test_sample = sample(negative_test, 0.1)

train_dataset = Dataset(tokenizer, positive_train_sample, negative_train_sample)
test_dataset = Dataset(tokenizer, positive_test_sample, negative_test_sample)

training_args = transformers.TrainingArguments(
    output_dir='models/qs-model/',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    eval_steps=200,
    per_device_train_batch_size=6,  # TODO: These may change
    per_device_eval_batch_size=6,  # TODO: These may change
    learning_rate=5e-05,
    num_train_epochs=6,
    logging_strategy='steps',
    logging_steps=20,
    save_strategy='steps',
    save_steps=1000,
    seed=42,
    fp16=True,
)

set_seed(training_args.seed)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model()
