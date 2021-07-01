from __future__ import annotations

import argparse
import json
import random
import typing

import torch
from tqdm.auto import tqdm
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


class Tokens(typing.TypedDict):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class QuestionSentencePair(typing.TypedDict):
    question: str
    sentence: str
    label: str


def tokenize(
    tokenizer: PreTrainedTokenizer,
    question: str,
    sentence: str,
    label: int
) -> Tokens:
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
        'labels': torch.FloatTensor([label]),  # type: ignore
    }


class Dataset(torch.utils.data.Dataset[Tokens]):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        data: list[QuestionSentencePair]
    ) -> None:
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tokens:
        instance = self.data[idx]

        return tokenize(
            self.tokenizer,
            question=instance['question'],
            sentence=instance['sentence'],
            label=1 if instance['label'] == 'positive' else 0,
        )


def get_arguments() -> argparse.Namespace:
    args = argparse.Namespace()

    args.transformer_model = 'dmis-lab/biobert-base-cased-v1.1'
    args.input = 'results/qs-train-dataset.json'
    args.seed = 42
    args.epochs = 2
    args.train_ratio = 0.9

    return args


def main() -> None:
    args = get_arguments()

    tokenizer = AutoTokenizer.from_pretrained(
        args.transformer_model,
        do_lower_case=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.transformer_model,
        num_labels=1,
    )

    with open(args.input) as f:
        data = json.load(f)

    training_size = int(args.train_ratio * len(data))

    rand = random.Random(args.seed)
    rand.shuffle(data)

    train_dataset = Dataset(tokenizer, data[:training_size])
    test_dataset = Dataset(tokenizer, data[training_size:])

    training_args = TrainingArguments(
        output_dir='models/qs-model/',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=200,
        per_device_train_batch_size=6,  # TODO: These may change
        per_device_eval_batch_size=6,  # TODO: These may change
        learning_rate=5e-05,
        num_train_epochs=args.epochs,
        logging_strategy='steps',
        logging_steps=20,
        save_strategy='steps',
        save_steps=1000,
        seed=args.seed,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    trainer.save_model()


if __name__ == '__main__':
    main()
