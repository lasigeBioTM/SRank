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
        'labels': torch.FloatTensor([label]),
    }


class Dataset(torch.utils.data.Dataset[Tokens]):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
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
    parser = argparse.ArgumentParser(
        description='Train a transformer to recognize when a sentence is '
                    'related to a question.'
    )

    parser.add_argument(
        'input', metavar='INPUT',
        help='The file that contains the question/sentence pairs. This should '
             'be an output of the `src/sentence_classification/merge_qs.py` '
             'script.'
    )

    parser.add_argument(
        '-m', '--model', default='dmis-lab/biobert-base-cased-v1.1',
        help='A string that identifies the hugging-face transformer model that '
             'will be used as the base of the learned model. Bu default, this '
             'is the BioBERT model of the DMIS lab, which is case sensitive.'
    )

    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='The seed to initialize the randomization of the script. '
             'Defaults to 42.'
    )

    parser.add_argument(
        '-n', '--epochs', type=int, default=2,
        help='The number of epoch to run the training phase. Defaults to 2.'
    )

    parser.add_argument(
        '-r', '--train-ratio', type=float, default=0.9,
        help='The fraction of the dataset to use as training; the rest will be '
             'used as testing data to monitor when the loss stops decreasing. '
             'Defaults to 0.9.'
    )

    parser.add_argument(
        '-o', '--output-dir', default='models/qs-model/',
        help='The directory to store the intermediate weights throughout the '
             'training. This directory must not exist.'
    )

    parser.add_argument(
        '--learning-rate', type=float, default=5e-5,
        help='The learning rate to use in the training phase. Because the '
             'training is done with a decaying learning rate and an initial '
             'warm up interval, this is not the actual leraning rate used '
             'througout the training, but instead the maximum value. Defaults '
             'to 5e-5.'
    )

    parser.add_argument(
        '--eval-steps', type=int, default=200,
        help='The number of steps between each validation measurement.'
    )

    parser.add_argument(
        '--per-device-train-batch-size', type=int, default=6,
        help='The batch size for the training phase'
    )

    parser.add_argument(
        '--per-device-eval-batch-size', type=int, default=6,
        help='The batch size for the validation phase'
    )

    parser.add_argument(
        '--logging-steps', type=int, default=20,
        help='The number of steps between each log. Defaults to 20.'
    )

    parser.add_argument(
        '--save-steps', type=int, default=1000,
        help='The number of steps between the moments where the current model '
             'weights get saved to disk. Defaults to 1000.'
    )

    parser.add_argument(
        '--fp16', action='store_true',
        help='Whether to use single precision for the training. This can speed '
             'the training, and requires less memory, so the batch sizes can '
             'be larger.'
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        do_lower_case=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
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
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        seed=args.seed,
        fp16=args.fp16,
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
