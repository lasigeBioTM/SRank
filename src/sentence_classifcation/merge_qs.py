# Merge question/snippet pairs, and include a label for positive and negative
# instances.
#
# Positive instances come from BioASQ formatted files, negative instances from
# outputs of the `src/sentence_classification/negative_sample.py` script

from __future__ import annotations

import json
import argparse
import sys
import typing


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Merge several files to create a dataset to train a '
                    'question/sentence pair classifier.'
    )

    parser.add_argument(
        '-p', '--positive', nargs='*', default=[],
        help='The file containing the positive question/sentence pairs. Note '
             'that the file must be in the BioASQ format, which means a JSON '
             'file with a "questions" property, where each question is '
             'associated with a "snippets" list.'
    )

    parser.add_argument(
        '-n', '--negative', nargs='*', default=[],
        help='The file containing the negative question/sentence pairs. Note '
             'that the file must be in the format output by '
             '`src/sentence_classification/negative_sample.py`, which means a '
             'JSON file containing a list of objects with "question_body" and '
             '"snippet" properties.'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    return parser.parse_args()


class Question(typing.TypedDict):
    body: str
    snippets: list[Snippet]


class Snippet(typing.TypedDict):
    text: str
    golden: bool


class QuestionSentencePair(typing.TypedDict):
    question: str
    sentence: str
    label: str


def get_positive_pairs(questions: list[Question]) -> list[QuestionSentencePair]:
    return [
        {
            'question': question['body'],
            'sentence': snippet['text'],
            'label': 'positive',
        }
        for question in questions
        for snippet in question['snippets']
        if snippet.get('golden', True)
    ]


def get_negative_pairs(pairs: list[QuestionSentencePair]) -> list[QuestionSentencePair]:
    return [{
        'question': pair['question'],
        'sentence': pair['sentence'],
        'label': 'negative',
    } for pair in pairs]


def main() -> None:
    args = get_arguments()

    output = []

    for filename in args.positive:
        with open(filename) as f:
            partial = json.load(f)['questions']

        output.extend(get_positive_pairs(partial))

    for filename in args.negative:
        with open(filename) as f:
            partial = json.load(f)

        output.extend(get_negative_pairs(partial))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    main()
