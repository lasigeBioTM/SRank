# We need to convert the following into a dataset of question/sentence pairs
# from the BioASQ synergy feedback files, which can be used to derive positive
# and negative pairs.

# Search each question with galago and get the abstract of the papers that are
# not related to the question. Sample the sentences that are more highly related
# to the questions based on the CLS embedding of the sentences and a cosine
# similarity of the vectors.

# Keep only a fraction of the negative sentences so that the overall ratio of
# positive to negative instances is some defined constant

# These options can be tuned by command line flags

# For this to work, we need to have the data above in hand. We also need to run
# all questions through the galago index before in order to gather a few
# unrelated documents before hand (see `src/retrieve.py`)

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import typing


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sample a selection of relevant and non-relevant sentences '
                    'to go along with each question in order to create a '
                    'dataset of question/sentence pairs that can be used to '
                    'train a transformer model in order to identify relevant '
                    'abstracts snippets to answer a question.'
    )

    parser.add_argument(
        'questions', metavar='QUESTIONS',
        help='The file containing the training questions. The file must be in '
             'the BioASQ format, and must contain a list of snippets and a '
             'list of relevant documents for each question. Snippets in this '
             'file are considered positive instances, unless they are marked '
             'with the `"golden": false` property. Non-golden snippets are '
             'used as negative instances (see the `--sample` and `--ratio` '
             'flags).'
    )

    parser.add_argument(
        'docset', metavar='DOCSET',
        help='The snippets extracted from a set of documents. This can be '
             'created with the `src/make_docset.py` file. These snippets '
             'will be used to sample negative instances. You can provide '
             'a dummy argument (eg. "-") if you know that there are enough '
             'negative instances in the input to not require sampling.'
    )

    parser.add_argument(
        '-f', '--from-source',
        help='By default, this script gathers negative instances from the '
             'input (if any exists). With this flag, we can decide how many '
             'of these known instances to keep. By default, all are kept. If '
             'an integer is given, that number of instances is kept; if it is '
             'a float, it represents the proportion of the final set of '
             'negative instances that correspond to the known instances. In '
             'any case, if the value is too big, all instances are kept.'
    )

    parser.add_argument(
        '-r', '--ratio', type=float, default=4,
        help='The amount of negative instances generated for each positive '
             'instance.'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='The seed to initialize the randomization process.'
    )

    return parser.parse_args()


class PositiveQuestionSnippetPair(typing.TypedDict):
    question_id: str
    question_body: str
    snippet: str


class NegativeQuestionSnippetPair(typing.TypedDict):
    question_body: str
    snippet: str


def partition_pairs(
    questions: list[typing.Any]
) -> tuple[list[PositiveQuestionSnippetPair], list[NegativeQuestionSnippetPair]]:

    positive_pairs: list[PositiveQuestionSnippetPair] = []
    negative_pairs: list[NegativeQuestionSnippetPair] = []

    for question in questions:
        for snippet in question['snippets']:
            if snippet.get('golden', True):
                positive_pairs.append({
                    'question_id': question['id'],
                    'question_body': question['body'],
                    'snippet': snippet['text'],
                })
            else:
                negative_pairs.append({
                    'question_body': question['body'],
                    'snippet': snippet['text'],
                })

    return positive_pairs, negative_pairs


def get_related_documents(questions: list[typing.Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    for question in questions:
        related_documents: list[str] = []

        for document in question['documents']:
            if isinstance(document, dict):
                related = document.get('golden', True)
                document = document['id']
            elif isinstance(document, str):
                related = True
            else:
                continue

            if related:
                related_documents.append(document)

        result[question['id']] = related_documents

    return result


def get_amount_to_keep(arg: str, output_size: int) -> int:
    try:
        arg_as_int = int(arg)
    except:
        pass
    else:
        return arg_as_int

    try:
        arg_as_float = float(arg)
    except:
        pass
    else:
        return int(output_size * arg_as_float)

    raise Exception(
        f'Argument of --from-source cannot be converted to integer or float'
    )


def main() -> None:
    args = get_arguments()

    with open(args.questions) as f:
        questions = json.load(f)['questions']

    positive_pairs, negative_pairs = partition_pairs(questions)

    rand = random.Random(args.seed)

    # Shuffle the positive and negative pairs
    rand.shuffle(positive_pairs)
    rand.shuffle(negative_pairs)

    output_size = int(args.ratio * len(positive_pairs))

    if args.from_source is not None:
        keep = get_amount_to_keep(args.from_source, output_size)

        negative_pairs = negative_pairs[:keep]

    if len(negative_pairs) / len(positive_pairs) > args.ratio:
        negative_pairs = negative_pairs[:output_size]

    elif len(negative_pairs) / len(positive_pairs) < args.ratio:
        with open(args.docset) as f:
            snippets = [
                snippet
                for snippets in json.load(f).values()
                for snippet in snippets
            ]

        related_documents = get_related_documents(questions)

        positive_pair_iterator = itertools.cycle(positive_pairs)

        while len(negative_pairs) / len(positive_pairs) < args.ratio:
            # We're running through all questions/snippet pairs in order,
            # cycling whenever we reach the end. Because we already shuffled the
            # positive pairs, this should not have any bias towards any of the
            # questions

            pair = next(positive_pair_iterator)

            while True:
                random_snippet = rand.choice(snippets)

                if random_snippet['document'] not in related_documents[pair['question_id']]:
                    break

            negative_pairs.append({
                'question_body': pair['question_body'],
                'snippet': random_snippet['text'],
            })

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(negative_pairs, f)
    else:
        json.dump(negative_pairs, sys.stdout)


if __name__ == '__main__':
    main()
