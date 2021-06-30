from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import typing
from collections import defaultdict

import spacy


class Question(typing.TypedDict):
    id: str
    body: str


class DocScore(typing.TypedDict):
    rank: int
    score: float


class GalagoItem(typing.TypedDict):
    number: str
    text: str


class Retriever:
    def __init__(
        self,
        *,
        format_question: typing.Callable[[Question], GalagoItem],
        galago_path: str,
        index: str,
        requested: int = 100,
        scorer: str | None = None,
        thread_count: int = 1,
    ):
        self.format_question = format_question
        self.galago_path = galago_path
        self.index = index
        self.requested = requested
        self.scorer = scorer
        self.thread_count = thread_count

    def retrieve(self, questions: list[Question]) -> dict[str, dict[str, DocScore]]:
        # Create a temporary file to contain the galago query parameters,
        # including the questions being searched. This file will be deleted
        # after galago runs.

        query_filename = 'results/galago_query.json'

        self.write_galago_query(questions, query_filename)

        galago_output = self.run_galago(query_filename)

        os.remove(query_filename)

        return process_galago_output(galago_output)

    def write_galago_query(
        self,
        questions: list[Question],
        query_filename: str,
    ) -> None:
        config = {
            'threadCount': self.thread_count,
            'caseFold': True,
            'index': self.index,
            'requested': self.requested,
            'queries': [
                self.format_question(question)
                for question in questions
            ]
        }

        if self.scorer is not None:
            config['scorer'] = self.scorer

        with open(query_filename, 'w') as f:
            json.dump(config, f)

    def run_galago(self, query_filename: str) -> str:
        galago_process = subprocess.Popen(
            args=[
                self.galago_path,
                'threaded-batch-search',
                query_filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        result, _ = galago_process.communicate()

        return result.decode('utf8')


def process_galago_output(galago_output: str) -> dict[str, dict[str, DocScore]]:
    result: dict[str, dict[str, DocScore]] = defaultdict(dict)

    for line in galago_output.splitlines():
        fields = line.split()

        if not fields:
            # Ignore lines that do not conform to the expected format
            continue

        try:
            # Process the fields. If something goes wrong, ignore this line
            # This effectively could lead to data loss, but I think galago
            # (at least in the current version) never fails to uphold the
            # criteria I'm assuming here
            question_id = fields[0]

            # Note that the document is given with a full filename. I am
            # interested only in the basename of the file (which is its ID)
            try:
                document_id = fields[2].split('/')[-1].split('.')[0]
            except:
                print(fields)
                raise

            rank = int(fields[3])
            score = float(fields[4])
        except ValueError:
            # Again, if something does not conform to the assumptions,
            # ignore the line
            continue

        result[question_id][document_id] = {
            'rank': rank,
            'score': score,
        }

    return result


def format_question_flat_combine(question: Question, nlp: spacy.language.Language) -> GalagoItem:
    document = nlp(question['body'])

    tokens: list[str] = [
        token.text
        for token in document
        if not token.is_punct and not token.is_space and not token.is_stop
    ]

    return {
        'number': question['id'],
        'text': '#combine({})'.format(' '.join(tokens)),
    }


def format_question_with_dependency_operator(
    question: Question,
    nlp: spacy.language.Language,
    operator: str
) -> GalagoItem:
    document = nlp(question['body'])

    noun_chunks = list(document.noun_chunks)

    def in_noun_chunk(token: spacy.tokens.Token) -> bool:
        return any(chunk.start <= token.i < chunk.end for chunk in noun_chunks)

    pieces = [
        token.text
        for token in document
        if (
            not token.is_punct and
            not token.is_space and
            not token.is_stop and
            not in_noun_chunk(token)
        )
    ]

    for chunk in noun_chunks:
        keep = [
            token.text
            for token in chunk
            if (
                not token.is_punct and
                not token.is_space and
                not token.is_stop
            )
        ]

        if len(keep) == 0:
            continue
        elif len(keep) == 1:
            pieces.append(keep[0])
        else:
            pieces.append(
                '#{operator}({parts})'.format(
                    operator=operator,
                    parts=' '.join(keep)
                )
            )

    return {
        'number': str(question['id']),
        'text': '#combine({})'.format(' '.join(pieces)),
    }


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Retrieve relevant documents for a set of questions'
    )

    parser.add_argument(
        'questions_filename', metavar='QUESTIONS',
        help='The BioASQ-formatted list of questions'
    )

    parser.add_argument(
        'galago_index', metavar='GALAGO-INDEX',
    )

    parser.add_argument(
        '-g', '--galago-path',
        help='The path to the gaalgo executable. The default is the path '
             '`./galago-*-bin/bin/galago`, where * matches the highest galago '
             'installation that can be found with that glob pattern.'
    )

    parser.add_argument(
        '-k', '--requested', type=int, default=100,
        help='The requestes number of documents to retrieve for each question. '
             'Defaults to 100.'
    )

    parser.add_argument(
        '-S', '--scorer',
        help='The galago scorer to use. If unspecified, the scorer is the '
             'default galago one. Some options for this argument are "bm25" '
             'and "dirichlet".'
    )

    parser.add_argument(
        '-s', '--spacy-model', default='en_core_web_lg',
        help='The spacy NLP model to use to tokenize sentences. We use this to '
             'remove stop words, punctuation and irrelevant whitespace from '
             'the questions, as well as to find noun chunks (see the '
             '`--noun-chunk-operator` flag).'
    )

    parser.add_argument(
        '-c', '--noun-chunk-operator',
        help='If given, this flag describes the galago operator to use to group '
             'noun chunks in the questions. This is a way of ensuring wighing '
             'in the fact that this words appear sequentially in the question. '
    )

    parser.add_argument(
        '-t', '--thread-count', type=int, default=20,
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    args = parser.parse_args()

    if args.galago_path is None:
        try:
            args.galago_path = get_highest_galago_path()
        except IndexError:
            parser.error('Cannot find galago on `galago-*-bin/bin/galago`.')

    return args


def get_highest_galago_path() -> str:
    paths = glob.glob('galago-*-bin/bin/galago')

    paths.sort(
        key=lambda p: p[7:-15].split('.'),
        reverse=True,
    )

    return paths[0]


def main() -> None:
    args = get_arguments()

    with open(args.questions_filename) as f:
        questions: list[Question] = json.load(f)

    nlp = spacy.load(args.spacy_model)

    if args.noun_chunk_operator is None:
        def format_question(question: Question) -> GalagoItem:
            return format_question_flat_combine(question, nlp)
    else:
        def format_question(question: Question) -> GalagoItem:
            return format_question_with_dependency_operator(question, nlp, args.noun_chunk_operator)

    retriever = Retriever(
        format_question=format_question,
        galago_path=args.galago_path,
        index=args.galago_index,
        requested=args.requested,
        scorer=args.scorer,
        thread_count=args.thread_count,
    )

    galago_results = retriever.retrieve(questions)

    # Notice that mypy does not really like me to spread the `question` variable
    # with the doube asterisk operator. That's why I'm hinting this as a list of
    # `Any`.
    output: list[typing.Any] = [{
        **question,
        'documents': galago_results.get(question['id']),
    } for question in questions]

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    main()
