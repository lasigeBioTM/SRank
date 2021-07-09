from __future__ import annotations

import argparse
import json
import math
import operator
import sys
import typing
from collections import defaultdict


class QuestionSentencePair(typing.TypedDict):
    question_id: str
    question: str
    snippet: Snippet
    score: float


class Question(typing.TypedDict):
    snippets: list[Snippet]
    documents: list[str]


class Snippet(typing.TypedDict):
    text: str
    document: str


class ScoredSnippet(typing.TypedDict):
    snippet: Snippet
    score: float


# class Sentence(typing.TypedDict):
#     snippet: None


class GalagoItem(typing.TypedDict):
    answerReady: bool
    id: str
    type: str
    body: str
    documents: dict[str, Document]


class Document(typing.TypedDict):
    rank: int
    score: float


def load_galago_results(filename: str) -> dict[str, dict[str, float]]:
    print('Loading galago results ...')

    with open(filename) as f:
        raw: list[GalagoItem] = json.load(f)

    results: dict[str, dict[str, float]] = {}

    for raw_item in raw:
        question_id = raw_item['id']

        results[question_id] = {}

        for document_id, document in raw_item.get('documents', {}).items():
            results[question_id][document_id] = document['score']

    return results


def load_all_qs_pairs(filename: str) -> list[QuestionSentencePair]:
    print('Loading split sentences ...')

    with open(filename) as f:
        return [
            process_qs_pair(pair)
            for line in f
            for pair in json.loads(line)
        ]


def process_qs_pair(pair: QuestionSentencePair) -> QuestionSentencePair:
    # We need to remove the list of tokens of the snippet, because they are not
    # needed in this script and would not be accepted by BioASQ later on the
    # pipeline.

    del pair['snippet']['tokens']  # type: ignore

    return pair


def group_sentences_per_question(
    all_sentences: list[QuestionSentencePair]
) -> dict[str, list[ScoredSnippet]]:
    print('Grouping sentences by question id ...')

    result: dict[str, list[ScoredSnippet]] = defaultdict(list)

    for sentence in all_sentences:
        result[sentence['question_id']].append({
            'snippet': sentence['snippet'],
            'score': sentence['score'],
        })

    return result


def multiply_doc_score(
    galago_results: dict[str, dict[str, float]],
    sentences_per_question: dict[str, list[ScoredSnippet]],
    exponentiate: bool,
) -> None:
    print('Weighting each score by multiplying with the document score ...')

    for question_id, sentences in sentences_per_question.items():
        for sentence in sentences:
            doc_id = sentence['snippet']['document']
            doc_score = galago_results[question_id][doc_id]

            if exponentiate:
                doc_score = math.exp(doc_score)

            sentence['score'] *= doc_score


def get_most_relevant(
    sentences_per_question: dict[str, list[ScoredSnippet]],
    keep: int = 10
) -> dict[str, Question]:
    print('Sorting snippets by score ...')

    result: dict[str, Question] = {}

    for question_id, sentences in sentences_per_question.items():
        # Sort snippets from most to least relevant
        sentences.sort(
            key=operator.itemgetter('score'),
            reverse=True
        )

        # Grab the top most relevant snippets
        snippets = [
            sentence['snippet']
            for sentence in sentences[:keep]
        ]

        # Get the list of unique documents in the snippets, where the rank of a
        # document matches the highest rank of the sentences in that document
        documents = get_unique_documents(snippets)

        # Update the data structure with the selected snippets and documents
        result[question_id] = {
            'snippets': snippets,
            'documents': documents,
        }

    return result


def get_unique_documents(snippets: list[Snippet]) -> list[str]:
    # This cannot be simply a set because we want to guarantee the order of the
    # documents

    result = []

    for snippet in snippets:
        if snippet['document'] not in result:
            result.append(snippet['document'])

    return result


def update_bioasq_questions(
    questions: list[typing.Any],
    relevant_snippets_and_documents: dict[str, Question]
) -> None:
    for question in questions:
        relevant = relevant_snippets_and_documents.get(question['id'])

        if relevant is not None:
            question['snippets'] = relevant['snippets']
            question['documents'] = relevant['documents']
        else:
            question['documents'] = []
            question['snippets'] = []


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Choose the top 10 snippets for each question and create a '
                    'BioASQ-format output containing snippets and documents '
                    'for each question.'
    )

    parser.add_argument(
        'questions', metavar='QUESTIONS',
        help='The file containing the BioASQ-format questions.'
    )

    parser.add_argument(
        'sentences',
        help='The scored sentences associated with each question. This is the '
             'output of the `src/document_classification/rank_sentences.py` '
             'script.'
    )

    parser.add_argument(
        '-n', '--top', type=int, default=10,
        help='The top snippets to keep in the final output. Defaults to 10.'
    )

    parser.add_argument(
        '-m', '--multiply-document', action='store_true',
        help='Whether to multiply the score of a sentence with the galago '
             'score of the corresponding document.'
    )

    parser.add_argument(
        '-g', '--galago-results',
        help='The file containing the galago results for each question. This '
             'is only needed if you want to multiply the scores of each '
             'sentence by the score of the corresponding document.'
    )

    parser.add_argument(
        '-e', '--exponentiate', action='store_true',
        help='Whether to exponentiate the document score before multiplying. '
             'This is required when the document score follows a logarithmic '
             'scale, such as the Dirichlet score (not the BM25).'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    sentences_per_question = group_sentences_per_question(
        load_all_qs_pairs(args.sentences)
    )

    if args.multiply_document:
        galago_results = load_galago_results(args.galago_results)

        multiply_doc_score(
            galago_results,
            sentences_per_question,
            args.exponentiate
        )

    relevant_snippets_and_documents = get_most_relevant(sentences_per_question)

    with open(args.questions) as f:
        output: typing.Any = json.load(f)

    update_bioasq_questions(
        output['questions'],
        relevant_snippets_and_documents
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    main()
