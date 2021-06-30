from __future__ import annotations

import argparse
import json
import os
import subprocess
import typing

import spacy


class Question(typing.TypedDict):
    id: str
    body: str
    documents: list


class DocScore:
    rank: int
    score: float


def retrieve_documents(
    questions,
    requested,
    index_dirname,
    galago_path,
    format_question: typing.Callable[[str, str], str],
    scorer: str | None,
    thread_count: int,
):
    galago_output = run_galago(
        questions,
        requested,
        index_dirname,
        galago_path,
        format_question,
        scorer,
        thread_count
    )

    result: dict[str, dict[str, DocScore]] = {}

    for line in galago_output.splitlines():
        fields = line.split()

        if not fields or fields[-1] != 'galago':
            continue

        try:
            qid = fields[0]
            doc_id = fields[2].split('/')[-1].split('.')[0]
            rank = int(fields[3])
            score = float(fields[4])
        except ValueError:
            continue

        if qid not in result:
            result[qid] = {}

        result[qid][doc_id] = {
            'rank': rank,
            'score': score,
        }

    return result


def run_galago(
    questions,
    requested,
    index_dirname,
    galago_path,
    format_question: typing.Callable[[str, str], str],
    scorer,
    thread_count,
):
    galago_filename = 'results/galago_query.json'

    write_galago_query(
        questions,
        index_dirname,
        requested,
        galago_filename,
        format_question,
        scorer,
        thread_count,
    )

    galago_process = subprocess.Popen(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        args=[
            galago_path,
            'threaded-batch-search',
            galago_filename,
        ]
    )

    print('Running galago ...')

    try:
        result, _ = galago_process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        galago_process.kill()
        result, _ = galago_process.communicate()

    print('Done')

    os.remove(galago_filename)

    return result.decode('utf8')


def write_galago_query(
    questions: list[Question],
    index_dirname: str,
    requested: int,
    galago_filename: str,
    format_question: typing.Callable[[str, str], str],
    scorer,
    thread_count,
):
    config = {
        'threadCount': thread_count,
        'caseFold': True,
        'index': index_dirname,
        'requested': requested,
        'queries': [
            format_question(question['id'], question['body'])
            for question in questions
        ]
    }

    if scorer is not None:
        config['scorer'] = scorer

    with open(galago_filename, 'w') as f:
        json.dump(config, f)


def format_question_flat_combine(qid, body, nlp):
    document = nlp(body)

    tokens = [
        token.text
        for token in document
        if not token.is_punct and not token.is_space and not token.is_stop
    ]

    return {
        'number': str(qid),
        'text': '#combine({})'.format(' '.join(tokens)),
    }


def format_question_with_dependency_operator(qid, body, nlp, operator):
    document = nlp(body)

    noun_chunks = list(document.noun_chunks)

    def in_noun_chunk(token):
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
        'number': str(qid),
        'text': '#combine({})'.format(' '.join(pieces)),
    }


def process_galago_results(galago_results: dict[str, dict[str, DocScore]], questions):
    """
    Process document retrieval files to be used by AUEB system
    """

    for question in questions:
        update_question(question, galago_results.get(question['id']))


def update_question(question: Question, galago_results_for_question: dict[str, DocScore] | None):
    if galago_results_for_question is None:
        return

    retrieved_documents = retrieve_documents_for_question(
        question,
        galago_results_for_question
    )

    question['retrieved_documents'] = retrieved_documents

    question['num_ret'] = len(retrieved_documents)

    question['num_rel_ret'] = len(
        set(galago_results_for_question) &
        set(question['relevant_documents'])
    )


def retrieve_documents_for_question(question: Question, question_results: dict[str, DocScore]):
    return [{
        'doc_id': doc_id,
        'rank': document['rank'],
        'galago_score': document['score'],
        'is_relevant': doc_id in question['relevant_documents'],
        'score': document['score'],
    } for doc_id, document in question_results.items()]


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Retrieve relevant documents for a set of queries'
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
        # TODO: Default to the highest galago-*-bin/bin/galago
        default='galago-3.19-bin/bin/galago',
    )

    parser.add_argument(
        '-k', '--requested', type=int, default=100,
        help='The requestes number of documents to retrieve for each question. '
             'Defaults to 100.'
    )

    parser.add_argument(
        '-o', '--output', default='results/retrieved.json',
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
        '-r', '--raw', action='store_true',
        help='By default, this script appends the retrieved documents to a '
             'copy of the QUESTIONS file; if this flag is given, the output '
             'is a simpler JSON file where each item contains only its ID, '
             'question body and the retrieved document IDS, in rank order.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    nlp = spacy.load(args.spacy_model)

    with open(args.questions_filename) as f:
        questions = json.load(f)['questions']

    questions = [{
        'id': question['id'],
        'body': question['body'],
        'documents': question.get('documents', [])
    } for question in questions]

    if args.noun_chunk_operator is None:
        def format_question(qid: str, body: str) -> str:
            return format_question_flat_combine(qid, body, nlp)
    else:
        def format_question(qid: str, body: str) -> str:
            return format_question_with_dependency_operator(qid, body, nlp, args.noun_chunk_operator)

    galago_results = retrieve_documents(
        questions,
        args.requested,
        args.galago_index,
        args.galago_path,
        format_question,
        args.scorer,
        args.thread_count
    )

    if args.raw:
        questions = [{
            'id': question['id'],
            'body': question['body'],
            'documents': retrieve_documents_for_question(
                question,
                galago_results.get(question['id'])
            )
        } for question in questions]
    else:
        process_galago_results(galago_results, questions)

    with open(args.output, 'w') as f:
        json.dump({'queries': questions}, f)
