from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import typing

import pandas as pd
import spacy
from tqdm import tqdm


class Query(typing.TypedDict):
    query_id: str
    query_text: str


class DocScore:
    rank: int
    score: float


def write_galago_query(queries: list[Query], index_dirname, n, galago_filename, nlp):
    result = {
        'threadCount': 20,
        'caseFold': True,
        'index': index_dirname,
        'requested': n,
        'scorer': 'bm25',
        # 'mu': 2000,
        # 'lambda': 0.2,
        'queries': [format_query(q['query_id'], q['query_text'], nlp) for q in queries]
    }

    with open(galago_filename, 'w') as f:
        json.dump(result, f)


def format_query(qid, body, nlp):
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


def retrieve_documents(queries, n, index_dirname, galago_path, nlp):
    galago_output = run_galago(queries, n, index_dirname, galago_path, nlp)

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


def run_galago(queries, n, index_dirname, galago_path, nlp):
    galago_filename = 'results/galago_query.json'

    write_galago_query(queries, index_dirname, n, galago_filename, nlp)

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


def process_galago_results(galago_results: dict[str, dict[str, DocScore]], queries):
    """
    Process document retrieval files to be used by AUEB system
    """

    for query in queries:
        results_for_query = galago_results.get(query['query_id'])

        if results_for_query:
            update_query(query, results_for_query)


def update_query(query, galago_results_for_query):
    retrieved_documents = retrieve_documents_for_query(
        query,
        galago_results_for_query
    )

    query['retrieved_documents'] = retrieved_documents

    query['num_ret'] = len(retrieved_documents)

    query['num_rel_ret'] = len(
        set(galago_results_for_query) &
        set(query['relevant_documents'])
    )


def retrieve_documents_for_query(query, query_results):
    return [{
        'doc_id': doc_id,
        'rank': document['rank'],
        'bm25_score': document['score'],
        'norm_bm25_score': document['score'],
        'is_relevant': doc_id in query['relevant_documents'],
        'score': document['score'],
    } for doc_id, document in query_results.items()]


def get_docset(document_ids, use_mp):
    result: dict[str] = {}

    if use_mp:
        with multiprocessing.Pool(processes=20) as pool:
            doc_objects = pool.map(get_doc_text, document_ids)

            for i, doc in enumerate(doc_objects):
                result[str(document_ids[i])] = doc
    else:
        for doc_id in tqdm(document_ids):
            result[str(doc_id)] = get_doc_text(doc_id)

    for doc_id in document_ids:
        if result.get(str(doc_id), None) is None:
            if str(doc_id) in result:
                del result[str(doc_id)]

    return result


def get_doc_text(doc_id) -> dict[typing.Literal['title', 'abstract', 'publish_time'], str] | None:
    """
    Retrieve title and abstract of a CORD-19 paper
    """
    global metadata

    try:
        row = dict(metadata.loc[doc_id])
    except KeyError:
        return None

    return {
        'title': row['title'],
        'abstractText': row['abstract'],
        'publicationDate': row['publish_time'],
    }


def get_metadata(filename):
    metadata = pd.read_csv(filename, low_memory=False)

    metadata = metadata[
        pd.notna(metadata['title']) &
        pd.notna(metadata['abstract']) &
        pd.notna(metadata['pubmed_id'])
    ]

    metadata = metadata[['cord_uid', 'title', 'abstract', 'publish_time']]

    return metadata.drop_duplicates('cord_uid').set_index('cord_uid')


def get_arguments():
    parser = argparse.ArgumentParser(
        'Retrieve relevant documents for a set of queries'
    )

    parser.add_argument(
        '-q', '--questions-filename',
        default='data/BioASQ_Synergy9_v1/golden_round_4.json',
    )

    parser.add_argument(
        '-i', '--index',
        default='/mnt/data/jferreira_data/cord-galago-index/'
    )

    parser.add_argument(
        '-g', '--galago-path',
        default='galago-3.19-bin/bin/galago',
    )

    parser.add_argument(
        '-k', '--k', type=int, default=100,
    )

    parser.add_argument(
        '-m', '--multiprocess', action='store_true'
    )

    parser.add_argument(
        '-o', '--output', default='results/retrieved.json',
    )

    parser.add_argument(
        '-d', '--docset-path', default='docset.pkl'
    )

    parser.add_argument(
        '-s', '--spacy-model', default='en_core_web_lg',
    )

    parser.add_argument(
        '-c', '--cord-metadata',
        default='/mnt/data/jferreira_data/cord/2021-05-24/metadata.csv'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    nlp = spacy.load(args.spacy_model)

    with open(args.questions_filename) as f:
        questions = json.load(f)['questions']

    queries = [{
        'query_id': q['id'],
        'query_text': q['body'],
        'relevant_documents': q['documents']
    } for q in questions]

    galago_results = retrieve_documents(
        queries,
        n=args.k,
        index_dirname=args.index,
        galago_path=args.galago_path,
        nlp=nlp
    )

    process_galago_results(galago_results, queries)

    with open(args.output, 'w') as f:
        json.dump({'queries': queries}, f)

    global metadata
    metadata = get_metadata(args.cord_metadata)

    document_ids = list({
        doc_ids
        for q in galago_results
        for doc_ids in galago_results[q].keys()
    })

    docset = get_docset(document_ids, args.multiprocess)

    with open(args.docset_path) as f:
        json.dump(docset, f)

    print('All done')
