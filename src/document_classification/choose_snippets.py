import argparse
import json
import math
import operator
import sys
from collections import defaultdict


def load_questions(filename):
    print('Loading questions model ...')

    with open(filename) as f:
        return json.load(f)


def load_galago_results(filename):
    print('Loading galago results ...')

    with open(filename) as f:
        raw = json.load(f)['queries']

    results = {}

    for query in raw:
        query_id = query['query_id']

        results[query_id] = {}

        for document in query.get('retrieved_documents', []):
            doc_id = document['doc_id']
            galago_score = document['galago_score']

            results[query_id][doc_id] = galago_score

    return results


def load_all_sentences(filename):
    print('Loading split sentences ...')

    with open(filename) as f:
        return [
            sentence
            for line in f
            for sentence in json.loads(line)
        ]


def group_sentences_per_query(all_sentences):
    print('Grouping sentences by query id ...')

    result = defaultdict(list)

    for sentence in all_sentences:
        result[sentence['query_id']].append({
            'snippet': sentence['snippet'],
            'score': sentence['score'],
        })

    return result


def multiply_doc_score(galago_results, sentences_per_query, EXPONENTIATE):
    print('Weighting each score by multiplying with the document score ...')

    for query_id, sentences in sentences_per_query.items():
        for sentence in sentences:
            doc_id = sentence['snippet']['document']
            doc_score = galago_results[query_id][doc_id]

            if EXPONENTIATE:
                doc_score = math.exp(doc_score)

            sentence['score'] *= doc_score


def get_most_relevant(sentences_per_query, keep=10):
    print('Sorting snippets by score ...')

    result = {}

    for query_id, sentences in sentences_per_query.items():
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
        result[query_id] = {
            'snippets': snippets,
            'documents': documents,
        }

    return result


def get_unique_documents(snippets):
    # This cannot be simply a set because we want to guarantee the order of the
    # documents

    result = []

    for snippets in snippets:
        if snippets['document'] not in result:
            result.append(snippets['document'])

    return result


def update_bioasq_questions(questions, relevant_snippets_and_documents):
    for question in questions:
        relevant = relevant_snippets_and_documents.get(question['id'])

        if relevant is not None:
            question['snippets'] = relevant['snippets']
            question['documents'] = relevant['documents']
        else:
            question['documents'] = []
            question['snippets'] = []


def get_arguments():
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
        '-g', '--galago-results',
        help='The file containing the galago results for each question. This '
             'is only needed if you want to multiply the scores of each '
             'sentence by the score of the corresponding document.'
    )

    parser.add_argument(
        '-m', '--multiply-document', action='store_true',
        help='Whether to multiply the score of a sentence with the galago '
             'score of the corresponding document.'
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


def main():
    args = get_arguments()

    sentences_per_query = group_sentences_per_query(
        load_all_sentences(args.sentences)
    )

    if args.multiply_document:
        galago_results = load_galago_results(args.galago_results)

        multiply_doc_score(
            galago_results, sentences_per_query, args.exponentiate)

    relevant_snippets_and_documents = get_most_relevant(sentences_per_query)

    output = load_questions(args.questions)

    update_bioasq_questions(output['questions'],
                            relevant_snippets_and_documents)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    main()
