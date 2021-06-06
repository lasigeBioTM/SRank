"""
Merge BioASQ task B and Task Synergy data and then split into train and test
sets.
"""

import argparse
import json
import random
import os


def merge(task_b_contents, synergy_contents, prune_on_documents, prune_on_snippets):
    all_questions = [
        extract_question(question, 'TASK_B')
        for question in task_b_contents
    ] + [
        extract_question(question, 'SYNERGY')
        for question in synergy_contents
    ]

    # Keep only questions with documents and snippets
    if prune_on_documents or prune_on_snippets:
        def to_keep(question):
            return (
                (question['documents'] or not prune_on_documents) and
                (question['snippets'] or not prune_on_snippets)
            )

        all_questions = [
            question
            for question in all_questions
            if to_keep(question)
        ]

    return all_questions


def extract_question(question, source):
    return {
        'id': question['id'],
        'body': question['body'],
        'type': question['type'],
        'documents': question['documents'],
        'snippets': [
            {'text': s['text'], 'document': s['document']}
            for s in question['snippets']
        ],
        'source': source,
    }


def shuffle_and_split(all_questions, test_ratio, seed):
    rand = random.Random(seed)

    rand.shuffle(all_questions)

    test_size = int(test_ratio * len(all_questions))

    test_dataset = all_questions[:test_size]
    train_dataset = all_questions[test_size:]
    return test_dataset, train_dataset


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Merge BioASQ task B and Task Synergy data and then split '
                    'into train and test sets.'
    )

    parser.add_argument(
        'task_b', metavar='TASK-B',
        help='The file containing the BioASQ Task B training data.'
    )

    parser.add_argument(
        'synergy_round', metavar='SYNERGY',
        help='The file containing the Task Synergy gold standard'
    )

    parser.add_argument(
        '-s', '--seed', type=int, default=42
    )

    parser.add_argument(
        '-r', '--test-ratio', type=float, default=0.1
    )

    parser.add_argument(
        '--keep-no-documents', action='store_false', dest='prune_on_documents'
    )

    parser.add_argument(
        '--keep-no-snippets', action='store_false', dest='prune_on_snippets'
    )

    parser.add_argument(
        '-d', '--destination', default='data/merged/',
        help='The directory where the `questions.train.json` and `questions.test.json` '
             'files will be stored'
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    with open(args.task_b) as f:
        task_b_contents = json.load(f)['questions']

    with open(args.synergy_round) as f:
        synergy_contents = json.load(f)['questions']

    all_questions = merge(
        task_b_contents,
        synergy_contents,
        args.prune_on_documents,
        args.prune_on_snippets,
    )

    test_dataset, train_dataset = shuffle_and_split(
        all_questions,
        args.test_ratio,
        args.seed,
    )

    os.makedirs(args.destination, exist_ok=True)

    with open(os.path.join(args.destination, 'questions.train.json'), 'w') as f:
        json.dump(train_dataset, f)

    with open(os.path.join(args.destination, 'questions.test.json'), 'w') as f:
        json.dump(test_dataset, f)


if __name__ == '__main__':
    main()
