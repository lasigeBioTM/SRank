import argparse
import json
import sys


def format_questions(questions):
    return [{
        'id': question['id'],
        'body': question['body'],
    } for question in questions]


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Simplifies the BioASQ format to a key/value JSON file '
                    'containing all question in the input'
    )

    parser.add_argument(
        'input', metavar='INPUT',
        help='The BioASQ formatted input file.'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    with open(args.input) as f:
        questions = json.load(f)['questions']

    output = format_questions(questions)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    main()
