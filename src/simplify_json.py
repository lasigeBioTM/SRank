import argparse
import json
import sys
import typing


class SimpleQuestion(typing.TypedDict):
    id: str
    body: str


def format_questions(questions: list[typing.Any]) -> list[SimpleQuestion]:
    return [{
        'id': question['id'],
        'body': question['body'],
    } for question in questions]


def get_arguments() -> argparse.Namespace:
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


def main() -> None:
    args = get_arguments()

    with open(args.input) as f:
        questions: list[typing.Any] = json.load(f)['questions']

    output = format_questions(questions)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    main()
