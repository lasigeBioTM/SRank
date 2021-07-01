from __future__ import annotations

import argparse
import json
import typing
import sys

import pandas as pd


PREFIX = 'http://www.ncbi.nlm.nih.gov/pubmed/'
PREFIX_LEN = len(PREFIX)


def get_metadata(filename: str) -> dict[str, str]:
    metadata: pd.DataFrame = pd.read_csv(filename, low_memory=False)

    metadata = metadata[
        pd.notna(metadata['title']) &
        pd.notna(metadata['abstract']) &
        pd.notna(metadata['pubmed_id'])
    ]

    metadata = metadata[['cord_uid', 'pubmed_id']]

    metadata = metadata.drop_duplicates('pubmed_id').set_index('pubmed_id')

    return dict(metadata['cord_uid'])


def convert_documents(documents: list[str], metadata: dict[str, str]) -> list[str]:
    cord_ids = (
        pubmed_url_to_cord_id(document, metadata)
        for document in documents
    )

    return [i for i in cord_ids if i is not None]


def pubmed_url_to_cord_id(pubmed_url: str, map: dict[str, str]) -> str | None:
    return map.get(pubmed_url[PREFIX_LEN:])


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert a BioASQ golden file so that documents identified '
                    'with their pubmed ID link are converted to their CORD '
                    'identifier. The output contains, for each question, only '
                    'their id, the list of documents, the body of the question '
                    'and the snippets. The document identifier in the snippets '
                    'is *not* converted.'
    )

    parser.add_argument(
        'input', metavar='INPUT',
        help='The BioASQ file. In each question, each element in the '
             '"documents" list will get converted to a CORD identifier. If '
             'there is not correspondence, the element is removed from the '
             'list.'
    )

    parser.add_argument(
        'metadata', metavar='METADATA',
        help='The `metadata.csv` file describing title and abstract of the '
             'CORD dataset.'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    with open(args.input) as f:
        data = json.load(f)

    metadata = get_metadata(args.metadata)

    output = [
        convert_question(question, metadata)
        for question in data['questions']
    ]

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


def convert_question(question: dict[str, typing.Any], metadata: dict[str, str]) -> dict[str, typing.Any]:
    return {
        'id': question['id'],
        'body': question['body'],
        'documents': convert_documents(question['documents'], metadata),
        'snippets': question['snippets'],
    }


if __name__ == '__main__':
    main()
