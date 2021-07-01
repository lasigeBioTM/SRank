from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import typing

import pandas as pd
import spacy
from tqdm.auto import tqdm


if typing.TYPE_CHECKING:
    Language = spacy.language.Language


class DocDetails(typing.TypedDict):
    title: str
    abstract: str
    publish_time: str


class Snippet(typing.TypedDict):
    offsetInBeginSection: int
    offsetInEndSection: int
    text: str
    beginSection: str
    endSection: str
    document: str


def get_metadata(filename: str) -> None:
    global metadata

    metadata = pd.read_csv(filename, low_memory=False)

    metadata = metadata[
        pd.notna(metadata['title']) &
        pd.notna(metadata['abstract']) &
        pd.notna(metadata['pubmed_id'])
    ]

    metadata = metadata[['cord_uid', 'title', 'abstract', 'publish_time']]

    metadata = metadata.drop_duplicates('cord_uid').set_index('cord_uid')


def get_docset(document_ids: list[str], use_mp: bool) -> dict[str, DocDetails]:
    result: dict[str, DocDetails] = {}

    if use_mp:
        with mp.Pool(processes=20) as pool:
            doc_objects = pool.map(get_doc_text, document_ids)

            for i, doc in enumerate(doc_objects):
                if doc is not None:
                    result[document_ids[i]] = doc
    else:
        for doc_id in document_ids:
            doc = get_doc_text(doc_id)

            if doc is not None:
                result[doc_id] = doc

    return result


def get_doc_text(doc_id: str) -> DocDetails | None:
    global metadata

    try:
        row = dict(metadata.loc[doc_id])
    except KeyError:
        return None

    return {
        'title': row['title'],
        'abstract': row['abstract'],
        'publish_time': row['publish_time'],
    }


def split_section(doc_id: str, text: str, section: str, nlp: Language) -> list[Snippet]:
    return [
        {
            'offsetInBeginSection': sentence.start_char,
            'offsetInEndSection': sentence.end_char,
            'text': sentence.text,
            'beginSection': section,
            'endSection': section,
            'document': doc_id
        }
        for sentence in nlp(text).sents
    ]


def split_into_snippets(doc_id: str, document: DocDetails, nlp: Language) -> list[Snippet]:
    return (
        split_section(doc_id, document['title'], 'title', nlp) +
        split_section(doc_id, document['abstract'], 'abstract', nlp)
    )


def split_docset_into_sentences(docset: dict[str, DocDetails], nlp: Language) -> dict[str, list[Snippet]]:
    return {
        paper_id: split_into_snippets(paper_id, paper, nlp)
        for paper_id, paper in tqdm(docset.items())
    }


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Given the results generated by the `src/retrieve.py` '
                    'script, gather the set of retrieved documents, their '
                    'title and abstract, and split the documents into snippets, '
                    'which will be used later on the pipeline.'
    )

    parser.add_argument(
        'retrieved', metavar='RETRIEVED',
        help='The galago results retrieved for a set of questions. This must '
             'be the output of the `src/retrieve.py` script.'
    )

    parser.add_argument(
        'metadata', metavar='METADATA',
        help='The `metadata.csv` file describing title and abstract of the '
             'CORD dataset.'
    )

    parser.add_argument(
        '-m', '--multiprocess', action='store_true',
        help='Whether to use multiprocessing. This speeds up the process by '
             'using multiple cores of the machine.'
    )

    parser.add_argument(
        '-n', '--nlp-model', default='en_core_web_lg',
        help='The spacy model to use for sentence splitting. '
             'Defaults to "en_core_web_lg". The "ner", "attribute_ruler" and '
             '"lemmatizer" are not loaded, as the model is used only to split '
             'sentences.'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    with open(args.retrieved) as f:
        galago_results = json.load(f)

    nlp = spacy.load(args.nlp_model, exclude=[
        'ner',
        'attribute_ruler',
        'lemmatizer',
    ])

    document_ids = list({
        doc_id
        for question in galago_results
        for doc_id in question['documents']
    })

    # The following function creates a global metadata variable that is used by
    # other part of the script. This is the easiest way I could envision
    # to make the script work with multiprocessing elements
    get_metadata(args.metadata)

    docset = get_docset(document_ids, args.multiprocess)

    output = split_docset_into_sentences(docset, nlp)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    # Global type hint
    metadata: pd.DataFrame

    main()
