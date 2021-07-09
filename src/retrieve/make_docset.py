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


class Snippet(typing.TypedDict):
    offsetInBeginSection: int
    offsetInEndSection: int
    text: str
    beginSection: str
    endSection: str
    document: str
    tokens: list[str]


def get_metadata(filename: str) -> None:
    global metadata

    metadata = pd.read_csv(filename, low_memory=False)

    metadata = metadata[
        pd.notna(metadata['title']) &
        pd.notna(metadata['abstract']) &
        pd.notna(metadata['pubmed_id'])
    ]

    metadata = metadata[['cord_uid', 'title', 'abstract']]

    metadata = metadata.drop_duplicates('cord_uid').set_index('cord_uid')


def get_nlp(model: str) -> None:
    global nlp

    nlp = spacy.load(model, exclude=[
        'ner',
        'attribute_ruler',
        'lemmatizer',
    ])


def get_docset(document_ids: list[str], cores: int) -> dict[str, DocDetails]:
    result: dict[str, DocDetails] = {}

    if cores != 1:
        with mp.Pool(processes=cores if cores > 0 else None) as pool:
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
    try:
        row = dict(metadata.loc[doc_id])
    except KeyError:
        return None

    return {
        'title': row['title'],
        'abstract': row['abstract'],
    }


def split_section(doc_id: str, text: str, section: str) -> list[Snippet]:
    return [
        {
            'offsetInBeginSection': sentence.start_char,
            'offsetInEndSection': sentence.end_char,
            'text': sentence.text,
            'beginSection': section,
            'endSection': section,
            'document': doc_id,
            'tokens': get_useful_tokens(sentence),
        }
        for sentence in nlp(text).sents
    ]


def get_useful_tokens(sentence: typing.Any) -> list[str]:
    # TODO: spacy does not provide very good type hints. For now, assume we get
    # "Any" and later we can try to better document this function

    return [
        token.text
        for token in sentence
        if useful_token(token)
    ]


def useful_token(token: typing.Any) -> bool:
    return not (
        token.is_bracket or
        token.is_currency or
        token.is_left_punct or
        token.is_right_punct or
        token.is_punct or
        token.is_space or
        token.is_stop
    )


def split_into_snippets(doc_id: str, document: DocDetails) -> list[Snippet]:
    return (
        split_section(doc_id, document['title'], 'title') +
        split_section(doc_id, document['abstract'], 'abstract')
    )


def split_into_snippets_star(item: tuple[str, DocDetails]) -> list[Snippet]:
    return split_into_snippets(*item)


def split_docset_into_sentences(docset: dict[str, DocDetails], cores: int) -> dict[str, list[Snippet]]:
    if cores == 1:
        return {
            paper_id: split_into_snippets(paper_id, paper)
            for paper_id, paper in tqdm(docset.items())
        }

    result: dict[str, list[Snippet]] = {}

    papers = list(docset.items())

    with mp.Pool(processes=cores if cores > 0 else None) as pool:
        partial = pool.imap(split_into_snippets_star, papers)

        for i, snippets in enumerate(tqdm(partial, total=len(papers))):
            result[papers[i][0]] = snippets

    return result


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
        '-c', '--cores', type=int, default=1,
        help='How many cores to use in this process. By default, only one is '
             'used. Higher numbers speed up the process by making the script '
             'use multiple cores of the machine. A value of 0 makes the script '
             'use all cores of the machine.'
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

    document_ids = list({
        doc_id
        for question in galago_results
        for doc_id in question['documents']
    })

    # The following functions create the global `metadata` and `nlp` variables,
    # used by other parts of the script. This is the easiest way I could
    # envision to make the script work with multiprocessing elements
    get_metadata(args.metadata)
    get_nlp(args.nlp_model)

    docset = get_docset(document_ids, args.cores)

    output = split_docset_into_sentences(docset, args.cores)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f)
    else:
        json.dump(output, sys.stdout)


if __name__ == '__main__':
    # Global type hint
    metadata: pd.DataFrame
    nlp: spacy.language.Language

    main()