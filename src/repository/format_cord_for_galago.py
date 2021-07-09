import argparse
import os
import textwrap
import typing

import pandas as pd
from tqdm.auto import tqdm

# This script reads all the title and abstract of each CORD19 paper and produces
# a text file containing those details for each paper. Notice that Galago can
# consume files in a variety of formats. Their documentation, in particular
# https://sourceforge.net/p/lemur/wiki/Galago%20Indexing/, provides a (slightly
# underspecified) overview of the possible formats. Since I did not fully
# understand the way specialized formats work, I decided to use the text format,
# in which each document is represented as an actual text file, which includes
# the text of the document, optioanlly tagged with relevant HTML-like markers.

# Because (a) this schema requires that each document is its own file, and (b)
# file systems are not known to behave nicely when directories contain thousands
# of entries, I split the full content of the CORD19 repository in a recursive
# tree structure. For example, a file whose ID is "jFbUzuDhO0" will be placed in
# `/destination/j/F/jFbUzuDhO0.txt`. CORD19 (the version form 2021-05-31)
# contains 214494 documents, and the IDS are all strings of digits, upper-case
# and lower-case letters; thus, this schema splits the repository into 36**2 =
# 1296 subdirectories. Therefore, each subdirectory, on average, contains about
# 166 files, which is OK for most modern file systems.


def format_paper(paper: pd.Series) -> str:
    # For my purpose, I include only text and abstract in the file, but you can
    # include other details here as well

    title: str = paper['title']
    abstract: str = paper['abstract']

    return title + '\n' + abstract


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Format title and abstract of the CORD 19 papers '
                    'and save them in a destination directory in text format'
    )

    parser.add_argument(
        '-m', '--metadata',
        default='/mnt/data/jferreira_data/cord/2021-05-24/metadata.csv',
    )

    parser.add_argument(
        '-d', '--destination',
        default='/mnt/data/jferreira_data/cord/2021-05-24/to_index',
    )

    return parser.parse_args()


def read_metadata(filename: str) -> pd.DataFrame:
    metadata: pd.DataFrame = pd.read_csv(filename, low_memory=False)

    # We want only papers with title, abstract and pubmed_id
    metadata = metadata[
        pd.notna(metadata['title']) &
        pd.notna(metadata['abstract']) &
        pd.notna(metadata['pubmed_id'])
    ].drop_duplicates('cord_uid')

    return metadata


def main() -> None:
    args = get_arguments()

    os.makedirs(args.destination)

    metadata = read_metadata(args.metadata)

    for _, paper in tqdm(metadata.iterrows(), total=len(metadata)):
        # This is where the splitting in a recursive tree happens. For a paper
        # with a specific ID, we place the formatted text file into a
        # subdirectory that is based on the ID, ensuring that this subdirectory
        # exists

        uid: str = paper['cord_uid']

        subdir_path = os.path.join(args.destination, uid[0], uid[1])

        os.makedirs(subdir_path, exist_ok=True)

        with open(os.path.join(subdir_path, f'{uid}.txt'), 'w') as f:
            f.write(format_paper(paper))


if __name__ == '__main__':
    main()
