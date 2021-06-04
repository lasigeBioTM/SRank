import argparse
import os
import textwrap

import pandas as pd
from tqdm.auto import tqdm


def format_paper(paper):
    title = paper['title']
    abstract = paper['abstract']

    return textwrap.dedent(f'''\
        <TITLE>{title}</TITLE>
        <TEXT>{abstract}</TEXT>
    ''')


def get_arguments():
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


def read_metadata(filename):
    metadata = pd.read_csv(filename, low_memory=True)

    # We want only papers with title, abstract and pubmed_id
    metadata = metadata[
        pd.notna(metadata['title']) &
        pd.notna(metadata['abstract']) &
        pd.notna(metadata['pubmed_id'])
    ].drop_duplicates('cord_uid')

    return metadata


def main():
    args = get_arguments()

    os.makedirs(args.destination)

    metadata = read_metadata(args.metadata)

    for _, paper in tqdm(metadata.iterrows(), total=len(metadata)):
        uid = paper['cord_uid']

        subdir_path = os.path.join(args.destination, uid[0], uid[1])

        os.makedirs(subdir_path, exist_ok=True)

        with open(os.path.join(subdir_path, f'{uid}.txt'), 'w') as f:
            f.write(format_paper(paper))


if __name__ == '__main__':
    main()
