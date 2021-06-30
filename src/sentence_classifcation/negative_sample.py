# We need to convert the following into a dataset of question/sentence pairs
# from the BioASQ synergy feedback files, which can be used to derive positive
# and negative pairs.

# Search each question with galago and get the abstract of the papers that are
# not related to the question. Sample the sentences that are more highly related
# to the questions based on the CLS embedding of the sentences and a cosine
# similarity of the vectors.

# Keep only a fraction of the negative sentences so that the overall ratio of
# positive to negative instances is some defined constant

# These options can be tuned by command line flags

# For this to work, we need to have the data above in hand. We also need to run
# all questions through the galago index before in order to gather a few
# unrelated documents before hand (see `src/retrieve.py`)

import argparse
import json


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sample a selection of relevant and non-relevant sentences '
                    'to go along with each question in order to create a '
                    'dataset of question/sentence pairs that can be used to '
                    'train a transformer model in order to identify relevant '
                    'abstracts snippets to answer a question.'
    )

    parser.add_argument(
        'questions', metavar='QUESTIONS',
        help='The file containing the training questions. The file must be in '
             'the BioASQ format, and must contain a list of snippets and a '
             'list of relevant documents for each question. Snippets in this '
             'file are considered positive instances, unless they are marked '
             'with the `"golden": false` property. Non-golden snippets are '
             'used as negative instances (see the `--sample` and `--ratio` '
             'flags).'
    )

    parser.add_argument(
        'docset', metavar='DOCSET',
        help='The snippets extracted from a set of documents. This can be '
             'created with the `src/make_docset.py` file. These snippets '
             'will be used to sample negative instances. You can provide '
             'a dummy argument (eg. "-") if you know that there are enough '
             'negative instances in the input to not require sampling.'
    )

    parser.add_argument(
        '-o', '--only-generate', action='store_true',
        help='By default, this script gathers negative instances from the '
             'input (if any exists). By providing this flag, the script skips '
             'these known instances, ensuring that all the results are sampled '
             'from the docset.'
    )

    parser.add_argument(
        '-r', '--ratio', type=float, default=4,
        help='The amount of negative instances generated for each positive '
             'instance. Each positive question-snippet pair is counted, so '
             'the result contains each question with approximately the same '
             'frequency that it appears in the input.'
    )

    parser.add_argument(
        '-o', '--output',
        help='Where to place the output. Defaults to standard output.'
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()


if __name__ == '__main__':
    main()
