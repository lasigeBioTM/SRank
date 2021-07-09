import os
import argparse
import subprocess


def run_yes_no(cwd: str, snippets_filename: str) -> None:
    print('Processing yes/no questions')

    process = subprocess.Popen(
        cwd=os.path.join(cwd, 'yesno'),
        args=[
            'python',
            'predict_yesno.py',
            '--model_name=dmis-lab/biobert-base-cased-v1.1',
            '--checkpoint_input_path=../checkpoint/checkpoint_bio_yn.pt',
            '--predictions_output_path=predictions/pred_test.csv',
            f'--questions_path={snippets_filename}',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    process.communicate()


def run_list(cwd: str, snippets_filename: str) -> None:
    print('Processing list questions')

    process = subprocess.Popen(
        cwd=os.path.join(cwd, 'list'),
        args=[
            'python',
            'predict_list.py',
            '--model_name=dmis-lab/biobert-base-cased-v1.1',
            '--checkpoint_input_path=../checkpoint/checkpoint_list.pt',
            '--predictions_output_path=predictions/pred.csv',
            f'--questions_path={snippets_filename}',
            '--k_candidates=5',
            '--k_elected=2',
            '--voting=stv',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    process.communicate()


def run_factoid(cwd: str, snippets_filename: str) -> None:
    print('Processing factoid questions')

    process = subprocess.Popen(
        cwd=os.path.join(cwd, 'list'),
        args=[
            'python',
            'predict_factoid.py',
            '--model_name=dmis-lab/biobert-base-cased-v1.1',
            '--checkpoint_input_path=../checkpoint/checkpoint_factoid.pt',
            '--predictions_output_path=predictions/pred.csv',
            f'--questions_path={snippets_filename}',
            '--k_candidates=5',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    process.communicate()


def merge(cwd: str, snippets_filename: str, output: str) -> None:
    process = subprocess.Popen(
        cwd=os.path.join(cwd, 'list'),
        args=[
            'python',
            'format_bioasq.py',
            f'--questions_path={snippets_filename}',
            '--list_predictions_path=list/predictions/pred.csv',
            '--factoid_predictions_path=factoid/predictions/pred.csv',
            '--yesno_predictions_path=yesno/predictions/pred_test.csv',
            f'--output_path={output}',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    process.communicate()


def absolute_path(path: str) -> str:
    if path.startswith('/'):
        return path
    else:
        return os.path.join(os.getcwd(), path)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply Lasige\'s BioASQ 9B system to the set of snippets '
                    'extracted with the SRank system in order to find exact '
                    'answers for the questions.\n\n'
                    'Notice that all filename will be resolved with respect to '
                    'the current directory, unless they start with a forward '
                    'slash character (`/`).'
    )

    parser.add_argument(
        'snippets', metavar='SNIPPETS', type=absolute_path,
        help='The snippets file produced by the '
             '`src/sentence_classification/choose_snippets.py` script.'
    )

    parser.add_argument(
        'bioasq_path', metavar='BIOASQ_PATH', type=absolute_path,
        help='The path to Lasige\'s BioASQ 9B system, which you should have '
             'already cloned if you are following this repository instructions.'
    )

    parser.add_argument(
        'output', metavar='OUTPUT', type=absolute_path,
        help='The filename where you want to save the output.'
    )

    return parser.parse_args()


def main() -> None:
    args = get_arguments()

    run_yes_no(args.bioasq_path, args.snippets)
    run_list(args.bioasq_path, args.snippets)
    run_factoid(args.bioasq_path, args.snippets)

    merge(args.bioasq_path, args.snippets, args.output)


if __name__ == '__main__':
    main()
