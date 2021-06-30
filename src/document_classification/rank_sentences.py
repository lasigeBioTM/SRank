import argparse
import json
import math

import torch
import transformers
import spacy
from tqdm.auto import tqdm


class Ranker:
    def __init__(
        self,
        tokenizer,
        model,
        device,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device


    def find_batch_size(self):
        # Sensible default values to start the search
        bottom = 0
        top = 100

        # Find a batch size that blows the device RAM. While doing that, keep a
        # record of the lowest bound
        while self.batch_size_works(top):
            bottom = top
            top *= 2

        # Binary search: half the range each time, and depending on whether we
        # can use the middle value as batch size, adjust the bounds. When we
        # pinpoint a value, use it as our batch size.
        while top - bottom >= 2:
            mid = int((top + bottom) / 2)

            if self.batch_size_works(mid):
                bottom = mid
            else:
                top = mid

        return bottom


    @torch.no_grad()
    def batch_size_works(self, batch_size: int):
        print(f'  trying {batch_size} ...', end='')

        try:
            self.model(**self.tokenize(
                ['is this a question?'] * batch_size,
                ['yes it is'] * batch_size
            ))
        except RuntimeError:
            print(' does not work ✘')
            return False
        else:
            print(' works ✔')
            return True


    def tokenize(self, question, sentence):
        tokens = self.tokenizer(
            question,
            sentence,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokens['input_ids'].to(self.device),
            'token_type_ids': tokens['token_type_ids'].to(self.device),
            'attention_mask': tokens['attention_mask'].to(self.device),
        }

    @torch.no_grad()
    def score_question_snippet_pair(self, question, snippet_text):
        return self.model(**self.tokenize(question, snippet_text)).logits.flatten().tolist()


class TokenCounter:
    def __init__(self, nlp):
        self.nlp = nlp

    def count_tokens(self, text):
        doc = self.nlp(text, disable=[
            'tagger',
            'parser',
            'ner',
            'attribute_ruler',
            'lemmatizer',
        ])

        return sum(1 for token in doc if not self.ignore_token(token))


    def ignore_token(self, token):
        return (
            token.is_bracket or
            token.is_currency or
            token.is_left_punct or
            token.is_right_punct or
            token.is_punct or
            token.is_space or
            token.is_stop
        )


    def count_chars(self, text):
        return len(text)


def chunker(sequence, chunk_size):
    return (
        sequence[pos:pos + chunk_size]
        for pos in range(0, len(sequence), chunk_size)
    )


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Rank the sentences retrieved for a given question based '
                    'on how well they contain the answer to that question. '
                    'This produces a .jsonl file, where each line contains a '
                    'batch of the results. Each item in the result is a '
                    'question/snippet pair along with a score.'
    )

    parser.add_argument(
        'retrieved', metavar='RETRIEVED',
        help='The file containing the results retrieved with the '
             '`src/retrieve.py` script.'
    )

    parser.add_argument(
        'docset', metavar='DOCSET',
        help='The file containing the docset split into sentences '
             'corresponding to the results. This is the result of applying '
             'the `src/document_classification/sentence_splitter.py` script '
             'to the docset generated by the `src/retrieve.py` script.'
    )

    parser.add_argument(
        '-o', '--output', default='results/output.jsonl',
        help='The filename of the output. Defaults to "results/output.jsonl".'
    )

    parser.add_argument(
        '-t', '--tokenizer', default='dmis-lab/biobert-base-cased-v1.1',
        help='The hugging-face tokenizer to use to evaluate question/sentence '
             'pairs. Defaults to "dmis-lab/biobert-base-cased-v1.1".'
    )

    parser.add_argument(
        '-l', '--keep-upper-case', action='store_true',
        help='Whether to keep the case on the hugging-face front of the '
             'script. By default, the case is lowered. Consult the '
             'documentation for the tokenizer you are using to see whether you '
             'want this.'
    )

    parser.add_argument(
        '-m', '--transformer-model',
        default='models/qs-model/checkpoint-2000',
        help='The hugging-face model to use to score the question/sentence '
             'pairs. Defaults to "models/qs-model/checkpoint-2000", which only '
             'exists after the `src/sentence_classification/train.py` script '
             'is executed.'
    )

    parser.add_argument(
        '-d', '--device', default='cuda',
        help='The device where the model will execute. Options are "cpu", '
             '"cuda", "cuda:N", etc. Defaults to "cuda"'
    )

    parser.add_argument(
        '-b', '--batch-size', type=int,
        help='The batch size to use. If not given, the script pinpoints the '
             'highest batch size that is supported by the device. Attention: '
             'I do not know what could go wrong if you do NOT specify this '
             'when using the CPU.'
    )

    parser.add_argument(
        '-n', '--nlp-model', default='en_core_web_lg',
        help='The spacy model to use for filtering sentences based on token '
             'count. Defaults to "en_core_web_lg".'
    )

    parser.add_argument(
        '-T', '--min-token-length', type=int, default=8,
        help='The minimum size of a sentence, measured in tokens, to be ranked. '
             'Defaults to 8.'
    )

    parser.add_argument(
        '-C', '--min-chars-length', type=int, default=16,
        help='The minimum size of a sentence, measured in characters, to be ranked. '
             'Defaults to 16.'
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    print('Loading tokenizer ...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer,
        do_lower_case=not args.keep_upper_case,
    )

    print('Loading transformer model ...')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.transformer_model,
        num_labels=1,
    ).to(args.device)

    ranker = Ranker(tokenizer, model, args.device)

    print('Loading spacy language model ...')
    nlp = spacy.load(args.nlp_model, exclude=[
        'tagger',
        'parser',
        'ner',
        'attribute_ruler',
        'lemmatizer',
    ])

    token_counter = TokenCounter(nlp)

    print('Loading data ...')
    with open(args.retrieved) as f:
        galago_results = json.load(f)

    with open(args.docset) as f:
        paper_snippets = json.load(f)

    print('Pairing questions with paper sentences and filtering small sentences ...')
    question_snippet_pairs = [
        (query, snippet)
        for query in tqdm(galago_results['queries'])
        for document in query.get('retrieved_documents', [])
        for snippet in paper_snippets[document['doc_id']]
        if (
            token_counter.count_tokens(snippet['text']) >= args.min_token_length and
            token_counter.count_chars(snippet['text']) >= args.min_chars_length
        )
    ]

    if args.batch_size is None:
        print('Finding highest batch size ...')
        batch_size = ranker.find_batch_size()
    else:
        batch_size = args.batch_size

    chunks = chunker(question_snippet_pairs, batch_size)
    num_chunks = math.ceil(len(question_snippet_pairs) / batch_size)

    print('Scoring question/sentence pairs ...')

    output_handle = open(args.output, 'w')

    for chunk in tqdm(chunks, total=num_chunks):
        queries, snippets = zip(*chunk)

        scores = ranker.score_question_snippet_pair(
            [query['query_text'] for query in queries],
            [snippet['text'] for snippet in snippets]
        )

        items = [{
            'query_id': query['query_id'],
            'snippet': snippet,
            'score': score
        } for query, snippet, score in zip(queries, snippets, scores)]

        output_handle.write(json.dumps(items))
        output_handle.write('\n')
        output_handle.flush()

    output_handle.close()


if __name__ == '__main__':
    main()