# Given a question, extract a set of non-relevant sentences:
# - directly from the negatives found in the synergy dataset
# - by extracting random sentences from the non-relevant CORD-19 abstracts
#
# We generate `N` question-sentence pairs, using only the questions we have on
# the train dataset, and do it independently: not a fixed number of sentences
# per question but a total of `N` question-senten pairs. In each step, we seect
# a random question to process

from __future__ import annotations

import itertools
import json
import random
import re

import pandas as pd
import spacy
from tqdm.auto import tqdm


def read_json(filename):
    with open(filename) as f:
        return json.load(f)


def process_dataset(dataset) -> dict[str, list[str]]:
    return {
        question['id']: [
            sentence.text
            for snippet in question['snippets']
            for sentence in nlp(snippet['text']).sents
        ]
        for question in tqdm(dataset)
    }


def token_len(sentence):
    return len(re.split(r'[\s.,:;]', sentence))


def get_abstract_sentences() -> list[tuple[str, str]]:
    abstract_sentences = read_json('data/abstract_sentences.json')

    return [
        (sentence, paper_id)
        for paper_id, sentences in tqdm(abstract_sentences.items(), total=len(abstract_sentences))
        for sentence in sentences
        if len(sentence) >= 20 and token_len(sentence) >= 7
    ]


print('Loading spacy\'s NLP pipeline ...')
nlp = spacy.load('en_core_web_lg', exclude=[
    'ner',
    'attribute_ruler',
    'lemmatizer',
])

print('Reading abstract sentences ...')
all_abstract_sentences = get_abstract_sentences()

print('Reading training and negative data ...')
questions_ds = read_json('data/merge.train.json') + read_json('data/merge.test.json')
negatives_ds = read_json('data/merge.negative.json')

print('Splitting snippets into sentences (positive and negative)...')
known_positives = process_dataset(questions_ds)
known_negatives = process_dataset(negatives_ds)


class NegativeGenerator:
    def __init__(self, prob_of_existing=0.5, seed=42):
        self.questions = {
            question['id']: question
            for question in questions_ds
        }

        self.question_ids: list[str] = list(self.questions)

        self.prob_of_existing = prob_of_existing

        self.rand = random.Random(seed)

        self.negative_pool: list[tuple[str, str]] = [
            (self.questions[question_id]['body'], sentence)
            for question_id, sentences in known_negatives.items()
            if question_id in self.questions
            for sentence in sentences
        ]

    def generate(self):
        if self.negative_pool and self.rand.random() < self.prob_of_existing:
            idx = self.rand.randrange(len(self.negative_pool))

            return self.negative_pool.pop(idx)

        # Grab a random question
        question = self.questions[
            self.rand.choice(self.question_ids)
        ]

        while True:
            random_sentence, paper_id = self.rand.choice(
                all_abstract_sentences)

            if paper_id not in question['documents']:
                # TODO: Alternatively, we could accept sentences from relevant
                # papers if they do not overlap with an known golden snippet
                break

        return (question['body'], random_sentence)

    def __iter__(self):
        while True:
            yield self.generate()


# Let's find a number of negative question-sentence pairs equal in number to the
# amount of positive pairs. For that, we need to know how many positive pairs we
# have. That is the number of snippets
generator = NegativeGenerator()

all_negatives = list(itertools.islice(
    generator,
    sum(len(sentences) for sentences in known_positives.values())
))

with open('data/question-sentence-negative-pairs.json', 'w') as f:
    json.dump(all_negatives, f)
