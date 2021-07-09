# SRank - Sentence Ranker for Task Synergy v2 from BioASQ

<div style="font-style:italic;font-size:110%;margin:2rem 0;">
Finding answers to biomedical questions
</div>


## The idea

The high-level description of `SRank` is as follows:

1. given a biomedical question, search in a suitable scientific repository (such as CORD19 or Pubmed) for `M_0` documents, using a BM25-like approach;
2. based on the score progression of these `M_0` retrieved documents, define a threshold `M_1` (a sudden drop in the score, for example);
3. classify each sentence in the retrieved documents as relevant or irrelevant for answering the question;
4. keep only the `N` highest scored sentences, and the respective documents;
5. use a system that can ake as input a question and a set of snippets to produce an exact answer for the question

These steps are here described in more detail


### Step 1

Use a BM25-like indexing procedure to retrieve `M_0` documents from some indexed local installation of CORD19 (or other similar repository). Here `M_0` is a large number, typically `100`.

This retrieval step has some tuning that we can do. For example, I think that merely using a bag-of-words approach is not the best approach.


### Step 2

Study the score patterns of the resuling documents. I expect that the scores will follow a zipf-law decay. This score decay can be informative in order to decide on the number `M_1`, which is the number of documents that will pass to the following step.

The idea here is predict the value `M_1` (specific to each question), training a model with the existing data.

In the worst case, we can make `M_1 = M_0`, which corresponds to this step never running. Therefore, because this step is optional, let's leave its implementation to the end.

**NOTE**: This step is still not implemented, and in fact `M_1 = M_0` in the descriptions that follow.


### Step 3

Train a transformer to detect whether a question and sentence are related. We use the data we have from BioASQ to compile a set of question-sentence pairs. To do so, we look at the questions and the golden snippets, splitting the snippets into sentences. We then generate negative samples by pairing questions with sentences from unrelated abstracts, as well as leverage on the feedback from previous synergy rounds for negative samples.

We sample negative question/sentence pairs from abstracts that *are* retrieved in step 2. That is, we train the model to detect unrelated sentences in the set retrieved by the BM25 approach. This better replicates the expectations of the full system, as the sentences from unrelated abstracts will in theory be more similar to the question than sentences extracted randomly from the full repository.


### Step 4

Split the `M_1` papers (title and abstract) of step 2 into sentences, and run the model from step 3 with all the question/sentence pairs. For each question, keep only the `N` highest scored sentences. These are the snippets choosen for the questions; the documents chosen by the system are then simply the union of the documents the `N` sentences come from. This means that the number of documents might be (and usually is) smaller than `N`.


### Step 5

Use [Lasige's BioASQ9B](https://github.com/lasigeBioTM/BioASQ9B) system to find exact answers to the questions.


## Software requirements

- A local installation of [galago](https://https://sourceforge.net/p/lemur/galago)

- A clone of the [Lasige's BioASQ9B](https://github.com/lasigeBioTM/BioASQ9B) system. Ensure that you download the deep-learning model checkpoints mentioned in that repository.

- Python, version 3.7, with the requirements specific in `requirements.txt`. These are:
  - `pandas`
  - `spacy`
  - `torch`
  - `tqdm`
  - `transformers`


## Data requirements

- A local dump of the scientific repository relevant for the questions in hand. For our purpose, we downlaoded [CORD19](https://www.semanticscholar.org/cord19).

- The following BioASQ golden standards (note that you need a BioASQ participant account to access these):
  - [Tasks 9B](http://participants-area.bioasq.org/Tasks/9b/trainingDataset/)
  - [Synergy v1](http://participants-area.bioasq.org/datasets/download/training_dataset/synergy)


## Workflow

**Commands**
```bash
# Get the galago binary  (TODO)

# Get the repository. In our case, we're using CORD-19  (TODO)

python src/format_cord_for_galago.py \
  --metadata "/path/to/cord/metadata.csv" \
  --destination "/path/to/some/destination"

bash src/build-galago-index.sh
```




**Requirements**:
- [ ] Margarida's code and checkpoints












## System descriptions

System 1

- Use galago to retrieve documents
  - all useful tokens in a single #combine(...) bag
- Split documents into sentences
- Classify the relevance of each sentence to the question
- Sentence score = classification of previous step
- Choose documents based on the 10 highest scoring snippets
- Answer with Margarida's non-fine tuned code


System 2

- Use galago to retrieve documents
  - noun chunks in #sdm(...) constructs
  - other non-chunked tokens
  - all nested in a single #combine(...) bag
- Split documents into sentences
- Classify the relevance of each sentence to the question
- Sentence score = classification of previous step
- Choose documents based on the 10 highest scoring snippets
- Answer with Margarida's non-fine tuned code


System 3

- Use galago to retrieve documents
  - noun chunks in #sdm(...) constructs
  - other non-chunked tokens
  - all nested in a single #combine(...) bag
- Split documents into sentences
- Classify the relevance of each sentence to the question
- Sentence score = classification of previous step * galago score
- Choose documents based on the 10 highest scoring snippets
- Answer with Margarida's non-fine tuned code


System 4

- Use galago to retrieve documents (Use dirichlet score)
  - noun chunks in #sdm(...) constructs
  - other non-chunked tokens
  - all nested in a single #combine(...) bag
- Split documents into sentences
- Classify the relevance of each sentence to the question
- Sentence score = classification of previous step * exp(galago score)
- Choose documents based on the 10 highest scoring snippets
- Answer with Margarida's non-fine tuned code
