# Task Synergy 2 from BioASQ 9

<div style="font-style:italic;font-size:110%;margin:2rem 0;">
Finding answers to biomedical questions
</div>

The main idea is the following:

## Step 1

Use a BM25-like indexing procedure to retrieve `M_0` documents from some indexed local installation of pubmed (or other similar repository). Here `M` is a large number, typically `100`, but I think we can increase a little bit

This retrieval step has some tuning that we can do. For example, I think that merely using a bag-of-words approach is not the best approach.

**Requirements**:
- [x] a local installation of galago
- [x] a local dump of the paper repository
- [x] a galago index of title and abstract of the papers in the repository

**Commands**
```bash
# Get the galago binary  (TODO)

# Get the repository. In our case, we're using CORD-19  (TODO)

python src/format_cord_for_galago.py \
  --metadata "/path/to/cord/metadata.csv" \
  --destination "/path/to/some/destination"

bash src/build-galago-index.sh
```


## Step 2

Study the score patterns of the resuling documents. I expect that the scores will follow a zipf-law decay. This score decay can be informative in order to decide on the number `M_1`, which is the number of documents that will pass to the following step.

The idea here is predict the value `M_1` (specific to each question), training a model with the existing data.

In the worst case, w can make `M_1 = M_0`, which corresponds to this step never running. Therefore, because this step is optional, let's leave its implementation to the end.

**Requirements**:
- [ ] A train/test split of the BioASQ and Synergy data


## Step 3

Train a transformer to detect whether a question and sentence are related. We use the data we have from BioASQ to compile a set of question-sentence pairs. To do so, we look at the questions and the golden snippets, splitting the snippets into sentences. We then generate negative samples by pairing questions with sentences from unrelated abstracts, as well as leverage on the feedback from previous synergy rounds for negative samples.

**Requirements**:
- [x] The negative instances from the golden feedback of previous Synergy tasks
- [x] A sample of negative (unrelated) question-sentence pairs

**Commands**:
```bash
python src/sentence_classification/negative_sample.py

BLA BLA BLA
```

## Step 4

Run the questions through the galago index and extract `M_0` documents. Find the abstracts and split into sentences. Run the model from the previous step to classify all the sentences of the `M_0` abstracts. Those are the snippets that we choose, and so mark the documents the sentences come from as relevant

We could use an idea similar to Step 2 here, where we only classify a document if the score of the previous documents hasn't been steadily decreasing.

**Requirements**:
- [ ] A squad-trained transformer: we want the "BERT-meat" of it, not the final layers
- [x] A train/test split of the BioASQ and Synergy data

**Commands**:
```bash
# Merge Task B and Task Synergy data
python src/merge_split.py \
  data/bioasq-training-9b.json \
  data/bioasq-training-synergy-v1.json
```


## Step 4

Given a question and an abstract, we must extract the relevant snippets. We will do this by leveraging on the self-attention layers between the question and the abstract. We will train on the existing BioASQ and Synergy data to find a way to convert the attention layers into spans of text, which will be the snippets. Snippets will then extend to contain full sentences.

**Requirements**:
- [x] A train/test split of the BioASQ and Synergy data


## Step 5

For a question in the development set (the data from the challange), find $M_0$ documents with Step 1, decide on the value of $M_1$ with Step 2, classify the documents as relevant or not (use a score to sort) and keep the $N$ most relevant documents with Step 3, and extract snippets with Step 4.


## Step 6

Use Margarida's code to find exact answers to the questions

**Requirements**:
- [ ] Margarida's code and checkpoints
