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

- A local installation of [Galago](https://https://sourceforge.net/p/lemur/galago)

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

1. Get the Galago binary (for example, the [3.20 version](https://sourceforge.net/projects/lemur/files/lemur/galago-3.20/)). Untar and place the `galago-*-bin` directory directly inside the main directory of this repository.

1. Get the repository. In case you are working with CORD19, you should follow the instructions in [Semantic Scholar's CORD19 Download page](https://www.semanticscholar.org/cord19/download) or in [their AWS historical releases page](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html), and download the most recent metadata file. As an example, you can download the [2021-07-05 metadata file](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-07-05/metadata.csv). You can place this in the directory you want.

1. Format the repository in a way that can be understood by Galago. The `src/repository/format_cord_for_galago.py` script is prepared to format the CORD19 repository in a way that Galago can consume. If you want to format other repostories, use that script as an example of what needs to be done. Comments there can be guide you if you need some assistence. For our use case, the command was:

    ```bash
    python src/repository/format_cord_for_galago.py \
      --metadata /path/to/metadata.csv \
      --destination /path/to/galago-index
    ```

    You can chose whatever destination directory you want, as long as it does not exist.

1. Edit the `config/galago-build-params.json` parameters so that the `"inputPath"` and `"indexPath"` properties point to where you placed your formatted documents and the place where you want the index to be built. In other words, `"inputPath"` is the directory chosen in the previous step, and `"indexPath"` is the (non-existing) directory where you want to place your index.

1. Run `bash build_galago_index.sh`. This will take some time, particularly if you have hundreds of thousands of documents, and big documents to index. You should be able to monitor the state of the build in http://localhost:54321.

1. Follow the golden standard links above to get a copy of the BioASQ answers from previous tasks. The Synergy task files come as a ZIP file, which you must unzip. The important files in there are `golden_round_4.json` and `feedback_final.json`. The file for task 9B is named `bioasq-training-9b.json`. Place these three files in the `data/` directory.

1. Download the challenge file, containing the questions you want to answer. Save it as `data/testset.json`.

1. Clone the [BioASQ9B system](https://github.com/lasigeBioTM/BioASQ9B/) and consult the README file, and then follow the instructions there to download the model checkpoint files into the appropriate directories. The necessary checkpoints are `checkpoint_bio_yn.pt`, `checkpoint_list.pt` and `checkpoint_factoid.pt`.

1. Follow the code below, which will run `SRank`, produce a variety of intermediate files, and culminates in the creation of a file that associated to each question a set of papers and snippets.

    Notice that all the python scripts have multiple command line arguments, which you can explore either by reading the source or running `python /path/to/script.py --help`

    ```bash
    # Because the BioASQ Task 9 uses Pubmed IDs to identify documents, the
    # first step is to convert the training file so that it uses CORD19
    # identifiers. If you are dealing with a dataset that contains document identifiers from multiple repositories, converting them to the correct format is always going to be the first step.

    python src/repository/to_cord.py \
      data/bioasq-training-9b.json \
      /path/to/metadata.csv \
      --output data/bioasq-training-9b-cord-ids.json

    # Start by running the retrieve step on the training data

    python src/retrieve/retrieve.py \
      data/feedback_final.json \
      /path/to/galago-index \
      --requested 100 \
      --scorer bm25 \
      --output results/feedback_final-galago-results.json

    python src/retrieve/retrieve.py \
      data/bioasq-training-9b.json \
      /path/to/galago-index \
      --requested 100 \
      --scorer bm25 \
      --output results/bioasq-training-9b-galago-results.json

    # Create a docset (split documents into sentences) for the retrieved
    # documents. We could, in theory, split all papers, but that would take
    # a very long time and we only need a few of them anyway. Notice that the
    # --cores flag lets you select how many cores of your machine to use. Here
    # we set it to 20 because we had that many cores on our machine.

    python src/retrieve/make_docset.py \
      results/feedback_final-galago-results.json \
      /path/to/metadata.csv \
      --cores 20 \
      --output results/feedback_final-docset.json

    python src/retrieve/make_docset.py \
      results/bioasq-training-9b-galago-results.json \
      /path/to/metadata.csv \
      --cores 20 \
      --output results/bioasq-training-9b-docset.json

    # For each of these training sets, let's find unrelated sentences for each
    # question

    python src/sentence_classification/negative_sample.py \
      data/golden_round_4.json \
      results/feedback_final-docset.json \
      --ratio 1 \
      --output results/feedback_final-negative-qs-pairs.json

    python src/sentence_classification/negative_sample.py \
      data/bioasq-training-9b-cord-ids.json \
      results/bioasq-training-9b-docset.json \
      --ratio 1 \
      --output results/bioasq-training-9b-negative-qs-pairs.json

    # Merge all related and unrelated question/sentences pairs into a
    # single dataset, and then use that to train the QS classification model.

    # Please study the various arguments that you can provide to the `train.py`
    # script. Use that knowledge to influence the training process. In
    # particular, you should change the flags:
    #   --per-device-train-batch-size
    #   --per-device-eval-batch-size
    #   --log-steps
    #   --eval-steps
    #   --save-steps
    #   --fp16
    # in order to tune the training process to your system capabilities.
    # Also, if necessary, use the CUDA_VISIBLE_DEVICES environment variable
    # to instruct pytorch to use only a subset of the GPUs of the system.

    python src/sentence_classification/merge_qs.py \
      --positive \
        data/feedback_final.json \
        data/bioasq-training-9b.json \
      --negative \
        results/feedback_final-negative-qs-pairs.json \
        results/bioasq-training-9b-negative-qs-pairs.json \
      --output results/qs-train-dataset.json

    python src/sentence_classification/train.py \
      results/qs-train-dataset.json \
      --output-dir models/qs-model

    # Retrieve the documents for the particular questions in the challenge

    python src/retrieve/retrieve.py \
      data/testset.json \
      --requested 100 \
      --scorer bm25 \
      --output results/testset-galago-results.json

    python src/retrieve/make_docset.py \
      results/testset-galago-results.json \
      /path/to/metadata.csv \
      --cores 20 \
      --output results/testset-docset.json

    # Rank the sentences with respect to the question they are answering to.
    # Since we are using the deep-learning model here, you should also take care
    # to use the CUDA_VISIBLE_DEVICES environment variable if needed.

    python src/sentence_classification/rank_sentences.py \
      results/testset-galago-results.json \
      results/testset-docset.json \
      --keep-upper-case \
      --transformer-model models/qs-model \
      --output results/testset-ranked-sentences.jsonl

    # For each question, choose the 10 highest-ranking snippets

    python src/sentence_classification/choose_snippets.py \
      data/testset.json \
      results/testset-ranked-sentences.jsonl \
      --top 10 \
      --output results/testset-snippets.json

    # This produces a file that, for each question, contains a set of (at most)
    # 10 snippets. We'll use it with Lasige's BioASQ9B system to produce exact
    # answers

    python src/exact_answers.py \
      results/testset-snippets.json \
      /path/to/lasige/bioasq9/ \
      results/testset-exact-answers.json
    ```

## Variations

The system has been tried in 4 different variations, which can be executed by selecting different options at each step. There are 4 variations.


### Variation 1

All the default options are used. The system, running from the code above, should result in this variation. In particular, in this system:

- Documents are searched for in the galago index using a bag-of-words approach, where each word in the question weighs as much as any other, and their proximity in the question is entirely ignored

- The score of each sentence, with respect to the question, is the raw output logit output of the qs-model trained in Step 3.


### Variant 2

In this system:

- Noun chunks in the question are grouped in #sdm(...) constructs. This is achieved by passing the `--noun-chunk sdm` command line argument to the `src/retrieve/retrieve.py` script.

- The score of each sentence, with respect to the question, is the raw output logit output of the qs-model trained in Step 3.


### Variant 3

- Noun chunks in the question are grouped in #sdm(...) constructs. This is achieved by passing the `--noun-chunk sdm` command line argument to the `src/retrieve/retrieve.py` script.

- The score of each sentence, with respect to the question, is the raw output logit output of the qs-model trained in Step 3 multiplied by the galago score of the document from where the sentence came from. This is achieved by passing the `--multiply-document` and `--galago-results results/testset-galago-results.json` command line arguments to the `src/sentence_classification/choose_snippet.py` script.


### Variant 4

- The galago index search is performed not with the BM25 score, but with the Dirichlet score. This is achieved by omitting the `--scorer bm25` command line argument from the `src/retrieve/retrieve.py` script.

- Noun chunks in the question are grouped in #sdm(...) constructs. This is achieved by passing the `--noun-chunk sdm` command line argument to the `src/retrieve/retrieve.py` script.

- The score of each sentence, with respect to the question, is the raw output logit output of the qs-model trained in Step 3 multiplied by the galago score of the document from where the sentence came from. Because the galago score follows a logarithmic scale (that is thebehaviour of the Dirichlet formula), that score must first be exponentiated to a linear scale. All of this is achieved by passing the `--multiply-document`, `--galago-results results/testset-galago-results.json` and `--exponentiate` command line arguments to the `src/sentence_classification/choose_snippet.py` script.
