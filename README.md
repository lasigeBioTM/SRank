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

1. Edit the `config/galago-build-params.json` parameters so that the `"inputPath"` and `"indecPath"` properties point to where you placed your formatted documents and the place where you want the index to be built. In other words, `"inputPath"` is the directory chosen in the previous step, and `"indexPath"` is the (non-existing) directory where you want to place your index.

1. Run `bash build_galago_index.sh`. This will take some time, particularly if you have hundreds of thousands of documents, and big documents to index. You should be able to monitor the state of the build in http://localhost:54321.

1. Follow the golden standard links above to get a copy of the BioASQ answers from previous tasks. Place the files in the `data/` directory.

1. Download the challenge file, containing the questions you want to answer. Place it in the `data/` directory.

1. Clone the [BioASQ9B system](https://github.com/lasigeBioTM/BioASQ9B/) and consult the README file, and then follow the instructions there to download the model checkpoint files into the appropriate directories.

1. Follow the code below, which will run `SRank`, produce a variety of intermediate files, and culminates in the creation of a file that associated to each question a set of papers and snippets.

    Notice that all the python scripts have multiple command line arguments, which you can explore either by reading the source or running `python /path/to/script.py --help`

    ```bash
    python src/
    ```

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
