set -e

# TODO: Make this a python script that spawns the children processes to handle
# all the necessary steps. We make it python so that we can provide nice command
# line arguments, including:
# - the path to the bioasq system
# - the filename of the phase-a result
# - an output filename

# Change the next line to point to the local clone of BioASQ9B
BIOASQ_PATH="/home/jferreira/workspace/bus/BioASQ9B"

HERE="$(pwd)"

cd "$BIOASQ_PATH"

echo 'Processing yes/no questions'
(
    cd yesno

    python predict_yesno.py \
        --model_name=dmis-lab/biobert-base-cased-v1.1 \
        --checkpoint_input_path="$BIOASQ_PATH"/checkpoint/checkpoint_bio_yn.pt \
        --predictions_output_path=predictions/pred_test.csv \
        --questions_path="$HERE"/results/testset-phase-a.json \
        2>/dev/null
)

echo 'Processing list questions'
(
    cd list

    python predict_list.py \
        --model_name=dmis-lab/biobert-base-cased-v1.1 \
        --checkpoint_input_path="$BIOASQ_PATH"/checkpoint/checkpoint_list.pt \
        --predictions_output_path=predictions/pred.csv \
        --questions_path="$HERE"/results/testset-phase-a.json \
        --k_candidates=5 \
        --k_elected=2 \
        --voting=stv \
        2>/dev/null
)

echo 'Processing factoid questions'
(
    cd factoid

    python predict_factoid.py \
        --model_name=dmis-lab/biobert-base-cased-v1.1 \
        --checkpoint_input_path="$BIOASQ_PATH"/checkpoint/checkpoint_factoid.pt \
        --predictions_output_path=predictions/pred.csv \
        --questions_path="$HERE"/results/testset-phase-a.json \
        --k_candidates=5 \
        2>/dev/null
)

python format_bioasq.py \
    --questions_path="$HERE"/results/testset-phase-a.json \
    --list_predictions_path=list/predictions/pred.csv \
    --factoid_predictions_path=factoid/predictions/pred.csv \
    --yesno_predictions_path=yesno/predictions/pred_test.csv \
    --output_path="$HERE"/results/testset-exact-answers.json
