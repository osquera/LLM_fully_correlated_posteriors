# Sourced by the QA jobs (scripts/hpc/qa/): fine-tune and project on
# short-answer QA (TriviaQA) instead of Alpaca, with loss on the answer only
# and LoRA on every linear layer, so the posterior covers more of the network.
#
# SIZE picks the base model: 135M (default) or 360M. Every output carries the
# tag qa_<SIZE>, so the Alpaca runs' files are left alone.

LOSS_ON=response
UV_SYNC="${UV_SYNC:-0}" . scripts/hpc/env.sh

# No default: submit_qa.sh writes SIZE into each job. A job without it would
# otherwise run the wrong model and overwrite the other size's files.
case "${SIZE:-}" in
    135M|360M) export SIZE ;;
    *) echo "SIZE must be 135M or 360M, got '${SIZE:-}' (submit with scripts/hpc/qa/submit_qa.sh)"; exit 1 ;;
esac
export BASE_MODEL="HuggingFaceTB/SmolLM2-$SIZE"
export TAG="qa_$SIZE"
export ADAPTER="checkpoints/smollm2_$TAG"
export PARTS_DIR="samples/parts_$TAG"
export SAMPLES_QA="samples/posterior_token_$TAG.pt"

# data: fine-tune on QA_TRAIN, project on its first N_TRAIN examples; the
# semantic test uses TriviaQA *validation*, never seen in either
export QA_DATASET=triviaqa
export QA_TRAIN="train[:5000]"
export N_TRAIN=512
export MAX_LEN=128
export LORA_TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
mkdir -p "$PARTS_DIR"
echo "QA run: base=$BASE_MODEL adapter=$ADAPTER samples=$SAMPLES_QA"
