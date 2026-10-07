#!/bin/sh
# Submit the whole pipeline to DTU HPC (LSF) with dependencies:
#
#   01_finetune --> 02_sample_sequence --\
#              \--> 03_sample_token -------> 04_evaluate
#
# Usage (from anywhere):   sh scripts/hpc/submit_pipeline.sh
#        reuse theta_map:   SKIP_FINETUNE=1 sh scripts/hpc/submit_pipeline.sh
# Watch with `bstat`, `bpeek <jobid>`; kill with `bkill <jobid>`.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

submit() {  # prints the job id
    bsub "$@" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p'
}

if [ "${SKIP_FINETUNE:-0}" = "1" ]; then
    [ -d checkpoints/smollm2_lora ] || { echo "no checkpoints/smollm2_lora"; exit 1; }
    dep=""
    echo "skipping fine-tune, using checkpoints/smollm2_lora"
else
    j1=$(submit < scripts/hpc/01_finetune.lsf)
    dep="-w done($j1)"
    echo "finetune: $j1"
fi

j2=$(submit $dep < scripts/hpc/02_sample_sequence.lsf)
j3=$(submit $dep < scripts/hpc/03_sample_token.lsf)
echo "sequence: $j2   token: $j3"

# ended(), not done(), for the token job: evaluate whatever finished.
j4=$(submit -w "done($j2) && ended($j3)" < scripts/hpc/04_evaluate.lsf)
echo "evaluate: $j4"
