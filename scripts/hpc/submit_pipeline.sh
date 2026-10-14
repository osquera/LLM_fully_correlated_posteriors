#!/bin/sh
# Submit the whole pipeline to DTU HPC (LSF) with dependencies:
#
#   01_finetune --> 02_sample_sequence --\
#              \--> 03_sample_token -------> 04_evaluate --> 05_sweep_tolerance
#
# Usage (from anywhere):   sh scripts/hpc/submit_pipeline.sh
#        reuse theta_map:   SKIP_FINETUNE=1 sh scripts/hpc/submit_pipeline.sh
#        response only:     LOSS_ON=response sh scripts/hpc/submit_pipeline.sh
#        no token mode:     SKIP_TOKEN=1 sh scripts/hpc/submit_pipeline.sh  (~22 h saved)
# The variables combine. LOSS_ON=response writes *_response outputs, leaving
# the default run's alone (see env.sh).
# Watch with `bstat`, `bpeek <jobid>`; kill with `bkill <jobid>`.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

# bsub hands the submitting environment to the job, so env.sh sees these
export LOSS_ON="${LOSS_ON:-all}"
[ "$LOSS_ON" = "response" ] && adapter=checkpoints/smollm2_lora_response || adapter=checkpoints/smollm2_lora

submit() {  # prints the job id
    bsub "$@" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p'
}

if [ "${SKIP_FINETUNE:-0}" = "1" ]; then
    [ -d "$adapter" ] || { echo "no $adapter"; exit 1; }
    dep=""
    echo "skipping fine-tune, using $adapter"
else
    j1=$(submit < scripts/hpc/01_finetune.lsf)
    dep="-w done($j1)"
    echo "finetune: $j1 (loss on $LOSS_ON)"
fi

j2=$(submit $dep < scripts/hpc/02_sample_sequence.lsf)
if [ "${SKIP_TOKEN:-0}" = "1" ]; then
    wait_for="done($j2)"
    echo "sequence: $j2   token: skipped"
else
    j3=$(submit $dep < scripts/hpc/03_sample_token.lsf)
    # ended(), not done(), for the token job: evaluate whatever finished.
    wait_for="done($j2) && ended($j3)"
    echo "sequence: $j2   token: $j3"
fi

j4=$(submit -w "$wait_for" < scripts/hpc/04_evaluate.lsf)
j5=$(submit -w "$wait_for" < scripts/hpc/05_sweep_tolerance.lsf)
echo "evaluate: $j4   sweep: $j5"
