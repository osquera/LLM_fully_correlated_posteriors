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
#        token mode only:   SKIP_SEQUENCE=1 sh scripts/hpc/submit_pipeline.sh
#        another queue:     QUEUE=gpuh100 sh scripts/hpc/submit_pipeline.sh
#        token settings:    N_ITER=1500 N_SAMPLES=16 sh scripts/hpc/submit_pipeline.sh
#                           (tagged outputs, e.g. posterior_token_it1500_s16.pt;
#                            VEC_BATCH / ROW_BATCH lower memory, see env.sh)
# The variables combine. LOSS_ON=response writes *_response outputs, leaving
# the default run's alone (see env.sh).
# Watch with `bstat`, `bpeek <jobid>`; kill with `bkill <jobid>`.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

# written into every job script (bsub does not pass the environment on)
. scripts/hpc/submit_lib.sh
export LOSS_ON="${LOSS_ON:-all}"
JOB_ENV="LOSS_ON N_ITER N_SAMPLES VEC_BATCH ROW_BATCH"
[ "$LOSS_ON" = "response" ] && adapter=checkpoints/smollm2_lora_response || adapter=checkpoints/smollm2_lora

if [ "${SKIP_FINETUNE:-0}" = "1" ]; then
    [ -d "$adapter" ] || { echo "no $adapter"; exit 1; }
    dep=""
    echo "skipping fine-tune, using $adapter"
else
    j1=$(submit scripts/hpc/01_finetune.lsf)
    dep="-w done($j1)"
    echo "finetune: $j1 (loss on $LOSS_ON)"
fi

if [ "${SKIP_SEQUENCE:-0}" = "1" ] && [ "${SKIP_TOKEN:-0}" = "1" ]; then
    echo "SKIP_SEQUENCE and SKIP_TOKEN leave nothing to sample"; exit 1
fi
waits=""
if [ "${SKIP_SEQUENCE:-0}" = "1" ]; then
    echo "sequence: skipped"
else
    j2=$(submit scripts/hpc/02_sample_sequence.lsf $dep)
    waits="done($j2)"
    echo "sequence: $j2"
fi
if [ "${SKIP_TOKEN:-0}" = "1" ]; then
    echo "token: skipped"
else
    j3=$(submit scripts/hpc/03_sample_token.lsf $dep)
    # ended(), not done(), for the token job: evaluate whatever finished.
    waits="${waits:+$waits && }ended($j3)"
    echo "token: $j3"
fi
wait_for="$waits"

j4=$(submit scripts/hpc/04_evaluate.lsf -w "$wait_for")
j5=$(submit scripts/hpc/05_sweep_tolerance.lsf -w "$wait_for")
echo "evaluate: $j4   sweep: $j5"
