#!/bin/sh
# Token-mode error bars: two ~22 h projections (seeds 1, 2) in parallel, then
# one evaluation job over seeds 0-2 once both have ended (done or not).
# Usage:  sh scripts/hpc/submit_token_seeds.sh     (LOSS_ON=response sh ... too)
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs
. scripts/hpc/submit_lib.sh
export LOSS_ON="${LOSS_ON:-all}"
JOB_ENV="LOSS_ON"  # written into the job scripts (bsub does not pass it on)

j1=$(submit scripts/hpc/08_token_seeds.lsf)
j2=$(submit scripts/hpc/09_token_seeds_eval.lsf -w "ended($j1)")
echo "token seeds: $j1 (array)   evaluate: $j2   (loss on $LOSS_ON)"
