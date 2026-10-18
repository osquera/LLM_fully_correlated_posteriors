#!/bin/sh
# Token-mode error bars: two ~22 h projections (seeds 1, 2) in parallel, then
# one evaluation job over seeds 0-2 once both have ended (done or not).
# Usage:  sh scripts/hpc/submit_token_seeds.sh     (LOSS_ON=response sh ... too)
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs
export LOSS_ON="${LOSS_ON:-all}"

submit() {  # prints the job id
    bsub "$@" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p'
}

j1=$(submit < scripts/hpc/08_token_seeds.lsf)
j2=$(submit -w "ended($j1)" < scripts/hpc/09_token_seeds_eval.lsf)
echo "token seeds: $j1 (array)   evaluate: $j2   (loss on $LOSS_ON)"
