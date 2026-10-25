#!/bin/sh
# Submit the QA pipeline with dependencies:
#
#   01_finetune --> 02_sample_part[1..PARTS] (parallel) --> 03_merge_eval
#
# Usage (from anywhere):  sh scripts/hpc/qa/submit_qa.sh
#        360M model:       SIZE=360M sh scripts/hpc/qa/submit_qa.sh
#        other queue:      QUEUE=gpul40s sh scripts/hpc/qa/submit_qa.sh
#        reuse theta_map:  SKIP_FINETUNE=1 sh scripts/hpc/qa/submit_qa.sh
#        more parts:       PARTS=6 ...   (2 posterior samples per part)
# Outputs carry the tag qa_<SIZE> (see env_qa.sh).
set -e
cd "$(dirname "$0")/../../.."
mkdir -p logs
export SIZE="${SIZE:-135M}"
QUEUE="${QUEUE:-gpua100}"
PARTS="${PARTS:-4}"

. scripts/hpc/submit_lib.sh
JOB_ENV="SIZE"  # written into the job scripts (bsub does not pass it on)

# -q on the command line overrides #BSUB -q
if [ "${SKIP_FINETUNE:-0}" = "1" ]; then
    [ -d "checkpoints/smollm2_qa_$SIZE" ] || { echo "no checkpoints/smollm2_qa_$SIZE"; exit 1; }
    dep=""
    echo "skipping fine-tune"
else
    j1=$(submit scripts/hpc/qa/01_finetune.lsf)
    dep="-w done($j1)"
    echo "finetune: $j1"
fi
# parts of an earlier run would be merged too
rm -f "samples/parts_qa_$SIZE"/part*.pt
j2=$(submit scripts/hpc/qa/02_sample_part.lsf $dep -J "pb_qa_part[1-$PARTS]")
# ended(), not done(): merge whatever parts finished
j3=$(submit scripts/hpc/qa/03_merge_eval.lsf -w "ended($j2)")
echo "parts: $j2 (array of $PARTS)   merge + evaluate: $j3   (SIZE=$SIZE, queue $QUEUE)"
