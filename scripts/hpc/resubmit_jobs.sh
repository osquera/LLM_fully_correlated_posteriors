#!/bin/sh
# Resubmit the two jobs that failed on 2026-10-31 (see EXPERIMENTS.md, E29/E31).
#
#   pb_popqa  32_popqa_errors.lsf array 1-3
#     element 1  died in error_breakdown.py on a null PopQA subject, AFTER
#                the posterior had been sampled: popqa_eval1.json and
#                popqa_eval1_map.json are complete, only the breakdown is
#                missing. Resubmitted alone with all expensive steps skipped.
#     element 2  completed; not resubmitted.
#     element 3  never ran (OSError 70 reading /work3); resubmitted whole.
#
#   pb_full   34_bio_full.lsf, array 1-3
#     never produced results. Resubmitted whole: an array again, since a
#     sequential job would need more than gpuh100's 24 h wall clock. Each
#     element holds ~6.6 GB (members deleted as their deltas are folded
#     into the baselines file; samples deleted after evaluation), and every
#     step is skipped when its output exists, so resubmit to resume.
#
# IMPORTANT: element 1 keeps its existing data/popqa1/corpus.json. The
# null-subject filter added to make_popqa_corpus.py draws from a different
# row list, so regenerating it would make replication 1 come from a
# different question pool than replication 2 and break comparability.
#
#   sh scripts/hpc/resubmit_jobs.sh          # or: bsub < scripts/hpc/resubmit_jobs.sh
#
# The 32/34 scripts are read from the repo on the HPC, so `git pull` first.
set -e
cd "$(dirname "$0")/../.."

for s in 1 2 3; do
    [ -f "data/popqa$s/corpus.json" ] || echo "note: data/popqa$s/corpus.json absent, will be built"
done
# element 1 must not rebuild its corpus (it already exists); sample and
# fine-tune steps are skipped by their run.json / file guards.
[ -f data/popqa1/corpus.json ] || { echo "data/popqa1/corpus.json missing: element 1 cannot be repaired without the original corpus"; exit 1; }

# Keep element 1's complete eval: the script re-runs evaluate_bio.py
# unconditionally, which would overwrite popqa_eval1.json. Move the good copy
# aside and restore it if the rerun leaves the file changed. error_breakdown.py
# reads --eval, so it needs the file present.
keep=figures/popqa_eval1.json.keep
if [ -f figures/popqa_eval1.json ] && [ ! -f "$keep" ]; then
    cp figures/popqa_eval1.json "$keep"
    echo "kept a copy of the complete eval at $keep"
fi

bsub -J "pb_popqa_fix[1,3]" < scripts/hpc/32_popqa_errors.lsf
bsub < scripts/hpc/34_bio_full.lsf
echo "submitted pb_popqa_fix[1,3] and pb_full; check with bstat / bjobs -a"
echo "after pb_popqa_fix[1] finishes, copy figures/error_breakdown_popqa1.json"
echo "and check the kept eval $keep before trusting a regenerated one."
