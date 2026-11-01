#!/bin/sh
# Which E29 / E31 elements are actually done, and is anything duplicated?
#
# Run on gbar1, in the repo:
#   sh scripts/hpc/check_jobs.sh
#
# Prints three things:
#   1. the queue, with duplicate JOB IDs for the same element flagged;
#   2. per element, which output files exist (each step is skipped when its
#      output exists, so this is what a resubmission would redo);
#   3. a verdict per element: done / resumable / nothing yet.
#
# The steps are ordered as the job scripts run them. "redo" means the step
# would run again on resubmission; that is only harmless while no job is
# running on the same element (two jobs writing one checkpoint corrupt it).
set -u
cd "$(dirname "$0")/../.." || exit 1

echo "================= queue ================="
# Match the job NAME wherever it sits: bjobs -w puts JOB_NAME last, bstat
# and other formats differ, so no fixed column is safe.
mine='qa_fix|popqa|pb_full'
if command -v bjobs >/dev/null 2>&1; then
    hdr=$(bjobs -w 2>/dev/null | head -1)
    rows=$(bjobs -w 2>/dev/null | tail -n +2 | grep -E "$mine")
    printf '%s\n' "$hdr"
    if [ -n "$rows" ]; then printf '%s\n' "$rows"; else echo "  (none)"; fi
    echo
    # one "element<TAB>jobid" per live job; >1 distinct job id per element is a duplicate
    dup=$(printf '%s\n' "$rows" | awk -v OFS='\t' '{
              id=$1
              name=$0
              if (!match(name, /(qa_fix|popqa|pb_full)\[[0-9,]+\]/)) next
              nm=substr(name, RSTART, RLENGTH)
              fam=nm; sub(/\[.*/, "", fam)     # family: popqa fix vs full keep separate element spaces
              sub(/.*\[/, "", nm); sub(/\]/, "", nm)
              n=split(nm, e, ",")
              for (i=1; i<=n; i++) print fam " " e[i], id }' \
          | sort -u | awk -F'\t' '{c[$1]++; j[$1]=j[$1] " " $2}
              END {for (k in c) if (c[k] > 1) print k, j[k]}' | sort)
    if [ -n "$dup" ]; then
        echo "!! DUPLICATES -- more than one live job on the same element:"
        printf '%s\n' "$dup" | while read -r el jobs; do
            echo "   $el:$jobs"
        done
        echo "   keep one, bkill the rest: they write the same checkpoint."
    else
        echo "no duplicate elements queued"
    fi
else
    echo "  (bjobs not found: not on the cluster; showing files only)"
fi

# step <label> <path...>: one line per job step, listing what exists
step() {
    label=$1
    shift
    have=0
    total=0
    for p in "$@"; do
        total=$((total + 1))
        [ -e "$p" ] && have=$((have + 1))
    done
    if [ "$have" -eq "$total" ]; then
        printf '  %-22s ok\n' "$label"
    elif [ "$have" -eq 0 ]; then
        printf '  %-22s --\n' "$label"
    else
        printf '  %-22s %s/%s\n' "$label" "$have" "$total"
    fi
}

report_e29() {  # s
    s=$1
    echo
    echo "E29 PopQA element $s"
    step "corpus"        "data/popqa$s/corpus.json"
    step "theta_map"     "checkpoints/popqa${s}_lora/run.json"
    step "map eval"      "figures/popqa_eval${s}_map.json"
    step "posterior"     "samples/posterior_popqa$s.pt"
    step "baselines"     "samples/baselines_popqa$s.pt"
    step "eval"          "figures/popqa_eval$s.json"
    step "breakdown"     "figures/error_breakdown_popqa$s.json"
}

report_e31() {  # s
    s=$1
    echo
    echo "E31 full-model element $s"
    step "corpus"        "data/bio_err$s/corpus.json"
    step "theta_map"     "checkpoints/bio_full$s/run.json"
    step "map eval"      "figures/bio_full_eval${s}_map.json"
    step "posterior iso" "samples/posterior_bio_full${s}_iso.pt"
    step "posterior rms" "samples/posterior_bio_full${s}_rms.pt"
    step "ensemble (8)"  "samples/baselines_bio_full$s.pt"
    step "eval"          "figures/bio_full_eval$s.json"
    step "breakdown"     "figures/error_breakdown_bio_full$s.json"
}

verdict() {  # <final-output> <first-output>
    if [ -e "$1" ]; then
        echo "  => done"
    elif [ -e "$2" ]; then
        echo "  => resumable (some work exists; a resubmission continues it)"
    else
        echo "  => not started"
    fi
}

echo
echo "================= E29 (job 32) ================="
for s in 1 2 3; do
    report_e29 "$s"
    verdict "figures/error_breakdown_popqa$s.json" "data/popqa$s/corpus.json"
done

echo
echo "================= E31 (job 34) ================="
for s in 1 2 3; do
    report_e31 "$s"
    verdict "figures/error_breakdown_bio_full$s.json" "data/bio_err$s/corpus.json"
done

echo
echo "================= disk ================="
for d in checkpoints samples data; do
    [ -d "$d" ] && printf '  %-12s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
printf '  %-12s %s\n' "total" "$(du -sh . 2>/dev/null | cut -f1)"
quota=$(df -h . 2>/dev/null | awk 'NR==2 {print $3 " used of " $2}')
[ -n "$quota" ] && printf '  %-12s %s\n' "home" "$quota"
echo
echo "reminder: two jobs on one element corrupt its checkpoint --"
echo "bkill the duplicate before resubmitting anything."
