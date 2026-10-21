# Sourced by the submit_*.sh scripts.
#
# DTU's LSF does not hand the submitting shell's environment to a job: a
# variable exported before `bsub` (LOSS_ON, SIZE, ...) is unset inside the job,
# which then silently falls back to its default. So the variables named in
# JOB_ENV are written into the job script itself, as `export` lines right after
# its #BSUB block, and the script is piped to bsub.
#
#   JOB_ENV="LOSS_ON SIZE"
#   id=$(submit scripts/hpc/xx.lsf -w "done(123)")    # bsub options after the file

submit() {
    f=$1
    shift
    exports=""
    for v in $JOB_ENV; do
        eval "val=\${$v-}"
        exports="${exports}export $v='$val'; "
    done
    # the first line that is not a comment ends the #BSUB block
    awk -v env="$exports" '!done && !/^#/ { print env; done = 1 } { print }' "$f" \
        | bsub "$@" | sed -n 's/^Job <\([0-9]*\)>.*/\1/p'
}
