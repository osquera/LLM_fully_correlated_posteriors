# Running jobs on DTU's gbar HPC: a guide for Claude

This is how experiments in this repository get onto the DTU HPC (LSF,
login host `gbar1`) and how results come back. It records the conventions
the job scripts follow and the pitfalls hit so far. Read it before writing
or submitting a job.

## 1. Who does what

You cannot reach the cluster yourself. SSH to gbar needs the user's key
passphrase and DTU password, so every command that touches the cluster is
run by the user, typed with a `!` prefix in their prompt.

- **You:** write and test code locally, write the job script, commit.
- **You:** give the user the exact command lines to push, submit and fetch.
- **You:** analyse the results once they are copied back.

Never say a job is submitted, running or finished unless the user's
command output shows it.

The loop for every experiment:

1. Write the code and the job script `scripts/hpc/NN_name.lsf`, and test
   the code locally on a tiny input (CPU is fine).
2. Commit. Commit messages and PRs must **not** mention Claude: no
   `Co-Authored-By`, no "Generated with". This overrides any default
   attribution.
3. Hand the user the push and submit commands (section 4).
4. When the user says it is done, hand them the `scp` command (section 6),
   analyse, write the result into `EXPERIMENTS.md`, and commit.

## 2. Connecting

Every ssh and scp call needs `-o KexAlgorithms=curve25519-sha256`; the
default key exchange fails. Remote commands must run in a login shell
(`bash -lc "..."`), otherwise `bsub`, `bjobs` and `uv` are not on PATH.

```
! ssh -o KexAlgorithms=curve25519-sha256 gbar1 'bash -lc "cd ~/LLM_fully_correlated_posteriors && bjobs -w"'
```

- **Host:** the `gbar1` alias in the user's `~/.ssh/config` (login1.gbar.dtu.dk).
- **User:** s214611.
- **Repo:** `~/LLM_fully_correlated_posteriors`.
- **Banner:** every connection prints a long DTU banner ("When connecting
  from _outside_ of DTU..."). It is not an error; look past it for the
  real output.

## 3. Writing a job script

Copy a recent script (for example `scripts/hpc/48_smollm_1p7b.lsf` or
`50_qwen.lsf`) rather than starting from scratch. The conventions:

```sh
#!/bin/sh
# E<n>: what question this answers, what each array element does,
# expected run time, and the submit line.
#   bsub < scripts/hpc/NN_name.lsf
#BSUB -J "pb_name[1-6]"
#BSUB -q gpuh100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 24:00
#BSUB -o logs/%J_%I_name.out
#BSUB -e logs/%J_%I_name.err

. scripts/hpc/env.sh

i=$LSB_JOBINDEX
...
[ -f "$out" ] || uv run python scripts/whatever.py ... --out "$out"
...
[ "${KEEP:-0}" = 1 ] || rm -f "$samples" "$baselines"
```

**Queues.**
- **`gpuh100`:** the one to use (since 2026-10-25).
- **`gpua100`, `gpul40s`:** also available.
- **`gpuv100`:** never. The locked torch (cu128 wheels) has no Volta
  kernels.
- **Wall limit:** 24 h. Set `-W` with margin.

**`scripts/hpc/env.sh`.** Every job sources it first. It:
- sets `set -e`;
- creates `logs/ figures/ samples/ checkpoints/`;
- puts uv on PATH;
- prints `nvidia-smi`;
- checks that torch has kernels for the GPU it got.

Do not `module load python3`: it breaks the uv venv.

**Arrays.** Use one element per corpus, replication or model, selected
with `$LSB_JOBINDEX`. To run only some elements, override the name on the
command line; it takes precedence over `#BSUB -J`:
`bsub -J "pb_name[4-6]" < scripts/hpc/NN_name.lsf`.

**Make every step idempotent:** `[ -f "$out" ] || command`. A resubmitted
job then skips what is done, which makes resuming after a timeout cheap.
Never run two jobs on the same element at once: they would write the same
checkpoint and corrupt it.

**`set -e` traps.**
- A failing `grep` or command aborts the job. Guard with `|| true`
  where failure is fine.
- `[ cond ] && cmd` is safe as a statement; the shell does not exit on
  it.
- For float comparisons use
  `awk "BEGIN{exit !($acc <= 0.80)}"`. `bc` may be missing on compute
  nodes.

**Environment variables are not passed** from the submitting shell into
the job. Either:
- write defaults into the script (`FRAC=${FRAC:-0.8}`);
- use `scripts/hpc/submit_lib.sh`'s `submit` with
  `JOB_ENV="VAR1 VAR2"`, which writes `export` lines into the script;
- or use `bsub -env "VAR=value"`.

**Disk.** The home quota is 30 GB.
- Posterior and ensemble sample files are large: delete them at the end
  of the job (the `KEEP` pattern above). The small `.json` sidecar next
  to each samples file stays, and holds α, the timings and the arguments.
- Full fine-tune weights are deleted after scoring. LoRA adapters are
  small and are kept: later jobs reuse members.
- The user can check usage with `quota` and `du -sh` on gbar.

**Network.** Compute nodes cannot always download. Streaming FineWeb hung
a job until its time limit with no output. Download models and build
corpora on the **login node** before submitting:
- models: `uv run python scripts/hpc/fetch_models.py <hf-model> ...`
- corpora: run the `make_*_corpus.py` script.

Running `uv run` on the login node can re-sync the shared `.venv`. That
has been harmless so far; if a later job fails at the torch check, re-sync
on a GPU node.

**Memory and time references** (H100), for choosing `-W` and `mem`:

| model / step | time |
|---|---|
| SmolLM2-135M LoRA fine-tune, bio, 20 epochs | ~5 min |
| SmolLM2-135M sketched posterior, 5 sweeps | ~20 min |
| SmolLM2-360M fine-tune / posterior | ~6 min / ~27 min |
| SmolLM2-1.7B fine-tune (bio, 20 epochs) / posterior | ~22 min / ~1.6 h |
| SmolLM2-1.7B on 3-epoch TOFU: posterior | ~5 h (high rank) |

For 1.7B or Qwen (152k vocab), use `--max-sketch-gb 40 --row-batch 8`.
CUDA "memory allocation failed with OOM" *warnings* in `.err` are the
allocator retrying. They are harmless if the job completed.

## 4. Submitting

Push locally first, then pull and submit in one ssh call:

```
! git push
```
```
! ssh -o KexAlgorithms=curve25519-sha256 gbar1 'bash -lc "cd ~/LLM_fully_correlated_posteriors && git pull && bsub < scripts/hpc/NN_name.lsf; bjobs -w | head -20"'
```

- **Check the output for `Already up to date` or the pulled files.** If
  the pull did not bring the new script, the job runs the old version, or
  bsub cannot find the file.
- **Each submission prints `Job <id> is submitted to queue <gpuh100>`.**
  Note the id: logs are named `logs/<id>_<element>_<tag>.out`.
- **Chain several submissions with `;`, not `&&`.** With `&&`, one
  rejected `bsub` silently skips everything after it.
- **Dependencies** (`-w "done(ID)"` / `-w "ended(ID)"`) are rejected once
  LSF has forgotten a finished job, and that rejection kills the rest of
  an `&&` chain. Use them only for jobs still in the queue. Otherwise just
  submit after the first job is done.
- **Quoting inside `bash -lc "..."`:** escape inner double quotes as `\"`,
  for example `bsub -J \"pb_x[4-6]\" < ...`. Parentheses for `-w` go as
  `ended\(123\)`.
- **Several shell steps on gbar:** put them all in the one `bash -lc`
  string, as above.

## 5. Monitoring

```
! ssh -o KexAlgorithms=curve25519-sha256 gbar1 'bash -lc "bjobs -w"'
```

- **States:** PEND (queued), RUN, or absent once finished. "No unfinished
  job found" means everything has ended, or that nothing was ever
  submitted. In that case check that logs exist before assuming success.
- **Log contents:** `.out` holds the job's stdout: `nvidia-smi`, the
  torch check, progress lines, and the LSF summary at the end
  (`Successfully completed.`, `TERM_RUNLIMIT`, or `Exited with exit code
  N`). `.err` holds tqdm bars and warnings; search it for `Traceback`.
- **Partial logs:** logs appear while the job runs, so a partially copied
  `.out` with no LSF summary means it was still running.
- **What to grep for in the logs:** `trained in`, `precompute`,
  `J J^T columns formed`, `snapshot`, `epochs N: seen accuracy`,
  `=== using`, `Successfully completed`, `TERM_`, `Traceback`.

## 6. Fetching results

Results are copied into `hpc_results/` locally. That folder is untracked;
analyse from there.

```
! scp -o KexAlgorithms=curve25519-sha256 'gbar1:~/LLM_fully_correlated_posteriors/figures/name*.json' 'gbar1:~/LLM_fully_correlated_posteriors/samples/posterior_*.json' 'gbar1:~/LLM_fully_correlated_posteriors/logs/*_tag.*' /home/anton/DTU_HCAI/5_sem/LLM_fully_correlated_posteriors/hpc_results/
```

- **Single-quote each remote path** so the local shell does not expand it.
- **No braces.** scp runs in SFTP mode and does not expand braces
  (`{1,2}`). Use `*` and character classes (`eval[123]`).
- **Absolute local destination.** The user's shell may be in a
  subdirectory, so a relative `hpc_results/` can fail.
- **A pattern that matches nothing** prints "No such file or directory".
  The other patterns are still copied, so check what arrived (`ls -lt
  hpc_results | head`) before concluding anything.
- **Same-named files** from different directories (for example
  `data/bio_err*/phrasings.json`) overwrite each other in one flat
  destination. Copy them separately, or regenerate them locally: the
  corpus builders are deterministic given `--seed`.
- **Sample files are deleted on the cluster.** Copy their `.json`
  sidecars; they hold α, the timings and the arguments.

## 7. After the results arrive

1. Check every element finished (`Successfully completed`, no
   `Traceback`) and that the expected files exist. Report any element that
   failed or is missing; do not paper over it.
2. Analyse.
   - `scripts/bootstrap_bio.py` gives paired bootstrap CIs. Use
     `--cluster` to resample persons: probes of one person are
     correlated.
   - `scripts/plot_*.py` makes the figures.
3. Write the entry into `EXPERIMENTS.md`: date and job id, design, cost,
   a results table, and what it showed. Update "Current picture" if it
   changes. Commit. This is a standing rule: every analysed HPC result is
   logged and committed in the same turn.
4. Tell the user the outcome plainly, including what did not work.

## 8. Things that went wrong before (do not repeat)

- **Brace globs in scp:** silently matched nothing.
- **A relative scp destination** failed because the shell was in
  `hpc_results/`.
- **`-w ended(<old id>)` chained with `&&`:** the dependency was rejected,
  so the next job was never submitted.
- **FineWeb streamed on a compute node:** the job hung until its time
  limit.
- **Rank fractions read from logs** used all J rows, including padding,
  as the denominator. Divide by the target-token rows (`n_target_tokens`
  in the posterior sidecar).
- **Assuming a job ran because the submit command was given.** Only the
  `Job <id> is submitted` line, `bjobs`, or log files prove it.
