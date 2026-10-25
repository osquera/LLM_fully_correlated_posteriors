# Sourced by every job in scripts/hpc/. Jobs run from the directory bsub was
# called in, which submit_pipeline.sh makes the repository root.
#
# Queue: gpua100. Not gpuv100: the locked torch (cu128 wheels) has no kernels
# for Volta (compute capability 7.0), only >= 7.5. Running on a V100 would need
# the cu126 index in pyproject.toml and a check of torch.cuda.get_arch_list().

set -e
mkdir -p logs figures samples checkpoints

# No `module load python3/cuda/cudnn`: uv provides its own Python, and the
# PyTorch wheels bundle CUDA and cuDNN, needing only the driver. The python3
# module in particular sets variables that break a uv venv at startup
# ("Fatal Python error: Failed to import encodings module").
unset PYTHONHOME PYTHONPATH

nvidia-smi

# uv is not a module; install once into $HOME and it persists across jobs.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# A uv-managed interpreter (downloaded once), never the cluster's.
export UV_PYTHON=3.12
export UV_PYTHON_PREFERENCE=only-managed

# Only the first job syncs the environment: the two sampling jobs run in
# parallel and must not rebuild the same .venv concurrently. A venv whose
# interpreter cannot start (e.g. left over from a failed run) is rebuilt.
# GPU wheels come from the cu128 index, which needs a >=12.8 driver (see
# `nvidia-smi` above).
if [ "${UV_SYNC:-0}" = "1" ]; then
    venv="${UV_PROJECT_ENVIRONMENT:-.venv}"
    # A partially extracted interpreter (stdlib missing, sys.prefix stuck at
    # the build placeholder '/install') fails the same way: reinstall it.
    py=$(uv python find --system "$UV_PYTHON" 2>/dev/null || true)
    if [ -z "$py" ] || ! "$py" -c "import encodings, os" 2>/dev/null; then
        uv python install --reinstall "$UV_PYTHON"
        rm -rf "$venv"
    fi
    "$venv/bin/python" -c "import encodings" 2>/dev/null || rm -rf "$venv"
    uv sync --extra gpu
fi
export UV_NO_SYNC=1

# Fail fast, before hours in the queue are wasted, if this torch build has no
# kernels for the GPU we were given.
uv run python -c "
import sys, torch
cc = 'sm_%d%d' % torch.cuda.get_device_capability(0)
print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0), cc,
      'arch list:', torch.cuda.get_arch_list())
torch.ones(1, device='cuda').add_(1)
"

# Which tokens carry loss (llmproj.data): LOSS_ON=response fine-tunes and
# projects on the response only. Its outputs get a "_response" suffix, so the
# default run's checkpoint, samples and figures are left alone.
export LOSS_ON="${LOSS_ON:-all}"
case "$LOSS_ON" in
    all) SUF="" ;;
    response) SUF="_response" ;;
    *) echo "LOSS_ON must be all or response, got '$LOSS_ON'"; exit 1 ;;
esac
# Token-mode projection settings (03_sample_token.lsf). Non-default ones tag
# the token samples and the figures, e.g. N_ITER=1500 writes
# posterior_token_it1500.pt and posterior_eval_it1500.json, so runs with other
# settings never overwrite the default run. VEC_BATCH / ROW_BATCH only trade
# speed for memory (same result), so they are not part of the tag.
export N_ITER="${N_ITER:-500}"
export N_SAMPLES="${N_SAMPLES:-8}"
TOK_TAG=""
[ "$N_ITER" != 500 ] && TOK_TAG="${TOK_TAG}_it$N_ITER"
[ "$N_SAMPLES" != 8 ] && TOK_TAG="${TOK_TAG}_s$N_SAMPLES"
export ADAPTER="checkpoints/smollm2_lora$SUF"
export SAMPLES_SEQ="samples/posterior_sequence$SUF.pt"
export SAMPLES_TOK="samples/posterior_token$SUF$TOK_TAG.pt"
export EVAL_OUT="figures/posterior_eval$SUF$TOK_TAG.json"
export SWEEP_OUT="figures/tolerance_sweep$SUF$TOK_TAG"
export SUF TOK_TAG
echo "LOSS_ON=$LOSS_ON  adapter=$ADAPTER  token: $N_SAMPLES samples x $N_ITER sweeps -> $SAMPLES_TOK"
