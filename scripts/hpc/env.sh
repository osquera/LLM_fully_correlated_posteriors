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
