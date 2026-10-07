# Sourced by every job in scripts/hpc/. Jobs run from the directory bsub was
# called in, which submit_pipeline.sh makes the repository root.
#
# Queue names change; check `bqueues` / `nodestat` and the DTU HPC GPU docs.
# gpuv100 and gpua100 are the usual GPU queues.

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
# GPU wheels come from the cu128 index, which needs a >=12.8 driver; if
# `nvidia-smi` above shows an older one, point tool.uv.index at cu126.
if [ "${UV_SYNC:-0}" = "1" ]; then
    .venv/bin/python -c "import encodings" 2>/dev/null || rm -rf .venv
    uv sync --extra gpu
fi
export UV_NO_SYNC=1

uv run python -c "import sys, torch; print(sys.version.split()[0], 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
