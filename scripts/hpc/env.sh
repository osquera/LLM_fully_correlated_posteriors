# Sourced by every job in scripts/hpc/. Jobs run from the directory bsub was
# called in, which submit_pipeline.sh makes the repository root.
#
# Queue names change; check `bqueues` / `nodestat` and the DTU HPC GPU docs.
# gpuv100 and gpua100 are the usual GPU queues.

set -e
mkdir -p logs figures samples checkpoints

module load python3/3.11.4
module load cuda/12.3.2
module load cudnn/v8.9.1.23-prod-cuda-12.X

nvidia-smi

# uv is not a module; install once into $HOME and it persists across jobs.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Only the first job syncs the environment: the two sampling jobs run in
# parallel and must not rebuild the same .venv concurrently. GPU wheels: match
# the index to the loaded CUDA (cu128 needs a >=12.8 driver; with cuda/12.3 you
# may need to point tool.uv.index at cu126 instead).
if [ "${UV_SYNC:-0}" = "1" ]; then
    uv sync --extra gpu
fi
export UV_NO_SYNC=1

uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
