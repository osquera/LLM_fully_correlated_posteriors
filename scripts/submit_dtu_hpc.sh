#!/bin/sh
# DTU HPC (LSF) job script.
#
# Submit with:   bsub < scripts/submit_dtu_hpc.sh
# Watch with:    bstat        (or: bpeek <jobid>)
# Kill with:     bkill <jobid>
#
# Queue names change; check `bqueues` / `nodestat` and the DTU HPC GPU docs
# before trusting the -q line below. gpuv100 and gpua100 are the usual GPU
# queues; gpua10 / gpua40 also exist on some setups.

#BSUB -J projbayes
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 6:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -e
mkdir -p logs figures samples checkpoints

module load python3/3.11.4
module load cuda/12.3.2
module load cudnn/v8.9.1.23-prod-cuda-12.X

nvidia-smi

# uv is not a module; install once into $HOME and it persists across jobs.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# GPU wheels. Match the index to the loaded CUDA: cu128 needs a >=12.8 driver,
# so with cuda/12.3 loaded you may need to point tool.uv.index at cu126 instead.
uv sync --extra gpu

uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- stage 1: theta_map ----
uv run python scripts/finetune_lora.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --batch-size 16 \
    --epochs 3

# ---- stage 2: posterior samples ----
# Sequence mode: dense J^L stays small enough for the exact SVD route, and it
# keeps rank low -- which keeps ||delta|| down (README: alpha scaling trap).
#
# This run still takes alpha from Lemma 3.4, which at this scale is estimated
# rather than known: with P = 460,800 and rank = N = 256, the 8-probe std on
# kernel_dim is ~339, so the implied rank is mostly noise. sample_posterior.py
# prints both numbers -- if they are comparable, do not trust the closed form.
# Sweep ||delta|| against the worst per-datum loss change and re-run with an
# explicit --alpha (README: "The estimator degenerates before the formula does").
uv run python scripts/sample_posterior.py \
    --loss-mode sequence \
    --n-train 256 \
    --proj-batch-size 8 \
    --n-samples 16 \
    --n-iterations 500 \
    --tol 1e-3

# ---- stage 3: evaluation, baselines, figures ----
uv run python scripts/eval_underfitting.py
uv run python scripts/compare_methods.py --adapter checkpoints/smollm2_lora
# Same loss mode as stage 2, or the figure shows a posterior for a J^L that was
# never sampled. This script takes rank from an exact SVD, not from probes.
uv run python scripts/visualize_posterior.py --adapter checkpoints/smollm2_lora \
    --loss-mode sequence --out figures/posterior.pdf

echo "done"
