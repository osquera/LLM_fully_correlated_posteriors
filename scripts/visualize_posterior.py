"""Visualise the projected posterior and its baselines.

The direct analogue of `plt.imshow(sigma @ sigma.T)` from the reference repo's
VI notebooks, but computed *exactly* rather than from samples: with the dense
J^L in hand, the covariance restricted to a coordinate slice is available in
closed form for all three methods, with no Monte-Carlo noise.

  projected     alpha^-1 (I - W^T W)         W = right singular vectors of J^L
  lla           (J^T J + alpha I)^-1
  diagonal      (diag(J^T J) + alpha I)^-1

Panels:
  1-3  correlation heatmaps -- the "fully correlated" claim, made visible
   4   posterior std per LoRA module: where the freedom actually lives
   5   covariance spectra
   6   per-datum training-loss change: the underfitting claim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from llmproj.baselines import build_loss_jacobian
from llmproj.losses import make_causal_lm_loss
from llmproj.paramspace import ParamSpace


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="LoRA checkpoint; omit for a tiny random model")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--proj-batch-size", type=int, default=4)
    ap.add_argument("--loss-mode", choices=["sequence", "token"], default="sequence")
    ap.add_argument("--slice-size", type=int, default=96,
                    help="number of coordinates shown in the heatmaps")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--out", default="figures/posterior.pdf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_model(args):
    if args.adapter is None:
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=4,
                          max_position_embeddings=args.seq_len,
                          attn_implementation="eager")
        m = get_peft_model(
            LlamaForCausalLM(cfg).float(),
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                       lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"),
        )
        return m.to(args.device).eval(), 256

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    return PeftModel.from_pretrained(base, args.adapter).to(args.device).eval(), tok.vocab_size


def to_corr(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    return cov / np.outer(d, d)


def main():
    args = parse_args()
    torch.manual_seed(0)
    model, vocab = load_model(args)

    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"])
    theta, frozen = space.theta(model), space.frozen(model)
    f = make_causal_lm_loss(model, space, frozen, mode=args.loss_mode)

    ids = torch.randint(0, vocab, (args.n_seq, args.seq_len), device=args.device)
    S = args.proj_batch_size
    batches = [
        {"input_ids": ids[i:i + S],
         "attention_mask": torch.ones(min(S, args.n_seq - i), args.seq_len,
                                      dtype=torch.long, device=args.device)}
        for i in range(0, args.n_seq, S)
    ]

    J = build_loss_jacobian(f, theta, space, batches,
                            max_gb=args.max_gb, progress=True).float().cpu()
    _, s, Vh = torch.linalg.svd(J, full_matrices=False)
    keep = s > 1e-6 * s.max()
    s, W = s[keep], Vh[keep]
    rank = int(keep.sum())

    alpha = args.alpha
    if alpha is None:
        alpha = float(sum((t.float() ** 2).sum() for t in theta.values())) / max(rank, 1)
    print(f"P={space.P}  rows={J.shape[0]}  rank(J^L)={rank}  "
          f"kernel dim={space.P - rank}  alpha={alpha:.4e}")

    # ---- exact covariance on a coordinate slice, per method ----
    k = min(args.slice_size, space.P)
    idx = torch.linspace(0, space.P - 1, k).long()
    Ws = W[:, idx]
    I_k = torch.eye(k)
    diag_full = (J ** 2).sum(0)

    cov_proj = ((I_k - Ws.T @ Ws) / alpha).numpy()
    lla_scale = (s ** 2 + alpha).reciprocal() - 1.0 / alpha
    cov_lla = (I_k / alpha + Ws.T @ torch.diag(lla_scale) @ Ws).numpy()
    cov_diag = np.diag((1.0 / (diag_full[idx] + alpha)).numpy())

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    panels = [
        ("projected   alpha^-1 (I - P)", cov_proj),
        ("linearised Laplace", cov_lla),
        ("diagonal Laplace", cov_diag),
    ]
    for col, (name, cov) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        c = to_corr(cov)
        off = c - np.diag(np.diag(c))
        v = max(np.abs(off).max(), 1e-12)
        im = ax.imshow(c, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(f"{name}\ncorrelation, max |off-diag| = {np.abs(off).max():.3f}",
                     fontsize=10)
        ax.set_xlabel("parameter index (slice)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ---- per-module posterior std ----
    ax = fig.add_subplot(gs[1, 0])
    var_proj = (1.0 - (W ** 2).sum(0)) / alpha
    i, labels, vals = 0, [], []
    for n, ne in zip(space.names, space.numels):
        labels.append(n.replace("base_model.model.model.", "").replace(".default.weight", ""))
        vals.append(float(var_proj[i:i + ne].clamp(min=0).mean().sqrt()))
        i += ne
    ax.barh(range(len(vals)), vals, color="slategray")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("mean posterior std (projected)")
    ax.set_title("where the posterior freedom lives", fontsize=10)

    # ---- covariance spectra ----
    ax = fig.add_subplot(gs[1, 1])
    ev_proj = np.concatenate([np.zeros(rank), np.full(space.P - rank, 1.0 / alpha)])
    ev_lla = np.concatenate([(1.0 / (s ** 2 + alpha)).numpy(),
                             np.full(space.P - rank, 1.0 / alpha)])
    ev_diag = (1.0 / (diag_full + alpha)).numpy()
    for lbl, ev, style in [("projected", ev_proj, "-"), ("lla", ev_lla, "--"),
                           ("diagonal", ev_diag, ":")]:
        ax.semilogy(np.sort(ev)[::-1] + 1e-30, style, label=lbl)
    ax.set_xlabel("index")
    ax.set_ylabel("covariance eigenvalue")
    ax.set_title("covariance spectra (projected is exactly 0 or 1/alpha)", fontsize=10)
    ax.legend(fontsize=8)

    # ---- loss change vs perturbation size: the underfitting claim ----
    # Drawn at MATCHED ||delta||, not at each method's own alpha. Comparing at
    # alpha is uninformative whenever alpha* is small: every method then leaves
    # the linear regime and Lemma 4.3's O(||delta||^2) bound says nothing.
    ax = fig.add_subplot(gs[1, 2])
    gen = torch.Generator().manual_seed(7)
    base_loss = torch.cat([f(theta, b) for b in batches]).detach().cpu()

    def direction(kind):
        eps = torch.randn(space.P, generator=gen)
        if kind == "projected":
            d = eps - W.T @ (W @ eps)
        elif kind == "lla":
            d = (alpha ** -0.5 * eps
                 + W.T @ (((s ** 2 + alpha).rsqrt() - alpha ** -0.5) * (W @ eps)))
        else:
            d = eps * (1.0 / (diag_full + alpha)).sqrt()
        return d / d.norm()

    norms = np.logspace(-4, 0, 9)
    for kind, style in [("projected", "o-"), ("lla", "s--"), ("diagonal", "^:")]:
        u = direction(kind)
        curve = []
        for nrm in norms:
            d = (u * float(nrm)).to(space.dtype).to(space.device)
            pert = {kk: theta[kk] + dd for kk, dd in space.unflatten(d).items()}
            cur = torch.cat([f(pert, b) for b in batches]).detach().cpu()
            curve.append(max(float((cur - base_loss).abs().max()), 1e-12))
        ax.loglog(norms, curve, style, label=kind, markersize=4)
    ax.loglog(norms, norms ** 2, "k-", alpha=0.3, lw=1, label="$O(\\|\\delta\\|^2)$")
    ax.loglog(norms, norms, "k--", alpha=0.3, lw=1, label="$O(\\|\\delta\\|)$")
    ax.set_xlabel("$\\|\\delta\\|$")
    ax.set_ylabel("max |loss change| over training data")
    ax.set_title("Lemma 4.3: projected should track $O(\\|\\delta\\|^2)$", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle(f"Projected posterior over LoRA weights    P={space.P:,}    "
                 f"rank(J^L)={rank}    kernel dim={space.P - rank:,}    "
                 f"alpha={alpha:.3g}", fontsize=12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
