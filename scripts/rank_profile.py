"""Per-batch rank of J_b and the sketch's fidelity on a corpus (plan item 9).

The sketched precompute pays off because a batch's token-loss Jacobian J_b
(rows = target tokens) has a numerical rank r far below its row count n:
tokens the model already fits have ~zero loss gradients. This measures r / n
where that may not hold (a pretrained model on generic text, Alpaca, ...),
on --n-batches batches spread evenly over the constraint set, and checks the
sketch against the exact factor on the same batches (docs/sketch_proof.md):

  spectrum    eigenvalues of J_b J_b^T (exact): rank at rtol 1e-4 / 1e-6 /
              1e-8, and lambda_{r+1} / lambda_1 at the k the sketch stops at
  sketch      the adaptive sketch's size k and kept rank
  fidelity    for --n-test Gaussian v: cosine between the exact and the
              sketched one-batch projections of v; the constraint residual
              ||J_b P v|| / ||J_b v|| of each; and how much of the exact
              projection the sketch removes, ||P_S P_E v - P_E v|| / ||P_E v||
              (the proof: ~0, the sketch only ever under-projects)

  uv run python scripts/rank_profile.py --adapter checkpoints/bio_err1_lora \\
      --dataset data/bio_err1/corpus.json --n-train 100000 --max-length 96 \\
      --out figures/rank_profile_bio1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llmproj.data import encode_texts, load_texts
from llmproj.losses import make_causal_lm_loss
from llmproj.models import load_finetuned
from llmproj.paramspace import ParamSpace
from llmproj.projection import _batch_proj, _jjt, _jvps, precompute_batch_pinv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--include", nargs="+", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--loss-on", choices=["all", "response"], default=None)
    ap.add_argument("--proj-batch-size", type=int, default=16)
    ap.add_argument("--n-batches", type=int, default=16, help="batches checked, spread evenly")
    ap.add_argument("--sketch", type=int, default=192)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    run = Path(args.adapter) / "run.json"
    loss_on = args.loss_on or (json.loads(run.read_text()).get("loss_on", "all") if run.exists() else "all")
    tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, include = load_finetuned(args.base_model, args.adapter, args.device)
    space = ParamSpace.from_module(model, include=args.include or include, trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    f = make_causal_lm_loss(model, space, frozen, mode="token")

    texts = load_texts(args.dataset, args.split, args.text_field)[: args.n_train]
    enc = encode_texts(tok, texts, args.max_length, loss_on=loss_on)
    S = args.proj_batch_size
    starts = list(range(0, enc["input_ids"].size(0), S))
    pick = sorted({starts[round(i * (len(starts) - 1) / max(1, args.n_batches - 1))]
                   for i in range(min(args.n_batches, len(starts)))})
    print(f"{space}; {len(starts)} batches of {S}, checking {len(pick)}")

    gen = torch.Generator().manual_seed(args.seed)
    rows_out = []
    for i in pick:
        L = int(enc["attention_mask"][i: i + S].sum(1).max())
        b = {k: v[i: i + S, :L].to(args.device) for k, v in enc.items()}
        b["row_mask"] = b["loss_mask"][:, 1:].reshape(-1).bool()
        f_b = lambda p, b=b: f(p, b)  # noqa: E731
        R = int(f_b(theta).numel())
        rows = b["row_mask"].nonzero().squeeze(1).cpu()
        n = len(rows)

        lam = torch.linalg.eigvalsh(_jjt(f_b, theta, space, R, rows).double()).flip(0).clamp_min(0)
        rk = {f"{t:g}": int((lam > t * lam[0]).sum()) for t in (1e-4, 1e-6, 1e-8)}
        (exact,) = precompute_batch_pinv(f, theta, space, [b], rtol=args.rtol, progress=False)
        (sk,) = precompute_batch_pinv(f, theta, space, [b], rtol=args.rtol, progress=False,
                                      sketch=args.sketch, seed=args.seed + i)
        k = sk.formed
        tail = float(lam[k] / lam[0]) if k < n else 0.0

        V = torch.randn(args.n_test, space.P, generator=gen).to(space.device, space.dtype)
        V = V / V.norm(dim=1, keepdim=True)
        PE = _batch_proj(V, f_b, theta, space, exact)
        PS = _batch_proj(V, f_b, theta, space, sk)
        JV, JPE, JPS = (_jvps(f_b, theta, space, X)[:, rows.to(space.device)].norm(dim=1)
                        for X in (V, PE, PS))
        cos = (PE * PS).sum(1) / (PE.norm(dim=1) * PS.norm(dim=1))
        removed = (_batch_proj(PE, f_b, theta, space, sk) - PE).norm(dim=1) / PE.norm(dim=1)
        r = {"start": i, "rows": n, "rank": rk, "rank_exact_factor": exact.rank,
             "sketch_k": k, "sketch_rank": sk.rank, "lam_k_over_lam_1": tail,
             "cos_exact_sketch": float(cos.mean()),
             "resid_exact": float((JPE / JV).mean()), "resid_sketch": float((JPS / JV).mean()),
             "sketch_removes_from_exact": float(removed.mean())}
        rows_out.append(r)
        print(f"  batch @{i:5d}: rows {n:5d}  rank 1e-4/1e-6/1e-8 {rk['0.0001']}/{rk['1e-06']}/"
              f"{rk['1e-08']} ({rk['1e-06'] / n:.0%})  sketch k {k} rank {sk.rank}  "
              f"lam_k/lam_1 {tail:.1e}  cos {r['cos_exact_sketch']:.4f}  resid "
              f"{r['resid_exact']:.1e}/{r['resid_sketch']:.1e}  removed {r['sketch_removes_from_exact']:.1e}")

    frac = [r["rank"]["1e-06"] / r["rows"] for r in rows_out]
    summary = {"rank_fraction_mean": sum(frac) / len(frac), "rank_fraction_min": min(frac),
               "rank_fraction_max": max(frac),
               "sketch_k_over_rows": sum(r["sketch_k"] / r["rows"] for r in rows_out) / len(rows_out),
               "cos_min": min(r["cos_exact_sketch"] for r in rows_out),
               "removed_max": max(r["sketch_removes_from_exact"] for r in rows_out)}
    print("summary", json.dumps(summary))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "loss_on": loss_on, "P": space.P, "batches": rows_out,
               "summary": summary}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
