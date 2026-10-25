"""Single-forward-pass OOD and error scores from the hidden states of theta_map,
on exactly the sets and tokens analyze_posterior.py scores, for a like-for-like
comparison with the posteriors' MI.

Per scored token (the position that predicts it), from the hidden state at
each of --layers:
  mahalanobis  (h - mu)^T Sigma^-1 (h - mu), mu / Sigma fitted on every
               scored token of the fine-tuning split (shrunk covariance)
  knn          1 - cosine similarity to the k-th nearest of a sample of those
               training tokens (deep k-NN)
and the MAP's own entropy and 1 - confidence for reference.

Reports, like analyze_posterior.py:
  ood    AUROC of the per-sequence mean score, test vs each OOD set
  error  AUROC for "the MAP's next-token prediction is wrong", pooled over
         test + near-OOD tokens: raw, and within deciles of MAP entropy

  uv run python scripts/baselines_single_pass.py --score-on response
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_posterior import NEAR_OOD, build_analysis_sets, seq_index, stratified_auroc
from evaluate_posterior import auroc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default="samples/posterior_token.pt",
                    help="samples file whose training data defines the fit set")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--test-split", default="train[2000:2256]")
    ap.add_argument("--near-ood", nargs="+", default=list(NEAR_OOD), choices=list(NEAR_OOD))
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--max-unconstrained", type=int, default=None)
    ap.add_argument("--score-on", choices=["all", "response"], default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="hidden-state indices (0 = embeddings; default: middle and last)")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--knn-train-tokens", type=int, default=50_000)
    ap.add_argument("--shrinkage", type=float, default=1e-3, help="x mean variance, on Sigma")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="figures/baselines_single_pass.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def token_features(model, enc, layers, device, batch_size):
    """Hidden states at ``layers`` and the MAP's entropy / confidence / correctness,
    for every scored token, in seq_index order."""
    feats = {l: [] for l in layers}
    out = {"entropy": [], "conf": [], "correct": []}
    for i in range(0, enc["input_ids"].shape[0], batch_size):
        ids = enc["input_ids"][i:i + batch_size].to(device)
        mask = enc["attention_mask"][i:i + batch_size].to(device)
        valid = enc["loss_mask"][i:i + batch_size, 1:].to(device).bool()
        o = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        logp = o.logits[:, :-1].float().log_softmax(-1)
        p = logp.exp()
        out["entropy"].append(-(p * logp).sum(-1)[valid].cpu())
        conf, pred = p.max(-1)
        out["conf"].append(conf[valid].cpu())
        out["correct"].append((pred == ids[:, 1:])[valid].cpu())
        for l in layers:
            feats[l].append(o.hidden_states[l][:, :-1][valid].float().cpu())
    return {l: torch.cat(v) for l, v in feats.items()}, {k: torch.cat(v) for k, v in out.items()}


class Mahalanobis:
    def __init__(self, X: torch.Tensor, shrinkage: float, device: str):
        X = X.to(device, torch.float64)
        self.mu = X.mean(0)
        C = torch.cov((X - self.mu).T)
        C += shrinkage * C.diagonal().mean() * torch.eye(C.shape[0], device=device, dtype=C.dtype)
        self.L = torch.linalg.cholesky(C)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        Z = torch.linalg.solve_triangular(
            self.L, (X.to(self.L.device, torch.float64) - self.mu).T, upper=False)
        return (Z ** 2).sum(0).float().cpu()


class KNN:
    def __init__(self, X: torch.Tensor, k: int, n: int, device: str, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:n]
        self.T = torch.nn.functional.normalize(X[idx].to(device), dim=1)
        self.k = k

    def __call__(self, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        out = []
        for i in range(0, X.shape[0], chunk):
            q = torch.nn.functional.normalize(X[i:i + chunk].to(self.T.device), dim=1)
            out.append((1 - (q @ self.T.T).topk(self.k, dim=1).values[:, -1]).cpu())
        return torch.cat(out)


def seq_mean(score: torch.Tensor, enc) -> torch.Tensor:
    sid = seq_index(enc)
    n = enc["input_ids"].shape[0]
    tot = torch.zeros(n).index_add_(0, sid, score.float())
    cnt = torch.zeros(n).index_add_(0, sid, torch.ones_like(score, dtype=torch.float))
    keep = cnt > 0
    return tot[keep] / cnt[keep]


def main():
    args = parse_args()
    sargs = torch.load(args.like, weights_only=False)["args"]
    loss_on = sargs.get("loss_on", "all")
    score_on = args.score_on or loss_on

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    n_hidden = model.config.num_hidden_layers
    layers = args.layers or [n_hidden // 2, n_hidden]

    sets = build_analysis_sets(tokenizer, sargs, args, loss_on, score_on)
    feats, maps = {}, {}
    for name, enc in sets.items():
        feats[name], maps[name] = token_features(model, enc, layers, args.device, args.batch_size)
        print(f"features {name}: {next(iter(feats[name].values())).shape[0]} tokens")

    train = ("train_constrained", "train_unconstrained")
    scorers = {}
    for l in layers:
        X = torch.cat([feats[s][l] for s in train])
        scorers[f"mahalanobis_L{l}"] = (l, Mahalanobis(X, args.shrinkage, args.device))
        scorers[f"knn_L{l}"] = (l, KNN(X, args.knn_k, args.knn_train_tokens, args.device))

    eval_sets = [s for s in sets if s not in train]
    tok = {s: {"map_entropy": maps[s]["entropy"], "map_1-conf": 1 - maps[s]["conf"]}
           for s in eval_sets}
    for name, (l, fn) in scorers.items():
        for s in eval_sets:
            tok[s][name] = fn(feats[s][l])

    oods = [s for s in eval_sets if s != "test"]
    report = {"loss_on": loss_on, "score_on": score_on, "layers": layers, "ood": {}, "error": {}}
    for score in tok["test"]:
        base_seq = seq_mean(tok["test"][score], sets["test"])
        report["ood"][score] = {s: auroc(base_seq, seq_mean(tok[s][score], sets[s])) for s in oods}

    near = [s for s in oods if s != "wikitext"]
    for scope, group in (("test", ["test"]), ("test+wikitext", ["test", "wikitext"]),
                         ("test+near_ood", ["test"] + near)):
        wrong = torch.cat([~maps[s]["correct"] for s in group])
        ent = torch.cat([maps[s]["entropy"] for s in group])
        report["error"][scope] = {}
        for score in tok["test"]:
            x = torch.cat([tok[s][score] for s in group])
            report["error"][scope][score] = {
                "raw": auroc(x[~wrong], x[wrong]),
                "within_entropy": stratified_auroc(wrong, x, ent, args.n_bins)["weighted"],
            }

    print(f"\nOOD AUROC (per-sequence mean score; score_on={score_on})")
    print(f"  {'score':22s} " + " ".join(f"{s:>11s}" for s in oods))
    for score, r in report["ood"].items():
        print(f"  {score:22s} " + " ".join(f"{r[s]:11.3f}" for s in oods))
    print("\nerror AUROC, test + near-OOD tokens: raw / within MAP-entropy deciles")
    for score, r in report["error"]["test+near_ood"].items():
        print(f"  {score:22s} {r['raw']:.3f} / {r['within_entropy']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "args": vars(args), "sampler_args": sargs},
                              indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
