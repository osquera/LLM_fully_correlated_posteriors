"""Stage 3: evaluate every posterior method on train, held-out test and OOD text.

Three sets, all tokenised like the fine-tuning data:

  train  the exact sequences behind J^L (read from the samples file). This is
         where Lemma 4.3 speaks: we report the worst change in any sequence's
         mean NLL relative to theta_map.
  test   a held-out slice of the fine-tuning dataset (default Alpaca
         train[2000:2256]; fine-tuning uses train[:2000]).
  ood    WikiText-2 test.

For each method and set, with p_s the next-token distribution under sample s
and p_bar their mean (the Bayesian model average):

  nll_bma      mean over tokens of -log p_bar(y)
  nll_sample   mean over samples and tokens of -log p_s(y)
  acc          top-1 next-token accuracy of p_bar
  ece          expected calibration error of p_bar's top-1 confidence (15 bins)
  entropy      H(p_bar), total predictive uncertainty
  mi           H(p_bar) - mean_s H(p_s), the epistemic part (mutual information)

and, on ood, the AUROC for separating test from ood sequences by their mean
mi (by mean entropy for MAP, which has no epistemic term).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.data import encode_texts
from llmproj.paramspace import ParamSpace

N_BINS = 15


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+",
                    default=["samples/posterior_sequence.pt", "samples/posterior_token.pt"],
                    help="files from sample_posterior.py; missing ones are skipped")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--test-split", default="train[2000:2256]",
                    help="held out: must not overlap the fine-tuning split")
    ap.add_argument("--ood-dataset", default="wikitext")
    ap.add_argument("--ood-config", default="wikitext-2-raw-v1")
    ap.add_argument("--ood-split", default="test")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="figures/posterior_eval.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def encode(tokenizer, texts, max_length, loss_on="all"):
    texts = [t for t in texts if t and t.strip()]
    return encode_texts(tokenizer, texts, max_length, loss_on=loss_on)


def auroc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """P(score_pos > score_neg), ties counted half (Mann-Whitney U).

    Rank-based, so it scales to the ~65k tokens of a set."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    x = np.concatenate([neg.double().numpy(), pos.double().numpy()])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]  # average rank of ties
    n0, n1 = len(neg), len(pos)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def selective(correct: torch.Tensor, score: torch.Tensor) -> dict:
    """Is the model wrong where it is uncertain? ``score`` is an uncertainty
    (higher = less sure), one per token.

      err_auroc  P(a wrong token scores higher than a right one); 0.5 = blind
      aurc       area under the risk-coverage curve: answer only the most
                 certain fraction c of tokens, average the error rate over c.
                 Lower is better; the oracle that rejects exactly the errors
                 gets aurc_oracle, which depends only on the error rate
      acc@c      accuracy on the most certain c of the tokens; should rise as
                 c falls if the uncertainty is informative
    """
    c = correct.bool()
    order = torch.argsort(score.double(), stable=True)
    err = (~c[order]).double()
    risk = err.cumsum(0) / torch.arange(1, len(err) + 1, dtype=torch.float64)
    out = {"err_auroc": auroc(score[c], score[~c]), "aurc": float(risk.mean())}
    for cov in (0.5, 0.8):
        k = max(1, int(cov * len(err)))
        out[f"acc@{cov:g}"] = float(1 - err[:k].mean())
    return out


def selective_report(tok: dict, with_mi: bool) -> dict:
    """`selective` for each uncertainty score of one token set."""
    scores = {"entropy": tok["entropy"], "1-conf": 1 - tok["conf"]}
    if with_mi:
        scores["mi"] = tok["mi"]
    rep = {k: selective(tok["correct"], v) for k, v in scores.items()}
    n, e = len(tok["correct"]), float(1 - tok["correct"].float().mean())
    # risk-coverage of the oracle: all right tokens first
    k = int(round(n * (1 - e)))
    i = torch.arange(1, n + 1, dtype=torch.float64)
    rep["aurc_oracle"] = float(((i - k).clamp(min=0) / i).mean())
    return rep


@torch.no_grad()
def evaluate_set(model, frozen, theta, space, deltas, enc, device, batch_size):
    """Metrics of the BMA over ``deltas`` (n_samples, P) on one encoded set."""
    n_seq = enc["input_ids"].size(0)
    tot = {"nll_bma": 0.0, "nll_sample": 0.0, "correct": 0.0, "entropy": 0.0, "mi": 0.0}
    ntok = 0
    bin_n = torch.zeros(N_BINS)
    bin_conf = torch.zeros(N_BINS)
    bin_acc = torch.zeros(N_BINS)
    seq_mi = torch.zeros(n_seq)
    seq_h = torch.zeros(n_seq)
    seq_nll = torch.zeros(deltas.shape[0], n_seq)  # per sample, per sequence
    tok = {"correct": [], "conf": [], "entropy": [], "mi": []}  # per valid token

    params = []
    for d in deltas:
        upd = space.unflatten(d.to(device=space.device, dtype=space.dtype))
        params.append({**frozen, **{n: theta[n] + u for n, u in upd.items()}})

    for i in range(0, n_seq, batch_size):
        ids = enc["input_ids"][i : i + batch_size].to(device)
        mask = enc["attention_mask"][i : i + batch_size].to(device)
        tgt = ids[:, 1:]
        valid = enc["loss_mask"][i : i + batch_size, 1:].to(device).bool()
        vf = valid.float()

        p_bar, h_mean = None, None
        for s, prm in enumerate(params):
            out = functional_call(model, prm, args=(),
                                  kwargs={"input_ids": ids, "attention_mask": mask})
            logp = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
            p = logp.exp()
            nll = -logp.gather(-1, tgt[..., None]).squeeze(-1)
            h = -(p * logp).sum(-1)
            seq_nll[s, i : i + ids.size(0)] = ((nll * vf).sum(1) / vf.sum(1).clamp(min=1)).cpu()
            tot["nll_sample"] += float(nll[valid].sum())
            p_bar = p if p_bar is None else p_bar + p
            h_mean = h if h_mean is None else h_mean + h
            del out, logp, p
        p_bar /= len(params)
        h_mean /= len(params)

        logp_bar = p_bar.clamp_min(1e-30).log()
        nll_bma = -logp_bar.gather(-1, tgt[..., None]).squeeze(-1)
        h_bar = -(p_bar * logp_bar).sum(-1)
        mi = (h_bar - h_mean).clamp_min(0.0)
        conf, pred = p_bar.max(-1)
        correct = (pred == tgt).float()

        tot["nll_bma"] += float(nll_bma[valid].sum())
        tot["correct"] += float(correct[valid].sum())
        tot["entropy"] += float(h_bar[valid].sum())
        tot["mi"] += float(mi[valid].sum())
        ntok += int(valid.sum())
        seq_mi[i : i + ids.size(0)] = ((mi * vf).sum(1) / vf.sum(1).clamp(min=1)).cpu()
        seq_h[i : i + ids.size(0)] = ((h_bar * vf).sum(1) / vf.sum(1).clamp(min=1)).cpu()

        c, a = conf[valid].cpu(), correct[valid].cpu()
        tok["correct"].append(a.bool())
        tok["conf"].append(c)
        tok["entropy"].append(h_bar[valid].cpu())
        tok["mi"].append(mi[valid].cpu())
        b = (c * N_BINS).long().clamp(max=N_BINS - 1)
        bin_n += torch.bincount(b, minlength=N_BINS).float()
        bin_conf += torch.bincount(b, weights=c, minlength=N_BINS)
        bin_acc += torch.bincount(b, weights=a, minlength=N_BINS)
        del p_bar, h_mean, logp_bar

    nz = bin_n > 0
    ece = float((bin_n[nz] / ntok * (bin_acc[nz] / bin_n[nz] - bin_conf[nz] / bin_n[nz]).abs()).sum())
    return {
        "nll_bma": tot["nll_bma"] / ntok,
        "nll_sample": tot["nll_sample"] / (ntok * len(params)),
        "acc": tot["correct"] / ntok,
        "ece": ece,
        "entropy": tot["entropy"] / ntok,
        "mi": tot["mi"] / ntok,
        "n_tokens": ntok,
    }, {"mi": seq_mi, "entropy": seq_h,
        "tok": {k: torch.cat(v) for k, v in tok.items()}}, seq_nll


def build_sets(tokenizer, sargs, args):
    """train / test / ood encodings. ``sargs`` are the sampler's args, ``args``
    need test_split, n_eval and the ood_* fields."""
    # train: the very sequences behind J^L, taken from the sampler's own args
    ds = load_dataset(sargs["dataset"], split=sargs["split"])
    field = sargs["text_field"] if sargs["text_field"] in ds.column_names else ds.column_names[0]
    max_len = sargs["max_length"]
    # train and test score the tokens the projection constrained; WikiText has
    # no response, so ood always scores every token
    loss_on = sargs.get("loss_on", "all")
    train_enc = encode(tokenizer, list(ds[field])[: sargs["n_train"]], max_len, loss_on)
    test_ds = load_dataset(sargs["dataset"], split=args.test_split)
    test_enc = encode(tokenizer, list(test_ds[field])[: args.n_eval], max_len, loss_on)
    ood_ds = load_dataset(args.ood_dataset, args.ood_config, split=args.ood_split)
    ood_field = "text" if "text" in ood_ds.column_names else ood_ds.column_names[0]
    # WikiText has many short lines; keep the long ones so the sets are comparable
    ood_texts = [t for t in ood_ds[ood_field] if len(t.split()) >= 64]
    ood_enc = encode(tokenizer, ood_texts[: args.n_eval], max_len)
    sets = {"train": train_enc, "test": test_enc, "ood": ood_enc}
    print({k: tuple(v["input_ids"].shape) for k, v in sets.items()})
    return sets


def evaluate_method(model, frozen, theta, space, deltas, sets, device, batch_size,
                    map_seq_nll=None):
    """All sets for one method. Pass ``map_seq_nll=None`` for the MAP itself;
    returns the results and the per-sample, per-train-sequence NLL."""
    res, scores = {}, {}
    for sname, enc in sets.items():
        r, seq_scores, seq_nll = evaluate_set(model, frozen, theta, space, deltas, enc,
                                              device, batch_size)
        res[sname] = r
        scores[sname] = seq_scores
        if sname == "train":
            train_nll = seq_nll
            if map_seq_nll is None:
                res["train"]["worst_seq_dnll"] = 0.0
                res["train"]["mean_abs_seq_dnll"] = 0.0
            else:
                dn = (seq_nll - map_seq_nll[None]).abs()
                res["train"]["worst_seq_dnll"] = float(dn.max())
                res["train"]["mean_abs_seq_dnll"] = float(dn.mean())
    # MAP has one sample, so its mi is identically zero: score it by entropy
    key = "entropy" if map_seq_nll is None else "mi"
    res["ood"]["auroc"] = auroc(scores["test"][key], scores["ood"][key])
    # Wrong where uncertain, right where certain: per set, and pooled over
    # test + ood, where a good uncertainty must also rank the (mostly wrong)
    # ood tokens below the in-distribution ones.
    with_mi = map_seq_nll is not None
    for sname in ("test", "ood"):
        res[sname]["selective"] = selective_report(scores[sname]["tok"], with_mi)
    pooled = {k: torch.cat([scores["test"]["tok"][k], scores["ood"]["tok"][k]])
              for k in scores["test"]["tok"]}
    res["pooled"] = {"selective": selective_report(pooled, with_mi)}
    res["norm"] = float(deltas.norm(dim=1).mean())
    return res, train_nll


def print_selective(results):
    """Is each method wrong where it is uncertain? (see `selective`)"""
    print("\nselective prediction: err_auroc (higher better) / aurc (lower better), "
          "per uncertainty score")
    print(f"{'method':40s} {'set':7s} {'entropy':>15s} {'1-conf':>15s} {'mi':>15s} "
          f"{'acc@0.5':>8s} {'oracle':>7s}")
    for name, r in results.items():
        for sname in ("test", "ood", "pooled"):
            sel = r[sname]["selective"]
            cells = [f"{sel[k]['err_auroc']:.3f}/{sel[k]['aurc']:.3f}" if k in sel else "-"
                     for k in ("entropy", "1-conf", "mi")]
            print(f"{name:40s} {sname:7s} {cells[0]:>15s} {cells[1]:>15s} {cells[2]:>15s} "
                  f"{sel['entropy']['acc@0.5']:8.4f} {sel['aurc_oracle']:7.3f}")


def main():
    args = parse_args()
    files = [Path(p) for p in args.samples if Path(p).exists()]
    if not files:
        raise SystemExit(f"none of {args.samples} exist")
    blobs = {p: torch.load(p, weights_only=False) for p in files}
    sargs = next(iter(blobs.values()))["args"]
    same = ("dataset", "split", "text_field", "n_train", "max_length", "loss_on")
    for blob in blobs.values():
        blob["args"].setdefault("loss_on", "all")  # files from before --loss-on
    for p, blob in blobs.items():
        if any(blob["args"][k] != sargs[k] for k in same):
            raise SystemExit(f"{p} was projected against different training data: "
                             f"{ {k: blob['args'][k] for k in same} } vs { {k: sargs[k] for k in same} }")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta = space.theta(model)
    frozen = space.frozen(model)

    sets = build_sets(tokenizer, sargs, args)

    methods = {"map": torch.zeros(1, space.P)}
    meta = {}
    for p, blob in blobs.items():
        assert blob["info"]["P"] == space.P, f"{p}: P mismatch"
        mode = blob["info"]["loss_mode"]
        meta[mode] = blob["info"]
        for m, d in blob["methods"].items():
            methods[f"{mode}/{m}"] = d

    results, map_seq_nll = {}, None
    for name, deltas in methods.items():
        res, seq_nll = evaluate_method(model, frozen, theta, space, deltas, sets,
                                       args.device, args.batch_size, map_seq_nll)
        if name == "map":
            map_seq_nll = seq_nll[0]
        results[name] = res
        print(f"{name:40s} test nll_bma={res['test']['nll_bma']:.4f} "
              f"ece={res['test']['ece']:.4f}  train worst dnll="
              f"{res['train'].get('worst_seq_dnll', 0):.3g}  auroc={res['ood']['auroc']}")

    print("\n" + "=" * 100)
    hdr = (f"{'method':40s} {'norm':>8s} {'trainΔ':>9s} {'test NLL':>9s} {'acc':>7s} "
           f"{'ECE':>7s} {'test MI':>9s} {'ood MI':>9s} {'AUROC':>7s}")
    print(hdr)
    for name, r in results.items():
        print(f"{name:40s} {r['norm']:8.3g} {r['train']['worst_seq_dnll']:9.3g} "
              f"{r['test']['nll_bma']:9.4f} {r['test']['acc']:7.4f} {r['test']['ece']:7.4f} "
              f"{r['test']['mi']:9.3g} {r['ood']['mi']:9.3g} {r['ood']['auroc']:7.3f}")
    print("trainΔ = worst |change in a train sequence's mean NLL| over all samples "
          "(Lemma 4.3); AUROC uses MI (entropy for map)")
    print_selective(results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "sampling": meta,
                               "args": vars(args)}, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
