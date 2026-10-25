"""Three follow-up analyses on existing samples (no re-projection).

1. Does MI add information beyond the softmax?
   Target: the MAP's next-token prediction is wrong. Per scope (test; test +
   WikiText; test + near-OOD), compare
     - error AUROC of MI *within* deciles of MAP entropy (stratified), so MI
       cannot win by re-measuring what the entropy already ranks;
     - a logistic regression on [MAP entropy, MAP 1-conf] against the same
       plus log MI, 2-fold cross-validated with folds split by sequence.
       A held-out gain in AUROC / log loss is information the softmax lacks.

2. The training data the projection never saw.
   The sampler constrains the first n_train sequences of the fine-tuning split;
   the rest (e.g. 1,744 of 2,000) were fit by theta_map but not projected
   against. Per method: |change in a sequence's mean NLL| on both groups
   (mean, 95th percentile, max, fraction above the alpha tolerance).

3. Near-OOD in the Alpaca template.
   Dolly, CodeAlpaca, GSM8K and Spanish Alpaca, each formatted exactly like
   Alpaca's "text" field and scored on the same tokens as the test set. AUROC
   for telling them from the Alpaca test set by mean MI (posteriors) and by
   mean entropy (MAP and BMA), next to WikiText for reference.

Writes figures/analysis.json (and per-token arrays to --tokens-out if given).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_posterior import auroc, encode, evaluate_set
from llmproj.data import alpaca_text, load_texts
from llmproj.paramspace import ParamSpace

# name -> (hf path, config, split, row -> (instruction, input, output))
NEAR_OOD = {
    "dolly": ("databricks/databricks-dolly-15k", None, "train",
              lambda r: (r["instruction"], r["context"], r["response"])),
    "codealpaca": ("sahil2801/CodeAlpaca-20k", None, "train",
                   lambda r: (r["instruction"], r["input"], r["output"])),
    "gsm8k": ("openai/gsm8k", "main", "test",
              lambda r: (r["question"], "", r["answer"])),
    "alpaca_es": ("bertin-project/alpaca-spanish", None, "train",
                  lambda r: (r["instruction"], r["input"], r["output"])),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+",
                    default=["samples/posterior_sequence.pt", "samples/posterior_token.pt"])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="method keys to analyse (default: every *_measured one)")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--test-split", default="train[2000:2256]")
    ap.add_argument("--near-ood", nargs="+", default=list(NEAR_OOD), choices=list(NEAR_OOD))
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--score-on", choices=["all", "response"], default=None,
                    help="tokens of the test and near-OOD sets to score (default: the "
                         "constrained ones); 'response' leaves out the shared template")
    ap.add_argument("--max-unconstrained", type=int, default=None,
                    help="cap on the never-projected train sequences (default: all)")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="figures/analysis.json")
    ap.add_argument("--tokens-out", default=None, help="optional .pt with per-token arrays")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ---------------------------------------------------------------- analysis 1

def stratified_auroc(wrong, score, strata, n_bins):
    """Error AUROC of ``score`` within quantile bins of ``strata``, size-weighted."""
    edges = torch.quantile(strata.double(), torch.linspace(0, 1, n_bins + 1, dtype=torch.float64))
    b = torch.bucketize(strata.double(), edges[1:-1])
    tot, w, per_bin = 0.0, 0, []
    for k in range(n_bins):
        m = b == k
        y, s = wrong[m], score[m]
        if y.all() or (~y).all():
            per_bin.append(None)
            continue
        a = auroc(s[~y], s[y])
        per_bin.append(a)
        tot += a * int(m.sum())
        w += int(m.sum())
    return {"weighted": tot / max(w, 1), "per_bin": per_bin,
            "bin_error_rate": [float(wrong[b == k].float().mean()) for k in range(n_bins)]}


def fit_logistic(X, y, l2=1e-3, steps=100):
    """Logistic regression by L-BFGS on standardised features; returns a predictor."""
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-8)
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = Z @ w[1:] + w[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + l2 * (w[1:] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    w = w.detach()
    return lambda Xn: ((Xn - mu) / sd) @ w[1:] + w[0]


def logistic_gain(wrong, feats, seq_id):
    """Held-out AUROC / log loss of each feature set, 2-fold split by sequence."""
    y = wrong.double()
    fold = (seq_id % 2).bool()
    out = {}
    for name, X in feats.items():
        X = X.double()
        aucs, nlls = [], []
        for test_fold in (False, True):
            tr, te = fold != test_fold, fold == test_fold
            pred = fit_logistic(X[tr], y[tr])
            logit = pred(X[te])
            aucs.append(auroc(logit[~wrong[te]], logit[wrong[te]]))
            nlls.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logit, y[te])))
        out[name] = {"auroc": sum(aucs) / 2, "log_loss": sum(nlls) / 2}
    return out


def mi_beyond_softmax(map_tok, tok, seq_ids, n_bins):
    wrong = ~map_tok["correct"].bool()
    h, c = map_tok["entropy"].double(), 1 - map_tok["conf"].double()
    log_mi = torch.log(tok["mi"].double() + 1e-10)
    return {
        "n_tokens": int(wrong.numel()),
        "map_error_rate": float(wrong.float().mean()),
        "raw_auroc": {"map_entropy": auroc(h[~wrong], h[wrong]),
                      "map_1-conf": auroc(c[~wrong], c[wrong]),
                      "mi": auroc(log_mi[~wrong], log_mi[wrong])},
        "mi_within_entropy_bins": stratified_auroc(wrong, log_mi, h, n_bins),
        "logistic": logistic_gain(wrong, {
            "softmax": torch.stack([h, c], 1),
            "softmax+mi": torch.stack([h, c, log_mi], 1),
            "mi": log_mi[:, None],
        }, seq_ids),
    }


# ---------------------------------------------------------------- analysis 2

def dnll_summary(seq_nll, map_nll, tol):
    d = (seq_nll - map_nll[None]).abs()  # (samples, sequences)
    worst_per_seq = d.max(0).values
    return {"n_seq": int(d.shape[1]), "mean_abs": float(d.mean()),
            "p95": float(torch.quantile(d.flatten().double(), 0.95)),
            "max": float(d.max()),
            "frac_seq_over_tol": float((worst_per_seq > tol).float().mean())}


# ---------------------------------------------------------------- main

def seq_index(enc):
    """Sequence id of every scored token, in evaluate_set's order."""
    valid = enc["loss_mask"][:, 1:].bool()
    return torch.arange(valid.shape[0])[:, None].expand_as(valid)[valid]


def main():
    args = parse_args()
    files = [Path(p) for p in args.samples if Path(p).exists()]
    if not files:
        raise SystemExit(f"none of {args.samples} exist")
    blobs = {p: torch.load(p, weights_only=False) for p in files}
    sargs = next(iter(blobs.values()))["args"]
    loss_on = sargs.get("loss_on", "all")
    score_on = args.score_on or loss_on  # test and near-OOD; train stays on loss_on
    tol = next(iter(blobs.values()))["info"]["alpha_tol_nats"]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)

    # --- sets
    max_len = sargs["max_length"]
    texts = load_texts(sargs["dataset"], sargs["split"], sargs["text_field"])
    n = sargs["n_train"]
    unconstrained = texts[n:][: args.max_unconstrained]
    test = load_texts(sargs["dataset"], args.test_split, sargs["text_field"])
    sets = {
        "train_constrained": encode(tokenizer, texts[:n], max_len, loss_on),
        "train_unconstrained": encode(tokenizer, unconstrained, max_len, loss_on),
        "test": encode(tokenizer, test[: args.n_eval], max_len, score_on),
    }
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    sets["wikitext"] = encode(tokenizer, [t for t in wiki["text"] if len(t.split()) >= 64]
                              [: args.n_eval], max_len, "all")
    for name in args.near_ood:
        path, cfg, split, row = NEAR_OOD[name]
        d = load_dataset(path, cfg, split=split).shuffle(seed=0).select(range(args.n_eval))
        sets[name] = encode(tokenizer, [alpaca_text(*row(r)) for r in d], max_len, score_on)
    print({k: (tuple(v["input_ids"].shape), int(v["loss_mask"].sum())) for k, v in sets.items()},
          f"loss_on={loss_on} score_on={score_on}")
    near = [s for s in sets if s not in ("train_constrained", "train_unconstrained", "test")]

    methods = {"map": torch.zeros(1, space.P)}
    for p, blob in blobs.items():
        mode = blob["info"]["loss_mode"]
        for m, d in blob["methods"].items():
            key = f"{mode}/{m}"
            if (args.methods and key not in args.methods and m not in args.methods) or \
               (not args.methods and not m.endswith("_measured")):
                continue
            methods[key] = d

    # --- evaluate everything once
    raw = {}
    for name, deltas in methods.items():
        raw[name] = {}
        for sname, enc in sets.items():
            metrics, scores, seq_nll = evaluate_set(model, frozen, theta, space, deltas.float(),
                                                    enc, args.device, args.batch_size)
            raw[name][sname] = {"metrics": metrics, "scores": scores, "seq_nll": seq_nll}
        print(f"evaluated {name}")

    report = {"loss_on": loss_on, "score_on": score_on, "tol_nats": tol, "methods": list(methods),
              "sets": {k: int(v["input_ids"].shape[0]) for k, v in sets.items()},
              "mi_beyond_softmax": {}, "unconstrained_train": {}, "near_ood": {}}
    seq_ids = {s: seq_index(sets[s]) for s in sets}

    # 1. MI beyond the softmax
    scopes = {"test": ["test"], "test+wikitext": ["test", "wikitext"],
              "test+near_ood": ["test"] + [s for s in near if s != "wikitext"]}
    for name in methods:
        if name == "map":
            continue
        report["mi_beyond_softmax"][name] = {}
        for scope, members in scopes.items():
            def cat(src, key):
                return torch.cat([src[s]["scores"]["tok"][key] for s in members])
            ids, off = [], 0
            for s in members:  # sequence ids unique across the pooled sets
                ids.append(seq_ids[s] + off)
                off += sets[s]["input_ids"].shape[0]
            map_tok = {k: cat(raw["map"], k) for k in ("correct", "conf", "entropy")}
            tok = {"mi": cat(raw[name], "mi")}
            report["mi_beyond_softmax"][name][scope] = mi_beyond_softmax(
                map_tok, tok, torch.cat(ids), args.n_bins)

    # 2. unconstrained train
    for name in methods:
        if name == "map":
            continue
        report["unconstrained_train"][name] = {
            g: dnll_summary(raw[name][g]["seq_nll"], raw["map"][g]["seq_nll"][0], tol)
            for g in ("train_constrained", "train_unconstrained")}

    # 3. near-OOD
    for name in methods:
        r = {}
        t = raw[name]["test"]
        for s in near:
            o = raw[name][s]
            r[s] = {"nll_bma": o["metrics"]["nll_bma"], "acc": o["metrics"]["acc"],
                    "mi": o["metrics"]["mi"], "entropy": o["metrics"]["entropy"],
                    "auroc_entropy": auroc(t["scores"]["entropy"], o["scores"]["entropy"])}
            if name != "map":
                r[s]["auroc_mi"] = auroc(t["scores"]["mi"], o["scores"]["mi"])
        r["test"] = {k: t["metrics"][k] for k in ("nll_bma", "acc", "mi", "entropy")}
        report["near_ood"][name] = r

    # --- print
    print("\n1. MI beyond the softmax (target: MAP wrong). raw AUROC entropy / 1-conf / MI; "
          "MI within entropy deciles; logistic held-out AUROC softmax -> softmax+MI")
    for name, per in report["mi_beyond_softmax"].items():
        for scope, r in per.items():
            lg = r["logistic"]
            print(f"  {name:32s} {scope:14s} raw {r['raw_auroc']['map_entropy']:.3f} / "
                  f"{r['raw_auroc']['map_1-conf']:.3f} / {r['raw_auroc']['mi']:.3f}   "
                  f"stratified {r['mi_within_entropy_bins']['weighted']:.3f}   "
                  f"logistic {lg['softmax']['auroc']:.4f} -> {lg['softmax+mi']['auroc']:.4f} "
                  f"(log loss {lg['softmax']['log_loss']:.4f} -> {lg['softmax+mi']['log_loss']:.4f})")
    print(f"\n2. |Δ seq NLL| on train sequences, constrained vs never projected (tol {tol:.4f})")
    for name, g in report["unconstrained_train"].items():
        a, b = g["train_constrained"], g["train_unconstrained"]
        print(f"  {name:32s} mean {a['mean_abs']:.4f} -> {b['mean_abs']:.4f}   "
              f"p95 {a['p95']:.4f} -> {b['p95']:.4f}   max {a['max']:.4f} -> {b['max']:.4f}   "
              f"over tol {a['frac_seq_over_tol']:.1%} -> {b['frac_seq_over_tol']:.1%}")
    print("\n3. near-OOD vs Alpaca test: AUROC by MI (by entropy for map) | BMA NLL")
    print(f"  {'method':32s} " + " ".join(f"{s:>16s}" for s in near))
    for name, r in report["near_ood"].items():
        key = "auroc_entropy" if name == "map" else "auroc_mi"
        print(f"  {name:32s} " + " ".join(f"{r[s][key]:7.3f} | {r[s]['nll_bma']:6.3f}" for s in near))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "args": vars(args), "sampler_args": sargs},
                              indent=2, default=str))
    print(f"\nwrote {out}")
    if args.tokens_out:
        torch.save({n: {s: v["scores"]["tok"] for s, v in r.items()} for n, r in raw.items()},
                   args.tokens_out)
        print(f"wrote {args.tokens_out}")


if __name__ == "__main__":
    main()
