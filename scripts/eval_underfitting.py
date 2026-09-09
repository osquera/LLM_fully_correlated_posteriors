"""Stage 3: test the paper's two central claims on the LLM.

Claim 1 (Lemma 4.3, "no underfitting"): samples from the projected posterior
preserve the fine-tuning loss. The control is the *same* Gaussian without the
projection -- eps ~ N(0, alpha^-1 I), which is what an isotropic/diagonal
approximation samples. If the projection is doing its job, projected samples
leave train perplexity essentially unchanged while unprojected ones wreck it.

Claim 2 (Lemma 3.2, OOD variance): predictive spread must be strictly larger
off the training distribution than on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.func import functional_call
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.paramspace import ParamSpace


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--samples", default="samples/projected.pt")
    ap.add_argument("--id-dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--id-split", default="train[:2000]")
    ap.add_argument("--ood-dataset", default="wikitext")
    ap.add_argument("--ood-config", default="wikitext-2-raw-v1")
    ap.add_argument("--ood-split", default="test")
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="figures/underfitting.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def encode(tokenizer, texts, max_length):
    texts = [t for t in texts if t and t.strip()]
    return tokenizer(texts, truncation=True, max_length=max_length,
                     padding="max_length", return_tensors="pt")


@torch.no_grad()
def nll_and_entropy(model, params, enc, device, batch_size):
    """Mean token NLL and mean predictive entropy over an encoded set."""
    nlls, ents, ntok = 0.0, 0.0, 0
    for i in range(0, enc["input_ids"].size(0), batch_size):
        ids = enc["input_ids"][i : i + batch_size].to(device)
        mask = enc["attention_mask"][i : i + batch_size].to(device)
        out = functional_call(model, params, args=(),
                              kwargs={"input_ids": ids, "attention_mask": mask})
        logits = out.logits[:, :-1, :].float()
        tgt = ids[:, 1:]
        valid = mask[:, 1:].bool()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                              reduction="none").view(tgt.shape)
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        nlls += float(nll[valid].sum())
        ents += float(ent[valid].sum())
        ntok += int(valid.sum())
    return nlls / ntok, ents / ntok


def main():
    args = parse_args()
    blob = torch.load(args.samples, weights_only=False)
    deltas = blob["deltas"]
    print(f"loaded {deltas.shape[0]} samples, P = {deltas.shape[1]:,}")
    print("sampling info:", json.dumps(
        {k: v for k, v in blob["info"].items() if k != "batch_ranks"}, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()

    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"])
    assert space.P == deltas.shape[1], f"P mismatch: {space.P} vs {deltas.shape[1]}"
    theta = space.theta(model)
    frozen = space.frozen(model)

    id_ds = load_dataset(args.id_dataset, split=args.id_split)
    id_field = "text" if "text" in id_ds.column_names else id_ds.column_names[0]
    id_enc = encode(tokenizer, list(id_ds[id_field])[: args.n_eval], args.max_length)

    ood_ds = load_dataset(args.ood_dataset, args.ood_config, split=args.ood_split)
    ood_field = "text" if "text" in ood_ds.column_names else ood_ds.column_names[0]
    ood_enc = encode(tokenizer, list(ood_ds[ood_field])[: args.n_eval * 4], args.max_length)
    ood_enc = {k: v[: args.n_eval] for k, v in ood_enc.items()}

    results = {"map": {}, "projected": [], "unprojected": []}

    def evaluate(delta):
        params = {**frozen, **{n: theta[n] + d for n, d in space.unflatten(delta).items()}}
        idn, ide = nll_and_entropy(model, params, id_enc, args.device, args.batch_size)
        oon, ooe = nll_and_entropy(model, params, ood_enc, args.device, args.batch_size)
        return {"id_nll": idn, "id_entropy": ide, "ood_nll": oon, "ood_entropy": ooe}

    zero = torch.zeros(space.P)
    results["map"] = evaluate(zero)
    print(f"\nMAP: id_nll={results['map']['id_nll']:.4f}  ood_nll={results['map']['ood_nll']:.4f}")

    alpha = blob["info"]["alpha"]
    gen = torch.Generator().manual_seed(1234)

    for i in tqdm(range(deltas.shape[0]), desc="projected"):
        results["projected"].append(evaluate(deltas[i].to(space.dtype)))
    # control: identical marginal scale, no projection
    for i in tqdm(range(deltas.shape[0]), desc="unprojected control"):
        eps = torch.randn(space.P, generator=gen) / (alpha ** 0.5)
        results["unprojected"].append(evaluate(eps.to(space.dtype)))

    def summarise(rows, key):
        vals = torch.tensor([r[key] for r in rows])
        return {"mean": float(vals.mean()), "std": float(vals.std()) if len(vals) > 1 else 0.0}

    summary = {"map": results["map"]}
    for grp in ("projected", "unprojected"):
        summary[grp] = {k: summarise(results[grp], k)
                        for k in ("id_nll", "id_entropy", "ood_nll", "ood_entropy")}

    m = results["map"]
    print("\n=== Claim 1 (Lemma 4.3): train-set loss preservation ===")
    for grp in ("projected", "unprojected"):
        d = summary[grp]["id_nll"]["mean"] - m["id_nll"]
        print(f"  {grp:12s} id_nll = {summary[grp]['id_nll']['mean']:.4f}  "
              f"(delta vs MAP: {d:+.4f})")
    print("  -> projected delta should be near zero; unprojected should be clearly worse")

    print("\n=== Claim 2 (Lemma 3.2): OOD spread exceeds in-distribution ===")
    p = summary["projected"]
    print(f"  projected id_entropy  = {p['id_entropy']['mean']:.4f} +- {p['id_entropy']['std']:.4f}")
    print(f"  projected ood_entropy = {p['ood_entropy']['mean']:.4f} +- {p['ood_entropy']['std']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "raw": results,
                               "sampling_info": blob["info"]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
