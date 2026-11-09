"""Membership-inference baselines on the probe questions (E49 control).

Every trained fact is probed with its exact training question, so a model
can tell trained from held-out probes by how familiar the *question text*
is, without any notion of which facts it knows. These single-pass scores
test whether that is all the posterior picks up. Each is computed on the
probe prompt's tokens (the question, not the answer), higher = less
familiar:

  question NLL        mean token NLL under the MAP (loss attack)
  question min-20%    mean of the 20% highest token NLLs (Min-K% Prob,
                      Shi et al. 2024)
  question NLL ratio  MAP NLL minus the base model's NLL (reference-model
                      attack: how much fine-tuning lowered it; the strongest
                      of the three in the membership-inference literature)

They are added to an evaluate_bio.py output (--eval) like
semantic_entropy_bio.py does: AUROC raw and within MAP-entropy quintiles,
the held-out logistic gain over the softmax, and probe scores for
bootstrap_bio.py.

  uv run python scripts/membership_scores.py --corpus data/tofuh1/corpus.json \\
      --adapter checkpoints/tofuh1_ep3_lora --eval figures/tofuh_n48_eval1.json \\
      --out figures/e49_tofuh1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from analyze_posterior import logistic_gain, stratified_auroc
from evaluate_bio import FREQ_TASKS
from evaluate_posterior import auroc
from llmproj.models import load_finetuned


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True, help="evaluate_bio.py output to add the scores to")
    ap.add_argument("--min-k", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


@torch.no_grad()
def prompt_token_nll(model, tok, prompts, args) -> list[torch.Tensor]:
    """Per-token NLL of each prompt (tokens 2..n), as a list of 1-D tensors."""
    tok.padding_side = "right"
    out = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i: i + args.batch_size], return_tensors="pt", padding=True).to(args.device)
        logits = model(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        m = enc["attention_mask"][:, 1:].bool()
        out += [nll[j][m[j]].cpu() for j in range(len(tgt))]
    return out


def main():
    args = parse_args()
    probes = json.loads(Path(args.corpus).read_text())["probes"]
    ev = json.load(open(args.eval))
    r = ev["report"]
    if [q["freq"] for q in probes] != r["probe_freq"]:
        raise SystemExit("eval and corpus probes differ")
    tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, _ = load_finetuned(args.base_model, args.adapter, args.device)
    prompts = [q["prompt"] for q in probes]
    tuned = prompt_token_nll(model, tok, prompts, args)
    with model.disable_adapter():
        base = prompt_token_nll(model, tok, prompts, args)

    def min_k(x):
        k = max(1, int(round(args.min_k * len(x))))
        return float(x.topk(k).values.mean())

    scores = {
        "question NLL": torch.tensor([float(x.mean()) for x in tuned], dtype=torch.float64),
        "question min-20%": torch.tensor([min_k(x) for x in tuned], dtype=torch.float64),
        "question NLL ratio": torch.tensor([float(a.mean() - b.mean()) for a, b in zip(tuned, base)],
                                           dtype=torch.float64),
    }

    freq = np.array(r["probe_freq"])
    correct = np.array(r["probe_correct"], bool)
    strata = torch.tensor(r["probe_scores"]["MAP entropy"], dtype=torch.float64)
    soft = torch.stack([strata, torch.tensor(r["probe_scores"]["MAP 1-conf"], dtype=torch.float64)], 1)
    person = {n: i for i, n in enumerate(dict.fromkeys(r["probe_name"]))}
    pid = torch.tensor([person[n] for n in r["probe_name"]])
    tasks = {k: (pos(freq), neg(freq)) for k, (pos, neg) in FREQ_TASKS.items()}
    tasks["wrong"] = (freq >= 1) & ~correct, (freq >= 1) & correct
    for t, (pos, neg) in tasks.items():
        if t not in r["tasks"] or not pos.any() or not neg.any():
            continue
        keep = torch.from_numpy(pos | neg)
        y = torch.from_numpy(pos)[keep]
        for k, s in scores.items():
            x = s[keep]
            r["tasks"][t][k] = {"raw": auroc(x[~y], x[y]),
                                "within_entropy": stratified_auroc(y, x, strata[keep], args.n_bins)["weighted"]}
            g = logistic_gain(y, {"f": torch.cat([soft[keep], x[:, None]], 1)}, pid[keep])
            r["logistic"].setdefault(t, {})[f"softmax+{k}"] = g["f"]["auroc"]
        print(f"  {t:7s} " + "  ".join(f"{k} {r['tasks'][t][k]['within_entropy']:.3f} "
                                      f"(raw {r['tasks'][t][k]['raw']:.3f})" for k in scores))
    for k, s in scores.items():
        r["probe_scores"][k] = s.tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(ev, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
