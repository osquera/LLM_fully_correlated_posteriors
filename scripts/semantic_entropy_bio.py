"""Semantic entropy (Farquhar et al. 2024) as a baseline on the bio-format tasks.

For every probe of a bio / PopQA / TOFU corpus, the MAP samples --n-answers
continuations of the probe prompt (temperature --temperature), each cut at
the end of the answer (first newline; with --stop-at-period also the first
". "), and the answers are grouped by meaning:

  exact  same SQuAD-normalised string (bio and PopQA answers are short
         entities: a city, a year, a profession, a name)
  nli    bidirectional entailment under --nli-model, each answer read after
         its prompt (TOFU answers are sentences)

The score is the discrete semantic entropy, the entropy of the cluster
frequencies: no extra model, one generation pass per probe. It is added to an
evaluate_bio.py output (--eval) as the score "semantic entropy": AUROC raw
and within MAP-entropy quintiles on every task, the held-out logistic gain
over the softmax (folds by person, as evaluate_bio.py), and its probe scores,
so bootstrap_bio.py can compare it with the posterior in the same file.

  uv run python scripts/semantic_entropy_bio.py --corpus data/bio_err1/corpus.json \\
      --adapter checkpoints/bio_err1_lora --eval figures/bio_sketch_eval1.json \\
      --stop-at-period --out figures/bio_se_eval1.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_posterior import logistic_gain, stratified_auroc
from evaluate_bio import FREQ_TASKS
from evaluate_posterior import auroc
from llmproj.models import load_finetuned
from llmproj.semantic import entropy, greedy_clusters, normalize_answer

KEY = "semantic entropy"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval", required=True, help="evaluate_bio.py output to add the score to")
    ap.add_argument("--n-answers", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--stop-at-period", action="store_true", help="cut answers at the first '. '")
    ap.add_argument("--cluster", choices=["exact", "nli"], default="exact")
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--batch-size", type=int, default=32, help="prompts per generate call")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def cut(text: str, stop_at_period: bool) -> str:
    text = text.split("\n", 1)[0]
    if stop_at_period:
        text = re.split(r"\.(?:\s|$)", text, maxsplit=1)[0]
    return text.strip()


@torch.no_grad()
def sample_answers(model, tok, prompts, args) -> list[list[str]]:
    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i: i + args.batch_size], return_tensors="pt", padding=True).to(args.device)
        gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                             max_new_tokens=args.max_new_tokens, num_return_sequences=args.n_answers,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j in range(0, len(texts), args.n_answers):
            out.append([cut(t, args.stop_at_period) for t in texts[j: j + args.n_answers]])
    return out


def nli_same(args):
    tok = AutoTokenizer.from_pretrained(args.nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(args.device).eval()
    ent = {v.lower(): k for k, v in nli.config.id2label.items()}["entailment"]

    @torch.no_grad()
    def entails(a: list[str], b: list[str]) -> torch.Tensor:
        enc = tok(a, b, return_tensors="pt", padding=True, truncation=True).to(args.device)
        return nli(**enc).logits.argmax(-1).cpu() == ent

    def make(texts):
        def same(pairs):
            a = [texts[i] for i, _ in pairs]
            b = [texts[j] for _, j in pairs]
            return (entails(a, b) & entails(b, a)).tolist()
        return same
    return make


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    probes = json.loads(Path(args.corpus).read_text())["probes"]
    ev = json.load(open(args.eval))
    r = ev["report"]
    if [q["freq"] for q in probes] != r["probe_freq"]:
        raise SystemExit("eval and corpus probes differ")

    tok = AutoTokenizer.from_pretrained(args.adapter)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, _ = load_finetuned(args.base_model, args.adapter, args.device)
    model.eval()
    answers = sample_answers(model, tok, [q["prompt"] for q in probes], args)
    del model
    make_same = nli_same(args) if args.cluster == "nli" else None

    se, n_clusters = [], []
    for q, ans in zip(probes, answers):
        if make_same is None:
            ids = np.unique([normalize_answer(a) for a in ans], return_inverse=True)[1]
        else:
            texts = [f"{q['prompt']} {a}" for a in ans]
            ids = greedy_clusters(texts, make_same(texts))
        counts = np.bincount(ids)
        se.append(entropy(counts))
        n_clusters.append(len(counts))
    s = torch.tensor(se, dtype=torch.float64)
    print(f"{len(probes)} probes, {args.n_answers} answers each; clusters per probe "
          f"mean {np.mean(n_clusters):.2f}; semantic entropy mean {float(s.mean()):.3f}")

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
        x = s[keep]
        r["tasks"][t][KEY] = {"raw": auroc(x[~y], x[y]),
                              "within_entropy": stratified_auroc(y, x, strata[keep], args.n_bins)["weighted"]}
        gain = logistic_gain(y, {f"softmax+{KEY}": torch.cat([soft[keep], x[:, None]], 1)}, pid[keep])
        r["logistic"].setdefault(t, {})[f"softmax+{KEY}"] = gain[f"softmax+{KEY}"]["auroc"]
        print(f"  {t:7s} within-entropy AUROC {r['tasks'][t][KEY]['within_entropy']:.3f}  "
              f"logistic {r['logistic'][t].get('softmax', float('nan')):.3f} -> "
              f"{r['logistic'][t][f'softmax+{KEY}']:.3f}")
    r["probe_scores"][KEY] = s.tolist()
    r["semantic_entropy"] = {"args": vars(args), "answers": answers, "n_clusters": n_clusters}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(ev, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
