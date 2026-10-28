"""Does a posterior on a *pretrained* model flag what it does not know?

theta_map is the pretrained model itself (a zero-initialised LoRA adapter,
lora_B = 0, so the adapter changes nothing); the posterior lives on the
adapter's lora_B directions and is constrained on a sample of pretraining-like
text (scripts/make_text_corpus.py). No ensemble of the pretrained model exists
here; that is the case the posterior is for.

Closed-book QA, few-shot, greedy answer cut at the first newline:
  PopQA      Wikidata facts with the subject's popularity (Wikipedia page
             views), a ready-made "how often seen" signal
  TriviaQA   harder questions, many wrong answers
Every score is computed on the model's own greedy answer (teacher-forced,
mean over its tokens):
  MAP         entropy, 1 - confidence, NLL of the answer
  semantic entropy   over --n-gen sampled answers (T = 1), clustered by
              normalised string match / containment
  P(True)     1 - p(" True") / (p(" True") + p(" False")) for the proposed answer
  k-NN / Mahalanobis on the answer tokens' hidden states, bank = constraint text
  each samples-file method: MI of the samples (posterior, isotropic)
  --members   a LoRA ensemble (fine-tunes on the constraint text): its MI
Tasks, AUROC raw and within quintiles of the MAP's entropy:
  popqa_rare   subject popularity in the bottom vs top quartile
  popqa_wrong  greedy answer wrong vs right
  trivia_wrong
plus Spearman with -log(popularity), per-popularity-quintile accuracy, and the
held-out logistic test softmax (entropy + 1-conf) -> + log score.
Fit check: per-document NLL change vs theta_map on the constraint text and on
held-out text of the same corpus, against the sampler's tolerance.

  uv run python scripts/evaluate_pretrained_qa.py --base-model HuggingFaceTB/SmolLM2-360M \\
      --adapter checkpoints/pt360_zero --corpus data/fineweb/corpus.json \\
      --samples samples/posterior_pt360.pt --members checkpoints/pt360_ens1 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_posterior import logistic_gain, stratified_auroc
from baselines_single_pass import KNN, Mahalanobis, seq_mean, token_features
from evaluate_bio import per_probe, spearman
from evaluate_posterior import auroc, evaluate_params
from llmproj.data import encode_texts, load_texts
from llmproj.paramspace import ParamSpace
from llmproj.semantic import entropy, normalize_answer
from llmproj.semantic import is_correct

FEW_SHOT = ("Q: What is the capital of France?\nA: Paris\n\n"
            "Q: Who wrote the play Hamlet?\nA: William Shakespeare\n\n"
            "Q: What is the chemical symbol for gold?\nA: Au\n\n"
            "Q: In which country is the city of Osaka?\nA: Japan\n\n"
            "Q: What is George Orwell's occupation?\nA: novelist\n\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--adapter", required=True, help="the zero-initialised adapter (theta_map)")
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--methods", nargs="*", default=None,
                    help="method keys (default: every *_measured one)")
    ap.add_argument("--members", nargs="*", default=[], help="LoRA ensemble adapters")
    ap.add_argument("--corpus", default="data/fineweb/corpus.json")
    ap.add_argument("--fit-split", default="train[:512]", help="the constraint text")
    ap.add_argument("--heldout-split", default="train[1000:1256]")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--n-popqa", type=int, default=2000)
    ap.add_argument("--n-trivia", type=int, default=2000)
    ap.add_argument("--n-gen", type=int, default=10, help="samples for semantic entropy")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/pretrained_qa.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_questions(n_popqa, n_trivia, seed):
    rng = np.random.default_rng(seed)
    qs = []
    if n_popqa:
        ds = load_dataset("akariasai/PopQA", split="test")
        for i in rng.choice(len(ds), size=min(n_popqa, len(ds)), replace=False):
            r = ds[int(i)]
            qs.append({"set": "popqa", "question": r["question"], "golds": json.loads(r["possible_answers"]),
                       "pop": int(r["s_pop"]), "prop": r["prop"]})
    if n_trivia:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        for i in rng.choice(len(ds), size=min(n_trivia, len(ds)), replace=False):
            r = ds[int(i)]
            qs.append({"set": "trivia", "question": r["question"],
                       "golds": list(r["answer"]["aliases"]) + [r["answer"]["value"]]})
    return qs


def prompt_of(q):
    return f"{FEW_SHOT}Q: {q['question']}\nA:"


def cut_at_newline(tokenizer, ids):
    """Generated ids up to (not including) the first token with a newline."""
    out = []
    for t in ids:
        if t == tokenizer.eos_token_id or "\n" in tokenizer.decode([t]):
            break
        out.append(t)
    return out


@torch.no_grad()
def generate(model, tokenizer, prompts, args, n=1, sample=False):
    """n continuations per prompt, cut at the first newline: list of lists of id lists."""
    tokenizer.padding_side = "left"
    out = []
    bs = max(1, args.batch_size // n)
    for i in tqdm(range(0, len(prompts), bs), desc=f"generate n={n}", leave=False):
        enc = tokenizer(prompts[i:i + bs], return_tensors="pt", padding=True).to(args.device)
        g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=sample,
                           temperature=1.0 if sample else None, top_k=0 if sample else None,
                           top_p=1.0 if sample else None, num_return_sequences=n,
                           pad_token_id=tokenizer.pad_token_id)
        g = g[:, enc["input_ids"].shape[1]:].tolist()
        for j in range(0, len(g), n):
            out.append([cut_at_newline(tokenizer, x) for x in g[j:j + n]])
    tokenizer.padding_side = "right"
    return out


def semantic_entropy(texts):
    """Entropy over clusters of answers equal or containing one another after
    normalisation (a cheap stand-in for bidirectional entailment on short answers)."""
    norm = [normalize_answer(t) for t in texts]
    reps, ids = [], []
    for a in norm:
        k = next((c for c, r in enumerate(reps)
                  if a == r or (a and r and (f" {a} " in f" {r} " or f" {r} " in f" {a} "))), None)
        if k is None:
            reps.append(a)
            k = len(reps) - 1
        ids.append(k)
    return entropy(np.bincount(ids).astype(float))


def encode_answers(tokenizer, prompts, answers):
    """prompt + answer ids, loss_mask on the answer, right-padded."""
    rows = []
    for p, a in zip(prompts, answers):
        pi = tokenizer(p).input_ids
        rows.append((pi + a, [0] * len(pi) + [1] * len(a)))
    T = max(len(x) for x, _ in rows)
    ids = torch.full((len(rows), T), tokenizer.pad_token_id)
    att = torch.zeros(len(rows), T, dtype=torch.long)
    lm = torch.zeros(len(rows), T, dtype=torch.long)
    for i, (x, m) in enumerate(rows):
        ids[i, :len(x)], att[i, :len(x)], lm[i, :len(x)] = torch.tensor(x), 1, torch.tensor(m)
    return {"input_ids": ids, "attention_mask": att, "loss_mask": lm}


@torch.no_grad()
def p_false(model, tokenizer, questions, answers, args):
    """1 - P(True) of the proposed answer, zero-shot."""
    t_id = tokenizer(" True", add_special_tokens=False).input_ids[0]
    f_id = tokenizer(" False", add_special_tokens=False).input_ids[0]
    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(questions), args.batch_size):
        texts = [f"Question: {q['question']}\nProposed answer: {a}\n"
                 f"Is the proposed answer true or false?\nAnswer:"
                 for q, a in zip(questions[i:i + args.batch_size], answers[i:i + args.batch_size])]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
        logits = model(**enc).logits[:, -1].float()
        lp = logits[:, [t_id, f_id]].log_softmax(-1)
        out.append(lp[:, 1].exp().cpu())
    tokenizer.padding_side = "right"
    return torch.cat(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()

    blobs = [torch.load(p, weights_only=False) for p in args.samples]
    include = blobs[0]["args"].get("include", ["lora_A", "lora_B"]) if blobs else ["lora_B"]
    space = ParamSpace.from_module(model, include=include, trainable_only=False)
    theta, frozen = space.theta(model), space.frozen(model)
    map_params = {**frozen, **theta}
    sets = {"map": [map_params]}
    tol = None
    for blob in blobs:
        mode = blob["info"]["loss_mode"]
        tol = blob["info"]["alpha_tol_nats"]
        for m, d in blob["methods"].items():
            key = f"{mode}/{m}"
            if (args.methods and key not in args.methods) or (not args.methods and not m.endswith("_measured")):
                continue
            sets[key] = [{**frozen, **{n: theta[n] + u for n, u in
                                       space.unflatten(x.to(space.device, space.dtype)).items()}}
                         for x in d.float()]
    if args.members:
        lora_names = [n for n in map_params if "lora_" in n]
        members = []
        for path in args.members:
            b = AutoModelForCausalLM.from_pretrained(args.base_model, attn_implementation="eager",
                                                     dtype=torch.float32)
            sd = {n: p.detach().to(args.device) for n, p in PeftModel.from_pretrained(b, path).named_parameters()
                  if "lora_" in n}
            if sorted(sd) != sorted(lora_names):
                raise SystemExit(f"{path}: adapter tensors differ from {args.adapter}'s")
            members.append({**map_params, **sd})
        sets["baseline/lora_ensemble"] = members
    print(f"{space}; methods {list(sets)}")

    # ---- questions, greedy answers, correctness ----
    qs = load_questions(args.n_popqa, args.n_trivia, args.seed)
    prompts = [prompt_of(q) for q in qs]
    greedy = [g[0] for g in generate(model, tokenizer, prompts, args)]
    keep = [i for i, g in enumerate(greedy) if g]
    print(f"{len(qs)} questions, {len(qs) - len(keep)} empty answers dropped")
    qs, prompts, greedy = [qs[i] for i in keep], [prompts[i] for i in keep], [greedy[i] for i in keep]
    answers = [tokenizer.decode(g).strip() for g in greedy]
    correct = np.array([is_correct(a, q["golds"]) for a, q in zip(answers, qs)])
    which = np.array([q["set"] for q in qs])
    pop = np.array([q.get("pop", -1) for q in qs])
    for s in ("popqa", "trivia"):
        if (which == s).any():
            print(f"  {s}: accuracy {correct[which == s].mean():.3f} over {(which == s).sum()}")

    # ---- scores ----
    enc = encode_answers(tokenizer, prompts, greedy)
    scores = {}
    for name, params in sets.items():
        _, s, nll = evaluate_params(model, params, enc, args.device, args.batch_size)
        if name == "map":
            scores["MAP entropy"] = per_probe(s["tok"]["entropy"], enc)
            scores["MAP 1-conf"] = per_probe(1 - s["tok"]["conf"], enc)
            scores["MAP NLL"] = nll[0]
        else:
            scores[f"{name} MI"] = per_probe(s["tok"]["mi"], enc)
        print(f"  scored {name}")
    gens = generate(model, tokenizer, prompts, args, n=args.n_gen, sample=True)
    scores["semantic entropy"] = torch.tensor(
        [semantic_entropy([tokenizer.decode(x) for x in g]) for g in gens], dtype=torch.float)
    scores["1 - P(True)"] = p_false(model, tokenizer, qs, answers, args)
    n_hidden = model.config.num_hidden_layers
    layers = args.layers or [n_hidden // 2, n_hidden]
    bank_enc = encode_texts(tokenizer, load_texts(args.corpus, args.fit_split), args.max_length)
    f_bank, _ = token_features(model, bank_enc, layers, args.device, args.batch_size)
    f_ans, _ = token_features(model, enc, layers, args.device, args.batch_size)
    for l in layers:
        scores[f"Mahalanobis L{l}"] = seq_mean(Mahalanobis(f_bank[l], 1e-3, args.device)(f_ans[l]), enc)
        scores[f"k-NN L{l}"] = seq_mean(KNN(f_bank[l], args.knn_k, 50_000, args.device)(f_ans[l]), enc)

    # ---- fit on the constraint text and on held-out text ----
    fit = {}
    for split_name, split in (("constraint", args.fit_split), ("heldout", args.heldout_split)):
        e = encode_texts(tokenizer, load_texts(args.corpus, split), args.max_length)
        map_doc = evaluate_params(model, [map_params], e, args.device, args.batch_size)[2][0]
        for name, params in sets.items():
            if name == "map":
                continue
            dn = evaluate_params(model, params, e, args.device, args.batch_size)[2] - map_doc
            fit.setdefault(name, {})[split_name] = {
                "mean_dnll": float(dn.mean()), "mean_abs_dnll": float(dn.abs().mean()),
                "worst_dnll": float(dn.abs().max()),
                "frac_over_tol": float((dn.abs() > tol).float().mean()) if tol else None}
        fit.setdefault("_map_nll", {})[split_name] = float(map_doc.mean())

    # ---- tasks ----
    is_pop = which == "popqa"
    lo, hi = (np.quantile(pop[is_pop], [0.25, 0.75]) if is_pop.any() else (0, 0))
    tasks = {}
    if is_pop.any():
        tasks["popqa_rare"] = (is_pop & (pop <= lo), is_pop & (pop >= hi))
        tasks["popqa_wrong"] = (is_pop & ~correct, is_pop & correct)
    if (which == "trivia").any():
        tasks["trivia_wrong"] = ((which == "trivia") & ~correct, (which == "trivia") & correct)
    strata = scores["MAP entropy"]
    qid = torch.arange(len(qs))
    soft = torch.stack([scores["MAP entropy"], scores["MAP 1-conf"]], 1).double()
    report = {"n": len(qs), "accuracy": {s: float(correct[which == s].mean()) for s in set(which)},
              "tasks": {}, "logistic": {}, "spearman_popularity": {}, "per_pop_quintile": {},
              "fit": fit, "tol_nats": tol, "layers": layers, "pop_quartiles": [float(lo), float(hi)]}
    for t, (pos, neg) in tasks.items():
        m = torch.from_numpy(pos | neg)
        y = torch.from_numpy(pos)[m]
        report["tasks"][t] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        feats = {"softmax": soft[m]}
        for k, v in scores.items():
            x = v[m]
            report["tasks"][t][k] = {"raw": auroc(x[~y], x[y]),
                                     "within_entropy": stratified_auroc(y, x, strata[m], args.n_bins)["weighted"]}
            if k not in ("MAP entropy", "MAP 1-conf"):
                feats[f"softmax+{k}"] = torch.cat([soft[m], v.double().clamp_min(1e-12).log()[m, None]], 1)
        report["logistic"][t] = {k: r["auroc"] for k, r in logistic_gain(y, feats, qid[m]).items()}
    if is_pop.any():
        lp = -np.log1p(pop[is_pop])
        report["spearman_popularity"] = {k: spearman(v.numpy()[is_pop], lp) for k, v in scores.items()}
        edges = np.quantile(pop[is_pop], np.linspace(0, 1, 6))
        b = np.clip(np.searchsorted(edges[1:-1], pop, side="right"), 0, 4)
        for k in range(5):
            m = is_pop & (b == k)
            report["per_pop_quintile"][k] = {"pop_range": [int(pop[m].min()), int(pop[m].max())],
                                             "acc": float(correct[m].mean()),
                                             **{s: float(v.numpy()[m].mean()) for s, v in scores.items()}}
    report["question_scores"] = {k: v.tolist() for k, v in scores.items()}
    report["questions"] = [{**q, "answer": a, "correct": bool(c)} for q, a, c in zip(qs, answers, correct)]

    # ---- print ----
    keys = list(scores)
    for t, r in report["tasks"].items():
        print(f"\n{t}: {r['n_pos']} positive / {r['n_neg']} negative; AUROC raw / within entropy quintiles")
        for k in keys:
            print(f"  {k:40s} {r[k]['raw']:.3f} / {r[k]['within_entropy']:.3f}")
    print("\nheld-out logistic AUROC, softmax (entropy + 1-conf) -> + log score:")
    for t, r in report["logistic"].items():
        print(f"  {t}: softmax {r['softmax']:.3f}")
        for k, v in r.items():
            if k != "softmax":
                print(f"    {k[8:]:40s} {v:.3f}")
    if report["spearman_popularity"]:
        print("\nSpearman(score, -log popularity), PopQA:")
        for k, v in report["spearman_popularity"].items():
            print(f"  {k:40s} {v:+.3f}")
        print("\nPopQA per popularity quintile: accuracy and mean scores")
        for k, r in report["per_pop_quintile"].items():
            print(f"  {k} pop {r['pop_range'][0]:>7d}-{r['pop_range'][1]:<8d} acc {r['acc']:.3f}  " +
                  "  ".join(f"{s[:18]} {r[s]:.3g}" for s in keys if s.endswith("MI")))
    print(f"\nfit (per-document NLL change vs theta_map; tol {tol}; MAP NLL {fit.get('_map_nll')}):")
    for name, r in fit.items():
        if name.startswith("_"):
            continue
        print(f"  {name:40s} " + "  ".join(
            f"{s}: mean {v['mean_dnll']:+.4f} worst {v['worst_dnll']:.4f} over tol "
            f"{v['frac_over_tol'] if v['frac_over_tol'] is None else round(v['frac_over_tol'], 3)}"
            for s, v in r.items()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "args": vars(args)}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
