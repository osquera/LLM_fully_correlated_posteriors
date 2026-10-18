"""Uncertainty about the meaning of an answer, aleatoric vs epistemic.

Token-level MI cannot tell "several wordings of one answer" from "several
answers": both spread the next-token distribution. Here every model answers
short questions N times, the answers are mapped to meaning, and the spread is
split with the posterior samples (see llmproj.semantic):

  aleatoric  each model alone gives answers that differ in meaning: the
             question has several valid answers, or the model is guessing
  epistemic  the posterior samples disagree on which meaning: the data did
             not pin the answer down ("semantic MI"), corrected for the
             finite-sample bias with a shuffle null

Meaning, two ways: clusters of answers that entail each other both ways under
an NLI model (semantic entropy), and the spread of sentence embeddings.

Questions (each in the Alpaca template the model was fine-tuned on):
  ambigqa   AmbigQA (sewon/ambig_qa): every annotator found the question
            ambiguous (several plausible answers) vs every annotator found one
            answer; balanced. The test of an *aleatoric* detector.
  triviaqa  TriviaQA (rc.nocontext) and
  nq_open   NQ-open: the MAP's greedy answer right or wrong. The test of
            detecting errors.
Models: the MAP (S groups of N answers from one model, so its epistemic part
is the estimator's noise floor) and every method of the samples files.

Writes one record per question to --out; scripts/semantic_tables.py makes
the tables.

  uv run python scripts/semantic_uncertainty.py --n-questions 200
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from llmproj.data import alpaca_prompt
from llmproj.paramspace import ParamSpace
from llmproj.semantic import (cluster_decomposition, embedding_decomposition,
                              greedy_clusters, is_correct, shuffle_corrected)

INSTRUCTION = "Answer the question with a short phrase."
# Worked examples in the same template, before the real question: without
# them a 135M model rarely answers briefly, and every sampled answer is its own
# meaning cluster, which saturates the entropies.
FEW_SHOT = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("How many legs does a spider have?", "Eight"),
    ("In which year did the Berlin Wall fall?", "1989"),
    ("What is the chemical symbol for gold?", "Au"),
]


def make_prompt(question: str, n_shot: int) -> str:
    shots = "".join(alpaca_prompt(INSTRUCTION, q) + a + "\n\n" for q, a in FEW_SHOT[:n_shot])
    return shots + alpaca_prompt(INSTRUCTION, question)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+",
                    default=["samples/posterior_sequence.pt", "samples/posterior_token.pt"])
    ap.add_argument("--methods", nargs="+", default=["projected_measured", "isotropic_measured"],
                    help="method keys of the samples files (the MAP is always included)")
    ap.add_argument("--datasets", nargs="+", default=["ambigqa", "triviaqa", "nq_open"])
    ap.add_argument("--n-questions", type=int, default=200, help="per dataset")
    ap.add_argument("--n-posterior", type=int, default=8, help="S: posterior samples per method")
    ap.add_argument("--n-answers", type=int, default=10, help="N: answers per posterior sample")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="at 1.0 a 135M model gives nearly all-distinct answers, saturating the entropies")
    ap.add_argument("--few-shot", type=int, default=len(FEW_SHOT), help="worked examples in the prompt")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-questions", type=int, default=16)
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--nli-prefilter", type=float, default=0.5,
                    help="only run NLI on answer pairs with embedding cosine >= this; "
                         "-1 runs it on every pair")
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--out", default="figures/semantic_uncertainty.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=None,
                    help="only these models, e.g. map (default: all)")
    return ap.parse_args()


# ---------------------------------------------------------------- questions

def load_questions(name: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if name == "ambigqa":
        ds = load_dataset("sewon/ambig_qa", "light", split="validation")
        amb, unamb = [], []
        for r in ds:
            types = set(r["annotations"]["type"])
            if types == {"multipleQAs"}:
                golds = [a for qa in r["annotations"]["qaPairs"] for ans in qa["answer"] for a in ans]
                amb.append({"id": r["id"], "question": r["question"], "golds": golds, "ambiguous": 1})
            elif types == {"singleAnswer"}:
                golds = [a for ans in r["annotations"]["answer"] for a in ans]
                unamb.append({"id": r["id"], "question": r["question"], "golds": golds, "ambiguous": 0})
        rows = rng.sample(amb, min(n // 2, len(amb))) + rng.sample(unamb, min(n - n // 2, len(unamb)))
    elif name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        idx = rng.sample(range(len(ds)), n)
        rows = [{"id": ds[i]["question_id"], "question": ds[i]["question"],
                 "golds": ds[i]["answer"]["aliases"]} for i in idx]
    elif name == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        idx = rng.sample(range(len(ds)), n)
        rows = [{"id": str(i), "question": ds[i]["question"], "golds": ds[i]["answer"]} for i in idx]
    else:
        raise ValueError(name)
    for r in rows:
        r["dataset"] = name
    return rows


# ---------------------------------------------------------------- generation

def first_line(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()
    return ""


@torch.no_grad()
def generate(model, tokenizer, prompts, args, greedy: bool):
    """Greedy: (answers, mean token entropy, mean token NLL) per prompt, both
    over the answer's first line. Sampled: N answers per prompt."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
    kw = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if not greedy:
        out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_k=0,
                             top_p=1.0, num_return_sequences=args.n_answers, **kw)
        texts = tokenizer.batch_decode(out[:, enc.input_ids.size(1):], skip_special_tokens=True)
        texts = [first_line(t) for t in texts]
        return [texts[i * args.n_answers:(i + 1) * args.n_answers] for i in range(len(prompts))]

    out = model.generate(**enc, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, **kw)
    gen = out.sequences[:, enc.input_ids.size(1):]
    logp = torch.stack(out.scores, 1).float().log_softmax(-1)  # (B, T, V)
    ent = -(logp.exp() * logp).sum(-1)
    nll = -logp.gather(-1, gen[..., None])[..., 0]
    answers, ents, nlls = [], [], []
    for b in range(len(prompts)):
        keep, seen = [], False
        for t, tok in enumerate(gen[b].tolist()):
            piece = tokenizer.decode([tok])
            if tok == tokenizer.eos_token_id or (seen and "\n" in piece):
                break
            seen |= bool(piece.strip())
            keep.append(t)
        keep = keep or [0]
        answers.append(first_line(tokenizer.decode(gen[b], skip_special_tokens=True)))
        ents.append(float(ent[b, keep].mean()))
        nlls.append(float(nll[b, keep].mean()))
    return answers, ents, nlls


# ---------------------------------------------------------------- meaning

class Meaning:
    """Sentence embeddings and bidirectional entailment, with a cache."""

    def __init__(self, args):
        self.args, dev = args, args.device
        self.etok = AutoTokenizer.from_pretrained(args.embed_model)
        self.emodel = AutoModel.from_pretrained(args.embed_model).to(dev).eval()
        self.ntok = AutoTokenizer.from_pretrained(args.nli_model)
        self.nmodel = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(dev).eval()
        labels = {v.lower(): k for k, v in self.nmodel.config.id2label.items()}
        self.entail = next(k for name, k in labels.items() if "entail" in name)
        self.cache: dict[tuple[str, str], bool] = {}
        self.n_nli = 0

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            enc = self.etok([t or "(no answer)" for t in texts[i:i + 256]], padding=True,
                            truncation=True, max_length=64, return_tensors="pt").to(self.args.device)
            h = self.emodel(**enc).last_hidden_state
            m = enc.attention_mask[..., None].float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            enc = self.ntok([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                            truncation=True, max_length=128, return_tensors="pt").to(self.args.device)
            pred = self.nmodel(**enc).logits.argmax(-1).tolist()
            self.cache.update({p: k == self.entail for p, k in zip(chunk, pred)})
            self.n_nli += len(chunk)
        return [self.cache[p] for p in pairs]

    def cluster(self, question: str, texts: list[str], emb: np.ndarray) -> np.ndarray:
        """Same meaning = entailment both ways, of "question answer" pairs."""
        full = [f"{question} {t or '(no answer)'}" for t in texts]

        def same(pairs):
            fwd = self.entails([(full[i], full[j]) for i, j in pairs])
            bwd = self.entails([(full[j], full[i]) for i, j in pairs])
            return [a and b for a, b in zip(fwd, bwd)]

        cand = None
        if self.args.nli_prefilter > -1:
            cand = lambda i, j: float(emb[i] @ emb[j]) >= self.args.nli_prefilter
        return greedy_clusters(texts, same, cand)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    questions = [q for name in args.datasets for q in load_questions(name, args.n_questions, args.seed)]
    prompts = [make_prompt(q["question"], args.few_shot) for q in questions]
    print({name: sum(q["dataset"] == name for q in questions) for name in args.datasets})

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    params = dict(model.named_parameters())
    theta0 = {n: params[n].detach().clone() for n in space.names}

    def set_delta(delta):
        upd = space.unflatten(delta.to(device=space.device, dtype=space.dtype))
        with torch.no_grad():
            for n in space.names:
                params[n].copy_(theta0[n] + upd[n])

    # model name -> (S, P) deltas; the MAP repeats zero, as S groups of N answers
    deltas = {"map": torch.zeros(args.n_posterior, space.P)}
    for p in args.samples:
        if not Path(p).exists():
            print(f"skipping missing {p}")
            continue
        blob = torch.load(p, weights_only=False)
        mode = blob["info"]["loss_mode"]
        for m in args.methods:
            if m in blob["methods"]:
                deltas[f"{mode}/{m}"] = blob["methods"][m][: args.n_posterior]
    if args.models:
        deltas = {k: v for k, v in deltas.items() if k in args.models}
    print(f"models: {list(deltas)}")

    B = args.batch_questions
    gen = {}  # model -> {"answers": [Q][S][N], "greedy": [Q], "tok_entropy", "tok_nll"}
    for name, ds in deltas.items():
        g = {"answers": [[] for _ in questions]}
        for s, d in enumerate(ds):
            set_delta(d)
            torch.manual_seed(args.seed * 1000 + s)
            for i in range(0, len(prompts), B):
                for k, a in enumerate(generate(model, tokenizer, prompts[i:i + B], args, greedy=False)):
                    g["answers"][i + k].append(a)
        set_delta(torch.zeros(space.P))
        if name == "map":  # greedy answers and token-level baselines, from the MAP only
            g["greedy"], g["tok_entropy"], g["tok_nll"] = [], [], []
            for i in range(0, len(prompts), B):
                a, e, n = generate(model, tokenizer, prompts[i:i + B], args, greedy=True)
                g["greedy"] += a
                g["tok_entropy"] += e
                g["tok_nll"] += n
        gen[name] = g
        print(f"generated {name}: {len(ds)} x {args.n_answers} answers x {len(questions)} questions "
              f"({time.time() - t0:.0f} s)")

    del model, base
    torch.cuda.empty_cache()
    meaning = Meaning(args)

    records = []
    for qi, q in enumerate(questions):
        # cluster every model's answers jointly, so ids are shared across models
        pooled = [a for name in deltas for row in gen[name]["answers"][qi] for a in row]
        texts = list(dict.fromkeys(pooled))
        emb = meaning.embed(texts)
        ids = meaning.cluster(q["question"], texts, emb)
        where = {t: i for i, t in enumerate(texts)}

        greedy = gen["map"]["greedy"][qi]
        rec = {**q, "map_greedy": greedy, "map_greedy_correct": is_correct(greedy, q["golds"]),
               "map_tok_entropy": gen["map"]["tok_entropy"][qi],
               "map_tok_nll": gen["map"]["tok_nll"][qi], "models": {}}
        for name in deltas:
            ans = gen[name]["answers"][qi]  # S x N strings
            pos = np.array([[where[a] for a in row] for row in ans])
            labels = ids[pos]
            rec["models"][name] = {
                "answers": ans,
                "clusters": labels.tolist(),
                "n_clusters": int(len(np.unique(labels))),
                "sample_acc": float(np.mean([is_correct(a, q["golds"]) for row in ans for a in row])),
                "sem": shuffle_corrected(labels, cluster_decomposition, args.n_shuffles, args.seed),
                "emb": shuffle_corrected(emb[pos], embedding_decomposition, args.n_shuffles, args.seed),
            }
        records.append(rec)
        if (qi + 1) % 50 == 0:
            print(f"scored {qi + 1}/{len(questions)} questions, {meaning.n_nli} NLI calls "
                  f"({time.time() - t0:.0f} s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "prompt_example": prompts[0],
                               "models": list(deltas), "records": records}))
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
