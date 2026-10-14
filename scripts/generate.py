"""Try a posterior by hand: generate from the Bayesian model average, or score a
given text, and show how uncertain the model is at every token.

At each step, every posterior sample s gives a next-token distribution p_s;
the prediction is their mean p_bar (the BMA). Per token we report

  p      p_bar of the chosen (or given) token
  H      entropy of p_bar: total uncertainty
  MI     H(p_bar) - mean_s H(p_s): the epistemic part, i.e. how much the
         samples disagree. Zero for the MAP.

Tokens are coloured by MI (or --color-by): green low, yellow, red high,
relative to the largest value in the output.

Examples:
  # generate from an Alpaca-style instruction
  uv run python scripts/generate.py --instruction "Name three primary colours."

  # score a text you supply (teacher forcing), e.g. something off-distribution
  uv run python scripts/generate.py --raw --score "The mitochondria is the powerhouse of the cell."

  # another posterior / scale; --scale multiplies the saved samples
  uv run python scripts/generate.py --samples samples/posterior_token.pt \\
      --method projected_measured --scale 2 --instruction "..."
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmproj.data import alpaca_prompt
from llmproj.paramspace import ParamSpace



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples/posterior_sequence.pt")
    ap.add_argument("--method", default="projected_measured",
                    help="a key of the samples file's methods, or 'map'")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every sample (scale^2 divides alpha)")
    ap.add_argument("--n-samples", type=int, default=None, help="use only the first k")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--input", default=None, help="optional Alpaca input field")
    ap.add_argument("--raw", action="store_true", help="use the prompt text as is, no template")
    ap.add_argument("--prompt", default="", help="raw prompt (with --raw)")
    ap.add_argument("--score", default=None,
                    help="score this continuation instead of generating")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy on p_bar")
    ap.add_argument("--color-by", choices=["mi", "entropy", "p"], default="mi")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--adapter", default="checkpoints/smollm2_lora")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def color(text: str, x: float) -> str:
    """x in [0, 1]: green -> yellow -> red."""
    code = 32 if x < 1 / 3 else 33 if x < 2 / 3 else 31
    return f"\033[{code}m{text}\033[0m"


@torch.no_grad()
def bma_step(model, params, ids, caches):
    """Next-token p_bar and per-sample entropies, advancing every sample's cache."""
    probs, ents = [], []
    for s, prm in enumerate(params):
        inp = ids if caches[s] is None else ids[:, -1:]
        out = functional_call(model, prm, args=(), kwargs={
            "input_ids": inp, "past_key_values": caches[s], "use_cache": True})
        caches[s] = out.past_key_values
        logp = F.log_softmax(out.logits[0, -1].float(), -1)
        probs.append(logp.exp())
        ents.append(-(logp.exp() * logp).sum())
    p_bar = torch.stack(probs).mean(0)
    return p_bar, torch.stack(ents).mean()


@torch.no_grad()
def bma_score(model, params, ids):
    """p_bar at every position of ``ids`` (teacher forcing), and mean_s H(p_s)."""
    p_bar, h_mean = 0.0, 0.0
    for prm in params:
        out = functional_call(model, prm, args=(), kwargs={"input_ids": ids})
        logp = F.log_softmax(out.logits[0, :-1].float(), -1)
        p_bar = p_bar + logp.exp()
        h_mean = h_mean - (logp.exp() * logp).sum(-1)
    return p_bar / len(params), h_mean / len(params)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, attn_implementation="eager", dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(args.device).eval()
    space = ParamSpace.from_module(model, include=["lora_A", "lora_B"], trainable_only=False)
    theta = space.theta(model)
    frozen = space.frozen(model)

    if args.method == "map":
        deltas = torch.zeros(1, space.P)
    else:
        blob = torch.load(args.samples, weights_only=False)
        if args.method not in blob["methods"]:
            raise SystemExit(f"{args.method!r} not in {list(blob['methods'])}")
        deltas = blob["methods"][args.method][: args.n_samples] * args.scale
    params = []
    for d in deltas:
        upd = space.unflatten(d.to(device=space.device, dtype=space.dtype))
        params.append({**frozen, **{n: theta[n] + u for n, u in upd.items()}})
    print(f"{args.method}: {len(params)} samples, mean ||delta|| = "
          f"{float(deltas.norm(dim=1).mean()):.3g}\n")

    if args.raw:
        prompt = args.prompt
    elif args.instruction is None:
        raise SystemExit("pass --instruction, or --raw with --prompt / --score")
    else:
        prompt = alpaca_prompt(args.instruction, args.input or "")
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)

    rows = []  # (token text, p, H, MI)
    if args.score is not None:
        cont = tokenizer(args.score, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([prompt_ids, cont.to(args.device)], 1)
        if ids.size(1) < 2:
            raise SystemExit("need at least two tokens to score")
        p_bar, h_mean = bma_score(model, params, ids)
        start = max(prompt_ids.size(1) - 1, 0)
        for t in range(start, ids.size(1) - 1):
            pb, y = p_bar[t], ids[0, t + 1]
            h = float(-(pb * pb.clamp_min(1e-30).log()).sum())
            rows.append((tokenizer.decode(y), float(pb[y]), h, max(h - float(h_mean[t]), 0.0)))
    else:
        ids, caches = prompt_ids, [None] * len(params)
        for _ in range(args.max_new_tokens):
            pb, h_mean = bma_step(model, params, ids, caches)
            if args.temperature > 0:
                y = torch.multinomial((pb.log() / args.temperature).softmax(-1), 1)[0]
            else:
                y = pb.argmax()
            h = float(-(pb * pb.clamp_min(1e-30).log()).sum())
            rows.append((tokenizer.decode(y), float(pb[y]), h, max(h - float(h_mean), 0.0)))
            if y.item() == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, y.view(1, 1)], 1)

    key = {"p": 1, "entropy": 2, "mi": 3}[args.color_by]
    vals = [r[key] for r in rows]
    top = max(vals) or 1.0
    print(prompt, end="")
    for r, v in zip(rows, vals):
        x = 1 - v if args.color_by == "p" else v / top
        print(color(r[0], x), end="")
    print(f"\n\n(coloured by {args.color_by}: green low, red high"
          f"{'' if args.color_by == 'p' else f', relative to max {top:.3g}'})\n")

    print(f"{'token':>16s} {'p':>7s} {'H':>7s} {'MI':>9s}")
    for tok, p, h, mi in rows:
        print(f"{tok!r:>16s} {p:7.3f} {h:7.3f} {mi:9.2e}")
    n = len(rows)
    print(f"\nmean over {n} tokens: p {sum(r[1] for r in rows) / n:.3f}  "
          f"H {sum(r[2] for r in rows) / n:.3f}  MI {sum(r[3] for r in rows) / n:.3e}")


if __name__ == "__main__":
    main()
